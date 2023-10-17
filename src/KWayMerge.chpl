module KWayMerge {
  use GPUIterator;
  use Math;

  config const TREE_MERGE: bool = false; 

  record KeysComparator {
    const dummy;
    inline proc key(k) { return k; }
  }

  record KeysRanksComparator {
    const dummy;
    inline proc key(kr) { const (k, _) = kr; return k; }
  }

  proc mergeSortedKeys(dst: [?aD] ?keyType, src: [aD] keyType, localesToMerge: [0..1] (locale,int), pivot: int) {
    coforall (loc,cut) in localesToMerge do on loc {
      var localSub = aD.localSubdomain();
      var dstLocal = dst.localSlice(localSub);
      var srcLocal = src.localSlice(localSub);
      //writeln("at ", loc, " merging ", localSub.dim(0).first,"..",(cut-1)," and ", cut, "..", localSub.dim(0).last);
      var chunks = [localSub.dim(0).first..<cut, cut..localSub.dim(0).last];
      directMerge(dst, src, chunks, new KeysComparator(max(keyType)));
    }
    //writeln("now A = ", dst);
  }

  proc mergeSortedKeys(ref dst: [?aD] ?keyType, ref src: [aD] keyType, numChunks: int) {
    mergeChunks(dst, src, numChunks, new KeysComparator(max(keyType)));
  }

  proc mergeSortedRanks(ref dst: [?aD] ?t, ref src: [aD] t, numChunks: int) where isTuple(t) {
    mergeChunks(dst, src, numChunks, new KeysRanksComparator(max(t)));
  }

  private proc mergeChunks(ref dst: [?aD] ?t, ref src: [aD] t, numChunks: int, comparator) {
    if TREE_MERGE {
      treeMerge(dst, src, numChunks, comparator);
    } else {
      directMerge(dst, src, numChunks, comparator);
    }
  }

  private proc directMerge(ref dst: [?aD] ?t, ref src: [aD] t, numChunks: int, comparator) {
    var cNextIdx: [0..<numChunks] int;
    var cLastIdx: [0..<numChunks] int;
    var chunks: [0..<numChunks] range(int);
    var draw: [0..<numChunks] t;
    for tid in 0..<numChunks {
      chunks[tid] = computeChunk(aD.dim(0), tid, numChunks);
    }
    directMerge(dst, src, chunks, comparator);
  }

  private proc directMerge(ref dst: [?aD] ?t, ref src: [aD] t, chunks: [?chunkD] range(int), comparator) {
    var numChunks = chunkD.size;
    var cNextIdx: [0..<numChunks] int;
    var cLastIdx: [0..<numChunks] int;
    var draw: [0..<numChunks] t;
    var totalSize: int = 0;
    for tid in 0..<numChunks {
      cNextIdx[tid] = chunks[tid].first;
      cLastIdx[tid] = chunks[tid].last;
      totalSize += chunks[tid].last - chunks[tid].first + 1;
      draw[tid] = src[cNextIdx[tid]];
      cNextIdx[tid] += 1;
    }
    for i in chunks[0].first..#totalSize {
      var minA = draw[0];
      var minALoc: int = 0;
      for j in 1..<numChunks {
        if (comparator.key(draw[j]) < comparator.key(minA)) {
          minA = draw[j];
          minALoc = j;
        }
      }
      dst[i] = minA;
      const next = cNextIdx[minALoc];
      if (next > cLastIdx[minALoc]) {
        draw[minALoc] = comparator.dummy;
      } else {
        draw[minALoc] = src[next];
      }
      cNextIdx[minALoc] = next + 1;
    }
  }

  private proc treeMerge(ref dst: [?aD] ?t, ref src: [aD] t, numChunks: int, comparator) {
    var cNextIdx: [0..<numChunks] int;
    var cLastIdx: [0..<numChunks] int;
    for chunkId in 0..<numChunks {
      const iters = computeChunk(aD.dim(0), chunkId, numChunks);
      //writeln("I think chunk ", chunkId, " is ", iters);
      cNextIdx[chunkId] = iters.first;
      cLastIdx[chunkId] = iters.last;
    }

    const numLevels = ceil(log2(numChunks:real)):int;
    const treeSize = 2**numLevels;
    //writeln("numLevels = ", numLevels, " treeSize = ", treeSize);

    // element 1 is parent of tree, 2 and 3 are left and right children of parent, etc.
    var tree: [0..#treeSize] (t, int);

    proc replayGames(node: int, contestant: (t, int)) {
      //writeln("replayGames(", node, ",", contestant, ")");
      var loser, winner: (t, int);
      if comparator.key(contestant(0)) < comparator.key(tree[node](0)) {
        loser = tree[node];
        winner = contestant;
      } else {
        loser = contestant;
        winner = tree[node];
      }
      //writeln("  winner ", winner, " loser ", loser);
      tree[node] = loser;
      if node != 1 {
        replayGames(node / 2, winner);
      } else {
      tree[0] = winner;
      }
    }

    proc buildTree(level: int, idx: int): (t, int) {
      if (level == numLevels) {
        // leaf node
        const chunkId = idx-treeSize;
        if chunkId < numChunks {
        //writeln("build tree bottom level, idx = ", idx, " real idx = ", chunkId, " src[cNextIdx[chunkId]] = ", src[cNextIdx[chunkId]]);
        return (src[cNextIdx[chunkId]], chunkId);
        } else {
        return (comparator.dummy, chunkId);
        }
      } else {
        // interior node
        const leftIdx = idx*2;
        const rightIdx = leftIdx+1;
        const left = buildTree(level+1, leftIdx);
        const right = buildTree(level+1, rightIdx);
        if (comparator.key(right(0)) < comparator.key(left(0))) {
          //writeln("buildTree level ", level, " idx ", idx, " loser idx ", left(1), " loser val ", left(0));
          tree[idx] = left;
          return right;
        } else {
          //writeln("buildTree level ", level, " idx ", idx, " loser idx ", right(1), " loser val ", right(0));
          tree[idx] = right;
          return left;
        }
      }
    }

    var winner = buildTree(0, 1);

    var i:int = 0;
    while winner(0) != comparator.dummy {
      //writeln("winner(", i, ") = ", winner); 
      dst[i] = winner(0);
      i = i+1;

      cNextIdx[winner(1)] += 1;
      var newVal: t;
      if (cNextIdx[winner(1)] > cLastIdx[winner(1)]) {
        newVal = comparator.dummy;
      } else {
        newVal = src[cNextIdx[winner(1)]];
      }
      const newContestant = (newVal, winner(1));

      replayGames((winner(1)+treeSize)/2, newContestant);
      winner = tree[0];
    }
  }
}
