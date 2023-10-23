module KWayMerge {
  use GPUIterator;
  use Math;
  use Time;

  config const TREE_MERGE: bool = false; 

  record KeysComparator {
    const dummy;
    inline proc key(k) { return k; }
  }

  record KeysRanksComparator {
    const dummy;
    inline proc key(kr) { const (k, _) = kr; return k; }
  }

  proc mergeSortedKeysAtCut(ref dst: [?aD] ?keyType, const ref src: [aD] keyType, cut: int) {
    var localSub = aD.localSubdomain();
    //writeln(dateTime.now(), " at ", here, " merging ", localSub.dim(0).first..<cut," and ", cut..localSub.dim(0).last);
    //writeln(dateTime.now(), " at ", here, " src ", src[localSub]);
    var chunks = (localSub.dim(0).first..<cut, cut..localSub.dim(0).last);
    mergeTwoChunks(dst[localSub], src[localSub], chunks, new KeysComparator(max(keyType)));
    //writeln(dateTime.now(), " at ", here, " dst ", dst.localSlice(localSub));
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

  private proc mergeTwoChunks(ref dst: [?aD] ?t, const ref src: [aD] t, chunks: (range(int), range(int)), comparator) {
    var next0 = chunks(0).first;
    var next1 = chunks(1).first;
    const last0 = chunks(0).last;
    const last1 = chunks(1).last;
    //writeln(dateTime.now(), " at ", here, " merging next0 ", next0, " next1 ", next1, " last0 ", last0, " last1 ", last1);
    var i = chunks(0).first;
    if next0 <= last0 && next1 <= last1 {
      var elem0 = src.localAccess[next0];
      var elem1 = src.localAccess[next1];
      while true {
        if (elem0 < elem1) {
          //writeln(here, " next0 ", next0);
          dst.localAccess[i] = elem0;
          next0 += 1;
          i += 1;
          if next0 > last0 then break;
          elem0 = src.localAccess[next0];
        } else {
          //writeln(here, " next1 ", next1);
          dst.localAccess[i] = elem1;
          next1 += 1;
          i += 1;
          if next1 > last1 then break;
          elem1 = src.localAccess[next1];
        }
      }
    }
    if next0 <= last0 {
      //writeln(dateTime.now(), " at ", here, " copying next0 ", next0..last0, " to dst ", i..last1);
      [(a, b) in zip(i..last1, next0..last0)] dst.localAccess[a] = src.localAccess[b];
    }
    if next1 <= last1 {
      //writeln(dateTime.now(), " at ", here, " copying next1 ", next1..last1, " to dst ", i..last1);
      [(a, b) in zip(i..last1, next1..last1)] dst.localAccess[a] = src.localAccess[b];
    }
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
      draw[tid] = src.localAccess[cNextIdx[tid]];
      cNextIdx[tid] += 1;
    }
    //writeln(dateTime.now(), " at ", here, " merging next index ", cNextIdx, " totalSize ", totalSize);
    for i in chunks[0].first..#totalSize {
      var minA = draw[0];
      var minALoc: int = 0;
      for j in 1..<numChunks {
        if (comparator.key(draw[j]) < comparator.key(minA)) {
          minA = draw[j];
          minALoc = j;
        }
      }
      dst.localAccess[i] = minA;
      const next = cNextIdx[minALoc];
      if (next > cLastIdx[minALoc]) {
        draw[minALoc] = comparator.dummy;
      } else {
        draw[minALoc] = src.localAccess[next];
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
