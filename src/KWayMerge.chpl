module KWayMerge {
  config const TREE_MERGE: bool = false; 

  record KeysComparator {
    const dummy;
    inline proc key(k) { return k; }
  }

  record KeysRanksComparator {
    const dummy;
    inline proc key(kr) { const (k, _) = kr; return k; }
  }

  proc mergeSortedChunks(dst: [?aD] ?keyType, src: [aD] keyType, chunks: int) {
    merge(dst, src, chunks, new KeysComparator(max(keyType)));
  }

  proc mergeSortedRanks(dst: [?aD] ?t, src: [aD] t, chunks: int) where isTuple(t) {
    merge(dst, src, chunks, new KeysRanksComparator(max(t)));
  }

  private proc merge(dst: [?aD] ?t, src: [aD] t, chunks: int, comparator) {
    if TREE_MERGE {
      treeMerge(dst, src, chunks, comparator);
    } else {
      directMerge(dst, src, chunks, comparator);
    }
  }

  private proc directMerge(dst: [?aD] ?t, src: [aD] t, chunks: int, comparator) {
    var cNextIdx: [0..<chunks] int;
    var cLastIdx: [0..<chunks] int;
    var draw: [0..<chunks] t;
    for tid in 0..<chunks {
      const iters = computeChunk(aD.dim(0), tid, chunks);
      cNextIdx[tid] = iters.first;
      cLastIdx[tid] = iters.last;
      draw[tid] = src[cNextIdx[tid]];
      cNextIdx[tid] += 1;
    }
    for i in 0..#aD.size {
      var minA = draw[0];
      var minALoc: int = 0;
      for j in 1..<chunks {
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

  private proc treeMerge(dst: [?aD] ?t, src: [aD] t, chunks: int, comparator) {
    var cNextIdx: [0..<chunks] int;
    var cLastIdx: [0..<chunks] int;
    for chunkId in 0..<chunks {
      const iters = computeChunk(aD.dim(0), chunkId, chunks);
      //writeln("I think chunk ", chunkId, " is ", iters);
      cNextIdx[chunkId] = iters.first;
      cLastIdx[chunkId] = iters.last;
    }

    const numLevels = ceil(log2(chunks:real)):int;
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
        if chunkId < chunks {
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
