module CUBRadixSort {
    use MultiTypeSymEntry;
    use GPUIterator;
    use GPUAPI;
    use CTypes;
    use KWayMerge;

    config param verbose = false;

    extern proc cubSortKeys_int32(keys_in: c_void_ptr, keys_out: c_void_ptr, N: c_size_t);
    extern proc cubSortKeys_int64(keys_in: c_void_ptr, keys_out: c_void_ptr, N: c_size_t);
    extern proc cubSortKeys_float(keys_in: c_void_ptr, keys_out: c_void_ptr, N: c_size_t);
    extern proc cubSortKeys_double(keys_in: c_void_ptr, keys_out: c_void_ptr, N: c_size_t);
    extern proc cubSortPairs_int32(keys_in: c_void_ptr, keys_out: c_void_ptr, values_in: c_void_ptr, values_out: c_void_ptr, N: c_size_t);
    extern proc cubSortPairs_int64(keys_in: c_void_ptr, keys_out: c_void_ptr, values_in: c_void_ptr, values_out: c_void_ptr, N: c_size_t);
    extern proc cubSortPairs_float(keys_in: c_void_ptr, keys_out: c_void_ptr, values_in: c_void_ptr, values_out: c_void_ptr, N: c_size_t);
    extern proc cubSortPairs_double(keys_in: c_void_ptr, keys_out: c_void_ptr, values_in: c_void_ptr, values_out: c_void_ptr, N: c_size_t);

    proc cubSortKeys(type t, devA: GPUArray, devAOut: GPUArray, N: int) {
        if t == int(32) {
            cubSortKeys_int32(devA.dPtr(), devAOut.dPtr(), N: c_size_t);
        } else if t == int(64) {
            cubSortKeys_int64(devA.dPtr(), devAOut.dPtr(), N: c_size_t);
        } else if t == real(32) {
            cubSortKeys_float(devA.dPtr(), devAOut.dPtr(), N: c_size_t);
        } else if t == real(64) {
            cubSortKeys_double(devA.dPtr(), devAOut.dPtr(), N: c_size_t);
        }
        DeviceSynchronize();
        
        if !disableMultiGPUs || nGPUs > 1 {
            devAOut.fromDevice();
        }
    }

    proc cubSortPairs(type t, devA: GPUArray, devAOut: GPUArray, devRanksIn: GPUArray, devRanksOut: GPUArray, N: int) {
        devRanksIn.toDevice();
        if t == int(32) {
            cubSortPairs_int32(devA.dPtr(), devAOut.dPtr(), devRanksIn.dPtr(), devRanksOut.dPtr(), N: c_size_t);
        } else if t == int(64) {
            cubSortPairs_int64(devA.dPtr(), devAOut.dPtr(), devRanksIn.dPtr(), devRanksOut.dPtr(), N: c_size_t);
        } else if t == real(32) {
            cubSortPairs_float(devA.dPtr(), devAOut.dPtr(), devRanksIn.dPtr(), devRanksOut.dPtr(), N: c_size_t);
        } else if t == real(64) {
            cubSortPairs_double(devA.dPtr(), devAOut.dPtr(), devRanksIn.dPtr(), devRanksOut.dPtr(), N: c_size_t);
        }
        DeviceSynchronize();
        /*
        if nGPUS > 1 {
            MergePartitions(device_buffers, elements, gpus, num_fillers);
        }
        */
        
        if !disableMultiGPUs || nGPUs > 1 {
            devAOut.fromDevice();
        }
        devRanksOut.fromDevice();
    }

    // TODO update SortMsg to call the SymEntry version of this proc
    proc cubRadixSortLSD_keys(a: [?aD] ?t) {
        var aEntry = new SymEntry(a);
        return cubRadixSortLSD_keys(aEntry);
    }

    // TODO update ArgSortMsg to call the SymEntry version of this proc
    proc cubRadixSortLSD_ranks(a: [?aD] ?t) {
        var aEntry = new SymEntry(a);
        var ranksEntry = cubRadixSortLSD_ranks(aEntry);
        var ranks = ranksEntry.a;
        return ranks;
    }

    /* Radix Sort Least Significant Digit
       radix sort a block distributed array,
       returning a sorted array, without updating the input
     */
    proc cubRadixSortLSD_keys(aEntry: SymEntry) {
        aEntry.createDeviceCache();

        var a = aEntry.a;
        var aD = a.domain;
        type t = aEntry.etype;

        var aOut: [aD] t;

        // TODO: proper lambda functions break Chapel compiler
        record Lambda {
            proc this(lo: int, hi: int, N: int) {
                var deviceId: int(32);
                GetDevice(deviceId);
                if (verbose) {
                    var count: int(32);
                    GetDeviceCount(count);
                    writeln("In cubSortKeysCallback, launching the CUDA kernel with a range of ", lo, "..", hi, " (Size: ", N, "), GPU", deviceId, " of ", count, " @", here);
                }
                aEntry.toDevice(deviceId);
                var devA = aEntry.getDeviceArray(deviceId);
                // these are temporary arrays that do not need to be cached on SymEntry
                var devAOut = new GPUArray(aOut.localSlice(lo .. hi));
                cubSortKeys(t, devA, devAOut, N);
            }
        }
        // get local domain's indices
        var lD = aD.localSubdomain();
        // calc task's indices from local domain's indices
        var tD = {lD.low..lD.high};
        var cubSortKeysCallback = new Lambda();
        forall i in GPU(tD, cubSortKeysCallback) {
            writeln("Should not reach this point!");
            exit(1);
        }

        if disableMultiGPUs || nGPUs == 1 {
            // no need to merge
            return aOut;
        } else {
            // merge sorted chunks
            var merged: [aD] t;
            mergeSortedKeys(aOut, merged, nGPUs);
            return merged;
        }
    }

    /* Radix Sort Least Significant Digit
       radix sort a block distributed array
       returning a permutation vector as a block distributed array */
    proc cubRadixSortLSD_ranks(aEntry: SymEntry) {
        aEntry.createDeviceCache();

        var a = aEntry.a;
        var aD = a.domain;
        type t = aEntry.etype;

        var ranksIn: [aD] int = [rank in aD] rank;
        var aOut: [aD] t;
        var ranksOut: [aD] int;
        var ranksEntry = new shared SymEntry(ranksOut);
        ranksEntry.createDeviceCache();

        // TODO: proper lambda functions break Chapel compiler
        record Lambda {
            proc this(lo: int, hi: int, N: int) {
                var deviceId: int(32);
                GetDevice(deviceId);
                if (verbose) {
                    var count: int(32);
                    GetDeviceCount(count);
                    writeln("In cubSortPairsCallback, launching the CUDA kernel with a range of ", lo, "..", hi, " (Size: ", N, "), GPU", deviceId, " of ", count, " @", here);
                }
                aEntry.toDevice(deviceId);
                var devA = aEntry.getDeviceArray(deviceId);
                var devRanksOut = ranksEntry.getDeviceArray(deviceId);
                // these are temporary arrays that do not need to be cached on SymEntry
                var devAOut = new GPUArray(aOut.localSlice(lo .. hi));
                var devRanksIn = new GPUArray(ranksIn.localSlice(lo .. hi));
                cubSortPairs(t, devA, devAOut, devRanksIn, devRanksOut, N);
            }
        }
        // get local domain's indices
        var lD = aD.localSubdomain();
        // calc task's indices from local domain's indices
        var tD = {lD.low..lD.high};
        var cubSortPairsCallback = new Lambda();
        forall i in GPU(tD, cubSortPairsCallback) {
            writeln("Should not reach this point!");
            exit(1);
        }

        if disableMultiGPUs || nGPUs == 1 {
            // no need to merge
            return ranksEntry;
        } else {
            // merge sorted chunks
            var ranks: [aD] int;
            var kr: [aD] (t,int) = [(key,rank) in zip(aOut,ranksOut)] (key,rank);
            var krOut: [aD] (t,int);
            mergeSortedRanks(krOut, kr, nGPUs);
            ranks = [(_, rank) in krOut] rank;
            return new shared SymEntry(ranks);
        }
    }
}
