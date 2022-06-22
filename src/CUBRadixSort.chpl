module CUBRadixSort {
    use GPUIterator;
    use GPUAPI;
    use CTypes;

    config param verbose = false;

    extern proc cubSortPairs(keys_in: c_void_ptr, keys_out: c_void_ptr, values_in: c_void_ptr, values_out: c_void_ptr, N: c_size_t);

    /* Radix Sort Least Significant Digit
       radix sort a block distributed array
       returning a permutation vector as a block distributed array */
    proc cubRadixSortLSD_ranks(a:[?aD] ?t): [aD] int {
	if (!disableMultiGPUs) {
	  writeln("Multiple GPUs are currently not supported, found ", nGPUs, " GPUs!");
	  exit(1);
	}
/*
        var nGPUs: int(32);
        GetDeviceCount(nGPUs);
        for i in 0..<nGPUs {
            // TODO GA Tech API's chunk calculation is uneven
            writeln("chunk ",i," is ",computeChunk(tD.dim(0), i, nGPUs));
        }
*/
        var ranksIn: [aD] int = [rank in aD] rank;
        var ranksOut: [aD] int;
        var aOut: [aD] t;

        // TODO: proper lambda functions break Chapel compiler
        record Lambda {
            proc this(lo: int, hi: int, N: int) {
                if (verbose) {
                    var device, count: int(32);
                    GetDevice(device);
                    GetDeviceCount(count);
                    writeln("In countDigitsCallback, launching the CUDA kernel with a range of ", lo, "..", hi, " (Size: ", N, "), GPU", device, " of ", count, " @", here);
                }
                var devA = new GPUArray(a.localSlice(lo .. hi));
                var devAOut = new GPUArray(aOut.localSlice(lo .. hi));
                var devRanksIn = new GPUArray(ranksIn.localSlice(lo .. hi));
                var devRanksOut = new GPUArray(ranksOut.localSlice(lo .. hi));
                devA.toDevice();
                devRanksIn.toDevice();
                cubSortPairs(devA.dPtr(), devAOut.dPtr(), devRanksIn.dPtr(), devRanksOut.dPtr(), N: c_size_t);
                DeviceSynchronize();
                devRanksOut.fromDevice(); // copy back indices; don't care about aOut
            }
        }
        // get local domain's indices
        var lD = aD.localSubdomain();
        // calc task's indices from local domain's indices
        var tD = {lD.low..lD.high};
        var cubSortCallback = new Lambda();
        forall i in GPU(tD, cubSortCallback) {
            writeln("Should not reach this point!");
            exit(1);
        }
        // TODO: work out how to get splits of ranks array and do a k-way merge in Chapel
        return ranksOut;
    }
}
