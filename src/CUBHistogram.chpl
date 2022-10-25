module CUBHistogram {
    use MultiTypeSymEntry;
    use GPUIterator;
    use GPUAPI;
    use CTypes;
    use IO;

    config const reduceOnGPU = true;

    extern proc cubHistogram_int32(samples: c_void_ptr, histogram: c_void_ptr, num_levels: int, lower_bound: int(32), upper_bound: int(32), N: int);
    extern proc cubHistogram_int64(samples: c_void_ptr, histogram: c_void_ptr, num_levels: int, lower_bound: int(64), upper_bound: int(64), N: int);
    extern proc cubHistogram_float(samples: c_void_ptr, histogram: c_void_ptr, num_levels: int, lower_bound: real(32), upper_bound: real(32), N: int);
    extern proc cubHistogram_double(samples: c_void_ptr, histogram: c_void_ptr, num_levels: int, lower_bound: real(64), upper_bound: real(64), N: int);

    extern proc gpuCommGetUniqueID(): c_void_ptr;
    extern proc gpuCommInitRank(numRanks: int(32), commId: c_void_ptr, rank: int(32)): c_void_ptr;
    extern proc gpuCommDestroy(comm: c_void_ptr);

    extern proc gpuAllReduce_sum_int32(src: c_void_ptr, dst: c_void_ptr, N: c_size_t, comm: c_void_ptr);
    extern proc gpuAllReduce_sum_int64(src: c_void_ptr, dst: c_void_ptr, N: c_size_t, comm: c_void_ptr);
    extern proc gpuAllReduce_sum_float(src: c_void_ptr, dst: c_void_ptr, N: c_size_t, comm: c_void_ptr);
    extern proc gpuAllReduce_sum_double(src: c_void_ptr, dst: c_void_ptr, N: c_size_t, comm: c_void_ptr);

    // Chapel doesn't seem to expose these common FP operations
    extern proc nextafterf(from: real(32), to: real(32)): real(32);
    extern proc nextafter(from: real(64), to: real(64)): real(64);

    var comm: [0..#nGPUs] c_void_ptr;

    private proc cubHistogram(type t, devSamples: GPUArray, histogram: [] int, lower_bound: t, upper_bound: t, N: int, deviceId: int(32)) {
        var num_levels = histogram.size + 1;
        var devHistogram = new GPUArray(histogram);

        // CUB histogram is exclusive of the upper bound, whereas Arkouda is inclusive
        if t == int(32) {
            var upper = upper_bound + 1;
            cubHistogram_int32(devSamples.dPtr(), devHistogram.dPtr(), num_levels, lower_bound, upper, N);
            if reduceOnGPU && nGPUs > 1 then gpuAllReduce_sum_int32(devHistogram.dPtr(), devHistogram.dPtr(), num_levels: c_size_t, comm[deviceId]);
        } else if t == int(64) {
            var upper = upper_bound + 1;
            cubHistogram_int64(devSamples.dPtr(), devHistogram.dPtr(), num_levels, lower_bound, upper, N);
            if reduceOnGPU && nGPUs > 1 then gpuAllReduce_sum_int64(devHistogram.dPtr(), devHistogram.dPtr(), num_levels: c_size_t, comm[deviceId]);
        } else if t == real(32) {
            var upper = nextafterf(upper_bound, max(real(32)));
            cubHistogram_float(devSamples.dPtr(), devHistogram.dPtr(), num_levels, lower_bound, upper, N);
            if reduceOnGPU && nGPUs > 1 then gpuAllReduce_sum_float(devHistogram.dPtr(), devHistogram.dPtr(), num_levels: c_size_t, comm[deviceId]);
        } else if t == real(64) {
            var upper = nextafter(upper_bound, max(real(64)));
            cubHistogram_double(devSamples.dPtr(), devHistogram.dPtr(), num_levels, lower_bound, upper, N);
            if reduceOnGPU && nGPUs > 1 then gpuAllReduce_sum_double(devHistogram.dPtr(), devHistogram.dPtr(), num_levels: c_size_t, comm[deviceId]);
        }
        DeviceSynchronize();
        if reduceOnGPU {
            if deviceId == 0 then devHistogram.fromDevice();
        } else {
            devHistogram.fromDevice();
        }
    }

    // TODO update HistogramMsg to call the SymEntry version of this proc
    proc cubHistogram(a: [?aD] ?t, aMin: t, aMax: t, bins: int, binWidth: real) {
        var aEntry = new SymEntry(a);
        return cubHistogram(aEntry, aMin, aMax, bins, binWidth);
    }

    proc cubHistogram(e: SymEntry, aMin: ?etype, aMax: etype, bins: int, binWidth: real) {
        e.createDeviceCache();

        // each device computes its histogram in a separate array
        var deviceHistograms: [0..#nGPUs][0..#bins] int;

        // TODO: proper lambda functions break Chapel compiler
        record Lambda {
            proc this(lo: int, hi: int, N: int) {
                var deviceId: int(32);
                GetDevice(deviceId);
                e.toDevice(deviceId);
                cubHistogram(e.etype, e.getDeviceArray(deviceId), deviceHistograms[deviceId], aMin, aMax, N, deviceId);
            }
        }
        // get local domain's indices
        var lD = e.a.domain.localSubdomain();
        // calc task's indices from local domain's indices
        var tD = {lD.low..lD.high};
        var cubHistogramCallback = new Lambda();
        forall i in GPU(tD, cubHistogramCallback) {
            writeln("Should not reach this point!");
            exit(1);
        }
        e.deviceCache!.isCurrent = true;

        if reduceOnGPU || disableMultiGPUs || nGPUs == 1 {
            // no need to merge
            return deviceHistograms[0];
        } else {
            return + reduce deviceHistograms;
        }
    }

    // TODO move collective communications to separate module?
    proc setupCommunicator() {
        var commId = gpuCommGetUniqueID();
        forall deviceId in 0..#nGPUs {
            SetDevice(deviceId:int(32));
            comm[deviceId] = gpuCommInitRank(nGPUs:int(32), commId, deviceId:int(32));
        }
    }
    proc destroyCommunicator() {
        forall deviceId in 0..#nGPUs {
            SetDevice(deviceId:int(32));
            gpuCommDestroy(comm[deviceId]);
        }
    }
}
