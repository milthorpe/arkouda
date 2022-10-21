module CUBHistogram {
    use MultiTypeSymEntry;
    use GPUIterator;
    use GPUAPI;
    use CTypes;
    use IO;

    extern proc cubHistogram_int32(samples: c_void_ptr, histogram: c_void_ptr, num_levels: int, lower_bound: int(32), upper_bound: int(32), N: int);
    extern proc cubHistogram_int64(samples: c_void_ptr, histogram: c_void_ptr, num_levels: int, lower_bound: int(64), upper_bound: int(64), N: int);
    extern proc cubHistogram_float(samples: c_void_ptr, histogram: c_void_ptr, num_levels: int, lower_bound: real(32), upper_bound: real(32), N: int);
    extern proc cubHistogram_double(samples: c_void_ptr, histogram: c_void_ptr, num_levels: int, lower_bound: real(64), upper_bound: real(64), N: int);

    // Chapel doesn't seem to expose these common FP operations
    extern proc nextafterf(from: real(32), to: real(32)): real(32);
    extern proc nextafter(from: real(64), to: real(64)): real(64);

    private proc cubHistogram(type t, devSamples: GPUArray, histogram: [] int, lower_bound: t, upper_bound: t, N: int, deviceId: int(32)) {
        var num_levels = histogram.size + 1;
        var devHistogram = new GPUArray(histogram);

        // CUB histogram is exclusive of the upper bound, whereas Arkouda is inclusive
        if t == int(32) {
            var upper = upper_bound + 1;
            cubHistogram_int32(devSamples.dPtr(), devHistogram.dPtr(), num_levels, lower_bound, upper, N);
        } else if t == int(64) {
            var upper = upper_bound + 1;
            cubHistogram_int64(devSamples.dPtr(), devHistogram.dPtr(), num_levels, lower_bound, upper, N);
        } else if t == real(32) {
            var upper = nextafterf(upper_bound, max(real(32)));
            cubHistogram_float(devSamples.dPtr(), devHistogram.dPtr(), num_levels, lower_bound, upper, N);
        } else if t == real(64) {
            var upper = nextafter(upper_bound, max(real(64)));
            cubHistogram_double(devSamples.dPtr(), devHistogram.dPtr(), num_levels, lower_bound, upper, N);
        }
        DeviceSynchronize();
        devHistogram.fromDevice();
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

        var finalHistogram: [0..#bins] int;
        if disableMultiGPUs || nGPUs == 1 {
            // no need to merge
            finalHistogram = deviceHistograms[0];
        } else {
            finalHistogram = + reduce deviceHistograms;
        }
        return finalHistogram;
    }
}
