module CUBHistogram {
    use GPUIterator;
    use GPUAPI;
    use CTypes;
    use IO;

    config param verbose = false;

    extern proc cubHistogram_int32(samples: c_void_ptr, histogram: c_void_ptr, num_levels: int, lower_bound: int(32), upper_bound: int(32), N: int);
    extern proc cubHistogram_int64(samples: c_void_ptr, histogram: c_void_ptr, num_levels: int, lower_bound: int(64), upper_bound: int(64), N: int);
    extern proc cubHistogram_float(samples: c_void_ptr, histogram: c_void_ptr, num_levels: int, lower_bound: real(32), upper_bound: real(32), N: int);
    extern proc cubHistogram_double(samples: c_void_ptr, histogram: c_void_ptr, num_levels: int, lower_bound: real(64), upper_bound: real(64), N: int);

    // Chapel doesn't seem to expose these common FP operations
    extern proc nextafterf(from: real(32), to: real(32)): real(32);
    extern proc nextafter(from: real(64), to: real(64)): real(64);

    proc cubHistogram(samples: [] ?t, histogram: [] int(32), lower_bound: t, upper_bound: t, N: int, lo: int, hi: int) {
        var num_levels = histogram.size + 1;
        //try! writeln("num_levels %t lower_bound %t upper_bound %t".format(num_levels, lower_bound, upper_bound));
        var devSamples = new GPUArray(samples.localSlice(lo .. hi));
        var devHistogram = new GPUArray(histogram);
        devSamples.toDevice();
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

    proc cubHistogram(a: [?aD] ?etype, aMin: etype, aMax: etype, bins: int, binWidth: real) {
        // TODO CUB doesn't support int(64) histogram bins due to lack of atomic ops on GPU
        if (a.size > max(int(32))) {
            try! stderr.writeln("warning: CUB histogram will overflow if there are more than %t items in a bin".format(max(int(32))));
        }

        var nGPUs: int(32);
        GetDeviceCount(nGPUs);

        // each device computes its histogram in a separate array
        var deviceHistograms: [0..#nGPUs][0..#bins] int(32);

        // TODO: proper lambda functions break Chapel compiler
        record Lambda {
            proc this(lo: int, hi: int, N: int) {
                var device: int(32);
                GetDevice(device);
                cubHistogram(a, deviceHistograms[device], aMin, aMax, N, lo, hi);
            }
        }
        // get local domain's indices
        var lD = aD.localSubdomain();
        // calc task's indices from local domain's indices
        var tD = {lD.low..lD.high};
        var cubHistogramCallback = new Lambda();
        forall i in GPU(tD, cubHistogramCallback) {
            writeln("Should not reach this point!");
            exit(1);
        }

        // need to return histogram bin counts at int(64) as per Arkouda API
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
