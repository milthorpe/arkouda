module CUBHistogram {
    use MultiTypeSymEntry;
    use GPUCollectives;
    use GPUIterator;
    use GPUAPI;
    use CTypes;
    use IO;
    use Time;

    config param histogramReduceOnGPU = true;

    extern proc cubHistogram_int32(samples: c_void_ptr, histogram: c_void_ptr, num_levels: int, lower_bound: int(32), upper_bound: int(32), N: int);
    extern proc cubHistogram_int64(samples: c_void_ptr, histogram: c_void_ptr, num_levels: int, lower_bound: int(64), upper_bound: int(64), N: int);
    extern proc cubHistogram_float(samples: c_void_ptr, histogram: c_void_ptr, num_levels: int, lower_bound: real(32), upper_bound: real(32), N: int);
    extern proc cubHistogram_double(samples: c_void_ptr, histogram: c_void_ptr, num_levels: int, lower_bound: real(64), upper_bound: real(64), N: int);

    // Chapel doesn't seem to expose these common FP operations
    extern proc nextafterf(from: real(32), to: real(32)): real(32);
    extern proc nextafter(from: real(64), to: real(64)): real(64);

    private proc cubHistogram(type t, devSamples: c_void_ptr, histogram: [] int, lower_bound: t, upper_bound: t, N: int, deviceId: int(32)) {
        var num_levels = histogram.size + 1;
        var devHistogram = new GPUArray(histogram);

        // CUB histogram is exclusive of the upper bound, whereas Arkouda is inclusive
        if t == int(32) {
            var upper = upper_bound + 1;
            cubHistogram_int32(devSamples, devHistogram.dPtr(), num_levels, lower_bound, upper, N);
        } else if t == int(64) {
            var upper = upper_bound + 1;
            cubHistogram_int64(devSamples, devHistogram.dPtr(), num_levels, lower_bound, upper, N);
        } else if t == real(32) {
            var upper = nextafterf(upper_bound, max(real(32)));
            cubHistogram_float(devSamples, devHistogram.dPtr(), num_levels, lower_bound, upper, N);
        } else if t == real(64) {
            var upper = nextafter(upper_bound, max(real(64)));
            cubHistogram_double(devSamples, devHistogram.dPtr(), num_levels, lower_bound, upper, N);
        }
        if histogramReduceOnGPU {
            if histogramReduceOnGPU && nGPUs > 1 then gpuAllReduce_sum_int64(devHistogram.dPtr():c_ptr(int(64)), devHistogram.dPtr():c_ptr(int(64)), num_levels: c_size_t, comm[deviceId]);
            DeviceSynchronize();
            if deviceId == 0 then devHistogram.fromDevice();
        } else {
            DeviceSynchronize();
            devHistogram.fromDevice();
        }
    }

    // TODO update HistogramMsg to call the SymEntry version of this proc
    proc cubHistogram(samples: [?aD] ?t, sampleMin: t, sampleMax: t, bins: int, binWidth: real) {
        var samplesEntry = new SymEntry(samples);
        return cubHistogram(samplesEntry, sampleMin, sampleMax, bins, binWidth);
    }

    proc cubHistogram(ref samples: SymEntry, sampleMin: ?etype, sampleMax: etype, bins: int, binWidth: real) {
        var hist: [0..#bins] int;
        ref a = samples.a;
        coforall loc in a.targetLocales() with (+ reduce hist) do on loc {
            // each device computes its histogram in a separate array
            var deviceHistograms: [0..#nGPUs][0..#bins] int;

            // TODO: proper lambda functions break Chapel compiler
            record Lambda {
                proc this(lo: int, hi: int, N: int) {
                    var deviceId: int(32);
                    GetDevice(deviceId);
                    samples.prefetchLocalDataToDevice(lo, hi, deviceId);
                    cubHistogram(samples.etype, samples.c_ptrToLocalData(lo), deviceHistograms[deviceId], sampleMin, sampleMax, N, deviceId);
                }
            }
            // get local domain's indices
            var lD = a.localSubdomain();
            // calc task's indices from local domain's indices
            var tD = {lD.low..lD.high};
            var cubHistogramCallback = new Lambda();
            forall i in GPU(tD, cubHistogramCallback) {
                writeln("Should not reach this point!");
                exit(1);
            }

            if histogramReduceOnGPU || disableMultiGPUs || nGPUs == 1 {
                // no need to merge devices
                hist += deviceHistograms[0];
            } else {
                hist += (+ reduce deviceHistograms);
            }
        }
        return hist;
    }

    // TODO update HistogramMsg to call the SymEntry version of this proc
    proc cubHistogramUnified(arr: GPUUnifiedArray, sampleMin: ?t, sampleMax: t, bins: int, binWidth: real) {
        // each device computes its histogram in a separate array
        var deviceHistograms: [0..#nGPUs][0..#bins] int;

        // TODO: proper lambda functions break Chapel compiler
        record Lambda {
            proc this(lo: int, hi: int, N: int) {
                var deviceId: int(32);
                GetDevice(deviceId);
                arr.prefetchToDevice(lo, hi, deviceId);
                cubHistogram(t, arr.dPtr(lo), deviceHistograms[deviceId], sampleMin, sampleMax, N, deviceId);
            }
        }
        var tD = arr.a.domain.localSubdomain();
        var cubHistogramCallback = new Lambda();
        forall i in GPU(tD, cubHistogramCallback) {
            writeln("Should not reach this point!");
            exit(1);
        }

        if histogramReduceOnGPU || disableMultiGPUs || nGPUs == 1 {
            // no need to merge
            return deviceHistograms[0];
        } else {
            return + reduce deviceHistograms;
        }
    }
}
