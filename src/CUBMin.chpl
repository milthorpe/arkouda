module CUBMin {
    use MultiTypeSymEntry;
    use GPUCollectives;
    use GPUIterator;
    use GPUAPI;
    use CTypes;
    use IO;

    config const minReduceOnGPU = true;

    extern proc cubMin_int32(input: c_ptr(void), output: c_ptr(void), num_items: c_size_t);
    extern proc cubMin_int64(input: c_ptr(void), output: c_ptr(void), num_items: c_size_t);
    extern proc cubMin_float(input: c_ptr(void), output: c_ptr(void), num_items: c_size_t);
    extern proc cubMin_double(input: c_ptr(void), output: c_ptr(void), num_items: c_size_t);

    private proc cubMinDevice(type etype, devInPtr: c_ptr(void), N: int, deviceId: int(32)) {
        var hostOut: [0..#1] etype;
        var devOut = new GPUArray(hostOut);
        if etype == int(32) {
            cubMin_int32(devInPtr, devOut.dPtr(), N: c_size_t);
            if minReduceOnGPU && nGPUs > 1 then gpuReduce_min_int32(devOut.dPtr():c_ptr(int(32)), devOut.dPtr():c_ptr(int(32)), 1, 0, comm[deviceId]);
        } else if etype == int(64) {
            cubMin_int64(devInPtr, devOut.dPtr(), N: c_size_t);
            if minReduceOnGPU && nGPUs > 1 then gpuReduce_min_int64(devOut.dPtr():c_ptr(int(64)), devOut.dPtr():c_ptr(int(64)), 1, 0, comm[deviceId]);
        } else if etype == real(32) {
            cubMin_float(devInPtr, devOut.dPtr(), N: c_size_t);
            if minReduceOnGPU && nGPUs > 1 then gpuReduce_min_float(devOut.dPtr():c_ptr(real(32)), devOut.dPtr():c_ptr(real(32)), 1, 0, comm[deviceId]);
        } else if etype == real(64) {
            cubMin_double(devInPtr, devOut.dPtr(), N: c_size_t);
            if minReduceOnGPU && nGPUs > 1 then gpuReduce_min_double(devOut.dPtr():c_ptr(real(64)), devOut.dPtr():c_ptr(real(64)), 1, 0, comm[deviceId]);
        }
        DeviceSynchronize();
        if !minReduceOnGPU || deviceId == 0 then devOut.fromDevice();
	    return hostOut[0];
    }

    // TODO update ReductionMsg to call the SymEntry version of this proc
    proc cubMin(a: [?aD] ?t) {
        var aEntry = new SymEntry(a);
        return cubMin(aEntry);
    }

    proc cubMin(ref e: SymEntry(?)) where e.GPU == true {
        var min: e.etype = 0;
        ref a = e.a;
        coforall loc in a.targetLocales() with (+ reduce min) do on loc {
            var deviceMin: [0..#nGPUs] e.etype;

            // TODO: proper lambda functions break Chapel compiler
            record Lambda {
                proc this(lo: int, hi: int, N: int) {
                    var localDom = a.localSubdomain();
                    var deviceId: int(32);
                    GetDevice(deviceId);
                    e.prefetchLocalDataToDevice(lo, hi, deviceId);
                    deviceMin[deviceId] = cubMinDevice(e.etype, e.c_ptrToLocalData(lo), N, deviceId);
                }
            }
            // get local domain's indices
            var lD = a.localSubdomain();
            // calc task's indices from local domain's indices
            var tD = {lD.low..lD.high};
            var cubMinCallback = new Lambda();

            forall i in GPU(tD, cubMinCallback) {
                writeln("Should not reach this point!");
                exit(1);
            }

            if minReduceOnGPU || disableMultiGPUs || nGPUs == 1 {
                // no need to merge
                min += deviceMin[0];
            } else {
                min += (+ reduce deviceMin);
            }
        }
        return min;
    }

    proc cubMinUnified(arr: GPUUnifiedArray(?)) {
        var deviceMin: [0..#nGPUs] arr.etype;

        // TODO: proper lambda functions break Chapel compiler
        record Lambda {
            proc this(lo: int, hi: int, N: int) {
                var deviceId: int(32);
                GetDevice(deviceId);
                arr.prefetchToDevice(lo, hi, deviceId);
                deviceMin[deviceId] = cubMinDevice(arr.etype, arr.dPtr(lo), N, deviceId);
            }
        }
        // get local domain's indices
        var lD = arr.a.domain.localSubdomain();
        // calc task's indices from local domain's indices
        var tD = {lD.low..lD.high};
        var cubMinCallback = new Lambda();

        forall i in GPU(tD, cubMinCallback) {
            writeln("Should not reach this point!");
            exit(1);
        }

        if minReduceOnGPU || disableMultiGPUs || nGPUs == 1 {
            // no need to merge
            return deviceMin[0];
        }

        return min reduce deviceMin;
    }
}
