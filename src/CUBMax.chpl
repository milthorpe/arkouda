module CUBMax {
    use MultiTypeSymEntry;
    use GPUCollectives;
    use GPUIterator;
    use GPUAPI;
    use CTypes;
    use IO;

    config const maxReduceOnGPU = true;

    extern proc cubMax_int32(input: c_void_ptr, output: c_void_ptr, num_items: c_size_t);
    extern proc cubMax_int64(input: c_void_ptr, output: c_void_ptr, num_items: c_size_t);
    extern proc cubMax_float(input: c_void_ptr, output: c_void_ptr, num_items: c_size_t);
    extern proc cubMax_double(input: c_void_ptr, output: c_void_ptr, num_items: c_size_t);

    private proc cubMaxDevice(type etype, devInPtr: c_void_ptr, N: int, deviceId: int(32)) {
        var hostOut: [0..#1] etype;
        var devOut = new GPUArray(hostOut);
        if etype == int(32) {
            cubMax_int32(devInPtr, devOut.dPtr(), N: c_size_t);
            if maxReduceOnGPU && nGPUs > 1 then gpuReduce_max_int32(devOut.dPtr():c_ptr(int(32)), devOut.dPtr():c_ptr(int(32)), 1, 0, comm[deviceId]);
        } else if etype == int(64) {
            cubMax_int64(devInPtr, devOut.dPtr(), N: c_size_t);
            if maxReduceOnGPU && nGPUs > 1 then gpuReduce_max_int64(devOut.dPtr():c_ptr(int(64)), devOut.dPtr():c_ptr(int(64)), 1, 0, comm[deviceId]);
        } else if etype == real(32) {
            cubMax_float(devInPtr, devOut.dPtr(), N: c_size_t);
            if maxReduceOnGPU && nGPUs > 1 then gpuReduce_max_float(devOut.dPtr():c_ptr(real(32)), devOut.dPtr():c_ptr(real(32)), 1, 0, comm[deviceId]);
        } else if etype == real(64) {
            cubMax_double(devInPtr, devOut.dPtr(), N: c_size_t);
            if maxReduceOnGPU && nGPUs > 1 then gpuReduce_max_double(devOut.dPtr():c_ptr(real(64)), devOut.dPtr():c_ptr(real(64)), 1, 0, comm[deviceId]);
        }
        DeviceSynchronize();
        if !maxReduceOnGPU || deviceId == 0 then devOut.fromDevice();
	    return hostOut[0];
    }

    // TODO update ReductionMsg to call the SymEntry version of this proc
    proc cubMax(a: [?aD] ?t) {
        var aEntry = new SymEntry(a);
        return cubMax(aEntry);
    }

    proc cubMax(ref e: SymEntry) {
        var max: e.etype = 0;
        ref a = e.a;
        coforall loc in a.targetLocales() with (+ reduce max) do on loc {
            var deviceMax: [0..#nGPUs] e.etype;

            // TODO: proper lambda functions break Chapel compiler
            record Lambda {
                proc this(lo: int, hi: int, N: int) {
                    var localDom = a.localSubdomain();
                    var deviceId: int(32);
                    GetDevice(deviceId);
                    e.prefetchLocalDataToDevice(lo, hi, deviceId);
                    deviceMax[deviceId] = cubMaxDevice(e.etype, e.c_ptrToLocalData(lo), N, deviceId);
                }
            }
            // get local domain's indices
            var lD = a.localSubdomain();
            // calc task's indices from local domain's indices
            var tD = {lD.low..lD.high};
            var cubMaxCallback = new Lambda();

            forall i in GPU(tD, cubMaxCallback) {
                writeln("Should not reach this point!");
                exit(1);
            }

            if maxReduceOnGPU || disableMultiGPUs || nGPUs == 1 {
                // no need to merge
                max += deviceMax[0];
            } else {
                max += (+ reduce deviceMax);
            }
        }
        return max;
    }

    proc cubMaxUnified(arr: GPUUnifiedArray) {
        var deviceMax: [0..#nGPUs] arr.etype;

        // TODO: proper lambda functions break Chapel compiler
        record Lambda {
            proc this(lo: int, hi: int, N: int) {
                var deviceId: int(32);
                GetDevice(deviceId);
                arr.prefetchToDevice(lo, hi, deviceId);
                deviceMax[deviceId] = cubMaxDevice(arr.etype, arr.dPtr(lo), N, deviceId);
            }
        }
        // get local domain's indices
        var lD = arr.a.domain.localSubdomain();
        // calc task's indices from local domain's indices
        var tD = {lD.low..lD.high};
        var cubMaxCallback = new Lambda();

        forall i in GPU(tD, cubMaxCallback) {
            writeln("Should not reach this point!");
            exit(1);
        }

        if disableMultiGPUs || nGPUs == 1 {
            // no need to merge
            return deviceMax[0];
        }

        if maxReduceOnGPU || disableMultiGPUs || nGPUs == 1 {
            // no need to merge
            return deviceMax[0];
        }

        return max reduce deviceMax;
    }
}
