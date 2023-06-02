module CUBSum {
    use MultiTypeSymEntry;
    use GPUCollectives;
    use GPUIterator;
    use GPUAPI;
    use CTypes;
    use IO;
    use Time;

    config var logSumKernelTime = false;
    config const sumReduceOnGPU = true;

    extern proc cubSum_int32(input: c_void_ptr, output: c_void_ptr, num_items: c_size_t);
    extern proc cubSum_int64(input: c_void_ptr, output: c_void_ptr, num_items: c_size_t);
    extern proc cubSum_float(input: c_void_ptr, output: c_void_ptr, num_items: c_size_t);
    extern proc cubSum_double(input: c_void_ptr, output: c_void_ptr, num_items: c_size_t);

    private proc cubSumDevice(type etype, devInPtr: c_void_ptr, N: int, deviceId: int(32)) {
        var hostOut: [0..#1] etype;
        var devOut = new GPUArray(hostOut);
        if etype == int(32) {
            cubSum_int32(devInPtr, devOut.dPtr(), N: c_size_t);
            if sumReduceOnGPU && nGPUs > 1 then gpuReduce_sum_int32(devOut.dPtr():c_ptr(int(32)), devOut.dPtr():c_ptr(int(32)), 1, 0, comm[deviceId]);
        } else if etype == int(64) {
            cubSum_int64(devInPtr, devOut.dPtr(), N: c_size_t);
            if sumReduceOnGPU && nGPUs > 1 then gpuReduce_sum_int64(devOut.dPtr():c_ptr(int(64)), devOut.dPtr():c_ptr(int(64)), 1, 0, comm[deviceId]);
        } else if etype == real(32) {
            cubSum_float(devInPtr, devOut.dPtr(), N: c_size_t);
            if sumReduceOnGPU && nGPUs > 1 then gpuReduce_sum_float(devOut.dPtr():c_ptr(real(32)), devOut.dPtr():c_ptr(real(32)), 1, 0, comm[deviceId]);
        } else if etype == real(64) {
            cubSum_double(devInPtr, devOut.dPtr(), N: c_size_t);
            if sumReduceOnGPU && nGPUs > 1 then gpuReduce_sum_double(devOut.dPtr():c_ptr(real(64)), devOut.dPtr():c_ptr(real(64)), 1, 0, comm[deviceId]);
        }
        DeviceSynchronize();
        if !sumReduceOnGPU || deviceId == 0 then devOut.fromDevice();
        return hostOut[0];
    }

    // TODO update ReductionMsg to call the SymEntry version of this proc
    proc cubSum(a: [?aD] ?t) {
        var aEntry = new SymEntry(a);
        return cubSum(aEntry);
    }

    /** Perform a sum over an Arkouda array in unified memory */
    proc cubSum(ref e: SymEntry) where e.GPU == true {
        var sum: e.etype = 0;
        ref a = e.a;
        coforall loc in a.targetLocales() with (+ reduce sum) do on loc {
            var deviceSum: [0..#nGPUs] e.etype;

            // TODO: proper lambda functions break Chapel compiler
            record Lambda {
                proc this(lo: int, hi: int, N: int) {
                    var deviceId: int(32);
                    GetDevice(deviceId);
                    e.prefetchLocalDataToDevice(lo, hi, deviceId);
                    var timer: stopwatch;
                    if logSumKernelTime {
                        timer.start();
                    }
                    deviceSum[deviceId] = cubSumDevice(e.etype, e.c_ptrToLocalData(lo), N, deviceId);
                    if logSumKernelTime {
                        timer.stop();
                        if deviceId == 0 then writef("%10.3dr", timer.elapsed()*1000.0);
                    }
                }
            }

            var cubSumCallback = new Lambda();
            forall i in GPU(a.localSubdomain(), cubSumCallback) {
                writeln("Should not reach this point!");
                exit(1);
            }

            if sumReduceOnGPU || disableMultiGPUs || nGPUs == 1 {
                // no need to merge
                sum += deviceSum[0];
            } else {
                sum += (+ reduce deviceSum);
            }
        }
        return sum;
    }

    proc cubSumGPUArray(a: [?D] ?etype){
        var sum: etype = 0;
        coforall loc in a.targetLocales() with (+ reduce sum) do on loc {
            var deviceSum: [0..#nGPUs] etype;

            // TODO: proper lambda functions break Chapel compiler
            record Lambda {
                proc this(lo: int, hi: int, N: int) {
                    var devA = new GPUArray(a.localSlice(lo..hi));
                    devA.toDevice();
                    var deviceId: int(32);
                    GetDevice(deviceId);
                    var timer: stopwatch;
                    if logSumKernelTime {
                        timer.start();
                    }
                    deviceSum[deviceId] = cubSumDevice(etype, devA.dPtr(), N, deviceId);
                    if logSumKernelTime {
                        timer.stop();
                        if deviceId == 0 then writef("%10.3dr", timer.elapsed()*1000.0);
                    }
                }
            }

            var cubSumCallback = new Lambda();
            forall i in GPU(a.localSubdomain(), cubSumCallback) {
                writeln("Should not reach this point!");
                exit(1);
            }

            if sumReduceOnGPU || disableMultiGPUs || nGPUs == 1 {
                // no need to merge
                sum += deviceSum[0];
            } else {
                sum += (+ reduce deviceSum);
            }
        }
        return sum;
    }

    proc cubSumUnified(arr: GPUUnifiedArray) {
        var deviceSum: [0..#nGPUs] arr.etype;

        // TODO: proper lambda functions break Chapel compiler
        record Lambda {
            proc this(lo: int, hi: int, N: int) {
                var deviceId: int(32);
                GetDevice(deviceId);
                arr.prefetchToDevice(lo, hi, deviceId);
                var timer: stopwatch;
                if logSumKernelTime {
                    timer.start();
                }
                deviceSum[deviceId] = cubSumDevice(arr.etype, arr.dPtr(lo), N, deviceId);
                if logSumKernelTime {
                    timer.stop();
                    if deviceId == 0 then writef("%10.3dr", timer.elapsed()*1000.0);
                }
            }
        }
        // get local domain's indices
        var lD = arr.a.domain.localSubdomain();
        var cubSumCallback = new Lambda();

        forall i in GPU(lD, cubSumCallback) {
            writeln("Should not reach this point!");
            exit(1);
        }

        if sumReduceOnGPU || disableMultiGPUs || nGPUs == 1 {
            // no need to merge
            return deviceSum[0];
        }

        return + reduce deviceSum;
    }
}
