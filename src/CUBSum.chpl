module CUBSum {
    use MultiTypeSymEntry;
    use GPUIterator;
    use GPUAPI;
    use CTypes;
    use IO;

    extern proc cubSum_int32(input: c_void_ptr, output: c_void_ptr, num_items: c_size_t);
    extern proc cubSum_int64(input: c_void_ptr, output: c_void_ptr, num_items: c_size_t);
    extern proc cubSum_float(input: c_void_ptr, output: c_void_ptr, num_items: c_size_t);
    extern proc cubSum_double(input: c_void_ptr, output: c_void_ptr, num_items: c_size_t);

    private proc cubSumDevice(type etype, devIn: GPUArray) {
        var num_items = devIn.size;
        var hostOut: [0..<1] etype;
        var devOut = new GPUArray(hostOut);
        if etype == int(32) {
            cubSum_int32(devIn.dPtr(), devOut.dPtr(), num_items: c_size_t);
        } else if etype == int(64) {
            cubSum_int64(devIn.dPtr(), devOut.dPtr(), num_items: c_size_t);
        } else if etype == real(32) {
            cubSum_float(devIn.dPtr(), devOut.dPtr(), num_items: c_size_t);
        } else if etype == real(64) {
            cubSum_double(devIn.dPtr(), devOut.dPtr(), num_items: c_size_t);
        }
        DeviceSynchronize();
        devOut.fromDevice();
        return hostOut[0];
    }

    // TODO update ReductionMsg to call the SymEntry version of this proc
    proc cubSum(a: [?aD] ?t) {
        var aEntry = new SymEntry(a);
        return cubSum(aEntry);
    }

    proc cubSum(e: SymEntry) {
        e.createDeviceCache();
        e.toDevice();

        var deviceSum: [0..#nGPUs] e.etype;

        // TODO: proper lambda functions break Chapel compiler
        record Lambda {
            proc this(lo: int, hi: int, N: int) {
                var deviceId: int(32);
                GetDevice(deviceId);
                deviceSum[deviceId] = cubSumDevice(e.etype, e.getDeviceArray(deviceId));
            }
        }
        // get local domain's indices
        var lD = e.a.domain.localSubdomain();
        // calc task's indices from local domain's indices
        var tD = {lD.low..lD.high};
        var cubSumCallback = new Lambda();
        forall i in GPU(tD, cubSumCallback) {
            writeln("Should not reach this point!");
            exit(1);
        }

        return + reduce deviceSum;
    }

}
