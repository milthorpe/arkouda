module CUBRadixSort {
use IO;
    use MultiTypeSymEntry;
    use GPUIterator;
    use GPUAPI;
    use CTypes;
    use KWayMerge;
    use DistMerge;
    use Time;

    config param logSortKernelTime = false;
    config param cubRadixSortVerbose = false;

    extern proc cubSortKeys_int32(keys_in: c_ptr(void), keys_out: c_ptr(void), N: c_size_t);
    extern proc cubSortKeys_int64(keys_in: c_ptr(void), keys_out: c_ptr(void), N: c_size_t);
    extern proc cubSortKeys_float(keys_in: c_ptr(void), keys_out: c_ptr(void), N: c_size_t);
    extern proc cubSortKeys_double(keys_in: c_ptr(void), keys_out: c_ptr(void), N: c_size_t);
    extern proc cubSortPairs_int32(keys_in: c_ptr(void), keys_out: c_ptr(void), values_in: c_ptr(void), values_out: c_ptr(void), N: c_size_t);
    extern proc cubSortPairs_int64(keys_in: c_ptr(void), keys_out: c_ptr(void), values_in: c_ptr(void), values_out: c_ptr(void), N: c_size_t);
    extern proc cubSortPairs_float(keys_in: c_ptr(void), keys_out: c_ptr(void), values_in: c_ptr(void), values_out: c_ptr(void), N: c_size_t);
    extern proc cubSortPairs_double(keys_in: c_ptr(void), keys_out: c_ptr(void), values_in: c_ptr(void), values_out: c_ptr(void), N: c_size_t);

    extern proc enablePeerAccess(devices: [] int(32), nGPUs: int(32));
    extern proc createDeviceBuffers_int32(num_elements: c_size_t, devices: [] int(32), nGPUs: int(32)): c_ptr(void);
    extern proc createDeviceBuffers_int64(num_elements: c_size_t, devices: [] int(32), nGPUs: int(32)): c_ptr(void);
    extern proc createDeviceBuffers_float(num_elements: c_size_t, devices: [] int(32), nGPUs: int(32)): c_ptr(void);
    extern proc createDeviceBuffers_double(num_elements: c_size_t, devices: [] int(32), nGPUs: int(32)): c_ptr(void);
    extern proc destroyDeviceBuffers_int32(device_buffers: c_ptr(void));
    extern proc destroyDeviceBuffers_int64(device_buffers: c_ptr(void));
    extern proc destroyDeviceBuffers_float(device_buffers: c_ptr(void));
    extern proc destroyDeviceBuffers_double(device_buffers: c_ptr(void));
    extern proc copyDeviceBufferToHost_int32(device_buffers: c_ptr(void), hostArray: [] int(32), N: c_size_t);
    extern proc copyDeviceBufferToHost_int64(device_buffers: c_ptr(void), hostArray: [] int(64), N: c_size_t);
    extern proc copyDeviceBufferToHost_float(device_buffers: c_ptr(void), hostArray: [] real(32), N: c_size_t);
    extern proc copyDeviceBufferToHost_double(device_buffers: c_ptr(void), hostArray: [] real(64), N: c_size_t);

    extern proc findPivot_int32(device_buffers: c_ptr(void), devices: [] int(32), nGPUs: int(32)): int;
    extern proc findPivot_int64(device_buffers: c_ptr(void), devices: [] int(32), nGPUs: int(32)): int;
    extern proc findPivot_float(device_buffers: c_ptr(void), devices: [] int(32), nGPUs: int(32)): int;
    extern proc findPivot_double(device_buffers: c_ptr(void), devices: [] int(32), nGPUs: int(32)): int;
    extern proc swapPartitions_int32(device_buffers: c_ptr(void), pivot: c_size_t, devices: [] int(32), nGPUs: int(32), devicesToMerge: [] int(32));
    extern proc swapPartitions_int64(device_buffers: c_ptr(void), pivot: c_size_t, devices: [] int(32), nGPUs: int(32), devicesToMerge: [] int(32));
    extern proc swapPartitions_float(device_buffers: c_ptr(void), pivot: c_size_t, devices: [] int(32), nGPUs: int(32), devicesToMerge: [] int(32));
    extern proc swapPartitions_double(device_buffers: c_ptr(void), pivot: c_size_t, devices: [] int(32), nGPUs: int(32), devicesToMerge: [] int(32));
    extern proc mergeLocalPartitions_int32(device_buffers: c_ptr(void), pivot: c_size_t, deviceToMerge: int, devices: [] int(32), nGPUs: int(32));
    extern proc mergeLocalPartitions_int64(device_buffers: c_ptr(void), pivot: c_size_t, deviceToMerge: int, devices: [] int(32), nGPUs: int(32));
    extern proc mergeLocalPartitions_float(device_buffers: c_ptr(void), pivot: c_size_t, deviceToMerge: int, devices: [] int(32), nGPUs: int(32));
    extern proc mergeLocalPartitions_double(device_buffers: c_ptr(void), pivot: c_size_t, deviceToMerge: int, devices: [] int(32), nGPUs: int(32));

    extern proc sortToDeviceBuffer_int32(keys_in: c_ptr(void), device_buffers: c_ptr(void), N: c_size_t);
    extern proc sortToDeviceBuffer_int64(keys_in: c_ptr(void), device_buffers: c_ptr(void), N: c_size_t);
    extern proc sortToDeviceBuffer_float(keys_in: c_ptr(void), device_buffers: c_ptr(void), N: c_size_t);
    extern proc sortToDeviceBuffer_double(keys_in: c_ptr(void), device_buffers: c_ptr(void), N: c_size_t);

    private proc createDeviceBuffers(type t, numElements: c_size_t, devices: [] int(32)): c_ptr(void) {
        if t == int(32) {
            return createDeviceBuffers_int32(numElements, devices, devices.size: int(32));
        } else if t == int(64) {
            return createDeviceBuffers_int64(numElements, devices, devices.size: int(32));
        } else if t == real(32) {
            return createDeviceBuffers_float(numElements, devices, devices.size: int(32));
        } else if t == real(64) {
            return createDeviceBuffers_double(numElements, devices, devices.size: int(32));
        } else {
	    // TODO Josh handle other types
            return nil;
	}
    }

    private proc copyDeviceBufferToHost(type t, deviceBuffers: c_ptr(void), hostArray: [] t, N: c_size_t) {
        if t == int(32) {
            copyDeviceBufferToHost_int32(deviceBuffers, hostArray, N);
        } else if t == int(64) {
            copyDeviceBufferToHost_int64(deviceBuffers, hostArray, N);
        } else if t == real(32) {
            copyDeviceBufferToHost_float(deviceBuffers, hostArray, N);
        } else if t == real(64) {
            copyDeviceBufferToHost_double(deviceBuffers, hostArray, N);
        }
    }

    private proc destroyDeviceBuffers(type t, deviceBuffers: c_ptr(void)) {
        if t == int(32) {
            destroyDeviceBuffers_int32(deviceBuffers);
        } else if t == int(64) {
            destroyDeviceBuffers_int64(deviceBuffers);
        } else if t == real(32) {
            destroyDeviceBuffers_float(deviceBuffers);
        } else if t == real(64) {
            destroyDeviceBuffers_double(deviceBuffers);
        }
    }

    private proc findPivot(type t, device_buffers: c_ptr(void), devices: [] int(32), nGPUs: int(32)): int {
        if t == int(32) {
            return findPivot_int32(device_buffers, devices, nGPUs);
        } else if t == int(64) {
            return findPivot_int64(device_buffers, devices, nGPUs);
        } else if t == real(32) {
            return findPivot_float(device_buffers, devices, nGPUs);
        } else if t == real(64) {
            return findPivot_double(device_buffers, devices, nGPUs);
        } else {
            return max(int); // error
        }
    }

    private proc swapPartitions(type t, device_buffers: c_ptr(void), pivot: c_size_t, devices: [] int(32), nGPUs: int(32), devicesToMerge: [] int(32)) {
        if t == int(32) {
            swapPartitions_int32(device_buffers, pivot, devices, nGPUs, devicesToMerge);
        } else if t == int(64) {
            swapPartitions_int64(device_buffers, pivot, devices, nGPUs, devicesToMerge);
        } else if t == real(32) {
            swapPartitions_float(device_buffers, pivot, devices, nGPUs, devicesToMerge);
        } else if t == real(64) {
            swapPartitions_double(device_buffers, pivot, devices, nGPUs, devicesToMerge);
        }
    }

    private proc mergeLocalPartitions(type t, device_buffers: c_ptr(void), pivot: c_size_t, deviceToMerge: int, devices: [] int(32), nGPUs: int(32)) {
        if t == int(32) {
            mergeLocalPartitions_int32(device_buffers, pivot, deviceToMerge, devices, nGPUs);
        } else if t == int(64) {
            mergeLocalPartitions_int64(device_buffers, pivot, deviceToMerge, devices, nGPUs);
        } else if t == real(32) {
            mergeLocalPartitions_float(device_buffers, pivot, deviceToMerge, devices, nGPUs);
        } else if t == real(64) {
            mergeLocalPartitions_double(device_buffers, pivot, deviceToMerge, devices, nGPUs);
        }
    }

    private proc mergePartitions(type t, deviceBuffers: c_ptr(void), devices: [] int(32)) {
        if (devices.size > 2) {
            coforall i in 0..1 {
                var devicesSubset: [0..#devices.size / 2] int(32);
                devicesSubset = devices((i * (devices.size / 2)) .. #(devices.size / 2));
                mergePartitions(
                    t,
                    deviceBuffers,
                    devicesSubset);
            }
        }

        var pivot = findPivot(t, deviceBuffers, devices, devices.size: int(32));
        if (pivot > 0) {
            var devicesToMerge: [0..1] int(32);
            swapPartitions(t, deviceBuffers, pivot, devices, devices.size: int(32), devicesToMerge);
            coforall deviceToMerge in devicesToMerge {
                mergeLocalPartitions(t, deviceBuffers, pivot, deviceToMerge, devices, devices.size: int(32));
            }
        }

        if (devices.size > 2) {
            coforall i in 0..1 {
                var devicesSubset: [0..#devices.size / 2] int(32);
                devicesSubset = devices((i * (devices.size / 2)) .. #(devices.size / 2));
                mergePartitions(
                    t,
                    deviceBuffers,
                    devicesSubset);
            }
        }
/*
// DEBUG - extra synchronization after merge
        coforall deviceId in devices {
          SetDevice(deviceId); 
          DeviceSynchronize();
        }
*/
    }

    private proc cubSortKeysMergeOnHost(type t, devInPtr: c_ptr(void), devAOut: GPUArray, N: int) {
        if t == int(32) {
            cubSortKeys_int32(devInPtr, devAOut.dPtr(), N: c_size_t);
        } else if t == int(64) {
            cubSortKeys_int64(devInPtr, devAOut.dPtr(), N: c_size_t);
        } else if t == real(32) {
            cubSortKeys_float(devInPtr, devAOut.dPtr(), N: c_size_t);
        } else if t == real(64) {
            cubSortKeys_double(devInPtr, devAOut.dPtr(), N: c_size_t);
        }
        DeviceSynchronize();

        if !disableMultiGPUs || nGPUs > 1 {
            devAOut.fromDevice();
        }
    }

    private proc cubSortKeysMergeOnGPU(type t, devInPtr: c_ptr(void), N: int, deviceBuffers: c_ptr(void)) {
        if t == int(32) {
            sortToDeviceBuffer_int32(devInPtr, deviceBuffers, N);
        } else if t == int(64) {
            sortToDeviceBuffer_int64(devInPtr, deviceBuffers, N);
        } else if t == real(32) {
            sortToDeviceBuffer_float(devInPtr, deviceBuffers, N);
        } else if t == real(64) {
            sortToDeviceBuffer_double(devInPtr, deviceBuffers, N);
        }
    }

    private proc cubSortPairs(type t, devInPtr: c_ptr(void), devAOut: GPUArray, devRanksIn: GPUArray, devRanksOutPtr: c_ptr(void), N: int) {
        devRanksIn.toDevice();
        if t == int(32) {
            cubSortPairs_int32(devInPtr, devAOut.dPtr(), devRanksIn.dPtr(), devRanksOutPtr, N: c_size_t);
        } else if t == int(64) {
            cubSortPairs_int64(devInPtr, devAOut.dPtr(), devRanksIn.dPtr(), devRanksOutPtr, N: c_size_t);
        } else if t == real(32) {
            cubSortPairs_float(devInPtr, devAOut.dPtr(), devRanksIn.dPtr(), devRanksOutPtr, N: c_size_t);
        } else if t == real(64) {
            cubSortPairs_double(devInPtr, devAOut.dPtr(), devRanksIn.dPtr(), devRanksOutPtr, N: c_size_t);
        }
        DeviceSynchronize();
        
        if !disableMultiGPUs || nGPUs > 1 {
            devAOut.fromDevice();
        }
    }

    /** On each locale, setup GPU peer access between all devices */
    proc setupPeerAccess() {
        coforall loc in Locales do on loc {
            var allDevices = devices;
            coforall i in allDevices {
                SetDevice(i);
                enablePeerAccess(allDevices, nGPUs);
            }
        }
    }

    // TODO update SortMsg to call the SymEntry version of this proc
    proc cubRadixSortLSD_keys(a: [?aD] ?t, mergeOnGPU: bool = false) {
        var aEntry = new SymEntry(a, GPU=true);
        if mergeOnGPU then
            return cubRadixSortLSDKeysMergeOnGPU(aEntry);
        else
            return cubRadixSortLSDKeysMergeOnHost(aEntry);
    }

    // TODO update ArgSortMsg to call the SymEntry version of this proc
    proc cubRadixSortLSD_ranks(a: [?aD] ?t) throws {
        var aEntry = new SymEntry(a, GPU=true);
        var ranksEntry = cubRadixSortLSD_ranks(aEntry);
        var ranks = ranksEntry.a;
        return ranks;
    }

    /* Radix Sort Least Significant Digit
       radix sort a block distributed array,
       returning a sorted array, without updating the input
     */
    proc cubRadixSortLSDKeysMergeOnHost(e: SymEntry(?)) {
        if !e.GPU then throw new owned Error("cubRadixSortLSDKeysMergeOnHost called on non-GPU SymEntry");
        ref a = e.a;
        var aD = a.domain;
        type t = e.etype;
        var timerLoc1: stopwatch;
        if logSortKernelTime {
            timerLoc1.start();
        }
        var dest: [aD] t = noinit;
        if logSortKernelTime {
            timerLoc1.stop();
            writef("create dest %10.3dr\n", timerLoc1.elapsed()*1000.0);
            timerLoc1.clear();
            timerLoc1.start();
        }

        coforall loc in a.targetLocales() with (ref dest) do on loc {
            var timer: stopwatch;
            if logSortKernelTime {
                timer.start();
            }

            // TODO: proper lambda functions break Chapel compiler
            record Lambda {
                proc this(lo: int, hi: int, N: int) {
                    var deviceId: int(32);
                    GetDevice(deviceId);
                    e.prefetchLocalDataToDevice(lo, hi, deviceId);
                    if (cubRadixSortVerbose) {
                        var count: int(32);
                        GetDeviceCount(count);
                        writeln(here, " GPU", deviceId, " of ", count, " cubSortKeysCallback, launching the CUDA kernel with a range of ", lo, "..", hi, " (Size: ", N, ")");
                    }
                    // these are temporary arrays that do not need to be cached on SymEntry
                    var devDest = new GPUArray(dest.localSlice(lo .. hi));
                    cubSortKeysMergeOnHost(t, e.c_ptrToLocalData(lo), devDest, N);
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

            if logSortKernelTime {
                timer.stop();
                writef("sort %10.3dr\n", timer.elapsed()*1000.0);
                timer.clear();
                timer.start();
            }

            if disableMultiGPUs || nGPUs == 1 {
                // no need to merge
            } else {
                // merge sorted chunks
                var merged: [aD] t;
                mergeSortedKeys(dest, merged, nGPUs);
                if logSortKernelTime {
                    timer.stop();
                    writef("merge on host %10.3dr\n", timer.elapsed()*1000.0);
                    timer.clear();
                    timer.start();
                }
                dest = merged; // TODO can we eliminate this copy?
            }
        }

        if (aD.targetLocales().size > 1) {
            mergeSortedChunks(dest);
        }
        return dest;
    }

    proc cubRadixSortLSDKeysMergeOnGPU(e: SymEntry(?)) where e.GPU == true {
        ref a = e.a;
        var aD = a.domain;
        type t = e.etype;

        var timerLoc1: stopwatch;
        if logSortKernelTime {
            timerLoc1.start();
        }

        var dest: [aD] t = noinit;

        if logSortKernelTime {
            timerLoc1.stop();
            writef("\ncreate_dest            %10.3dr\n", timerLoc1.elapsed()*1000.0);
            timerLoc1.clear();
            timerLoc1.start();
        }
        
        coforall loc in a.targetLocales() do on loc {
            // get local domain's indices
            var lD = aD.localSubdomain();
            var timer: stopwatch;
            if logSortKernelTime {
                timer.start();
            }

            var allDevices = GPUIterator.devices;
            if (cubRadixSortVerbose) {
                writeln(here, " creating device buffers for elements ", lD, " size ", lD.size, " allDevices stored at ", allDevices.locale, " ptr ", c_ptrTo(allDevices));
                try! stdout.flush();
            }

            const deviceBuffers = createDeviceBuffers(t, lD.size, allDevices);

            if logSortKernelTime {
                timer.stop();
                writef("create_dev_buffers  %2n %10.3dr\n", here.id, timer.elapsed()*1000.0);
                try! stdout.flush();
                timer.clear();
                timer.start();
            }

            // TODO: proper lambda functions break Chapel compiler
            record Lambda {
                proc this(lo: int, hi: int, N: int) {
                    var deviceId: int(32);
                    GetDevice(deviceId);
                    //e.prefetchLocalDataToDevice(lo, hi, deviceId); // TODO does this even work on AMD?
                    if (cubRadixSortVerbose) {
                        var count: int(32);
                        GetDeviceCount(count);
                        writeln(here, " GPU", deviceId, " of ", count, " cubSortKeysCallback launching cubSortKeysMergeOnGPU on dev ptr ", e.c_ptrToLocalData(lo), " with a range of ", lD.low+lo, "..", lD.low+hi, " (Size: ", N, ")");
                        try! stdout.flush();
                    }
                    cubSortKeysMergeOnGPU(t, e.c_ptrToLocalData(lo), N, deviceBuffers);
                }
            }
            
            // calc task's indices from local domain's indices
            var tD = {lD.low..lD.high};
            var cubSortKeysCallback = new Lambda();
            forall i in GPU(tD, cubSortKeysCallback) {
                writeln("Should not reach this point!");
                exit(1);
            }

            if logSortKernelTime {
                timer.stop();
                writef("sort                %2n %10.3dr\n", here.id, timer.elapsed()*1000.0);
                try! stdout.flush();
                timer.clear();
                timer.start();
            }

            mergePartitions(t, deviceBuffers, allDevices);

            if logSortKernelTime {
                timer.stop();
                writef("merge_on_GPU        %2n %10.3dr\n", here.id, timer.elapsed()*1000.0);
                try! stdout.flush();
                timer.clear();
                timer.start();
            }

            record Lambda2 {
                proc this(lo: int, hi: int, N: int) {
                    const lD = dest.localSubdomain();
                    if (cubRadixSortVerbose) {
                        var deviceId: int(32);
                        GetDevice(deviceId);
                        var count: int(32);
                        GetDeviceCount(count);
                        writeln(here, " GPU", deviceId, " of ", count, " copyBackCallback launching copyDeviceBufferToHost on dest.localSlice with a range of ", lD.low+lo, "..", lD.low+hi, " (Size: ", N, ")");
                        try! stdout.flush();
                    }
                    copyDeviceBufferToHost(t, deviceBuffers, dest.localSlice(lD.low+lo .. lD.low+hi), N);
                    //DeviceSynchronize(); // previous operation already synchronizes stream
                }
            }
            var copyBackCallback = new Lambda2();
            forall i in GPU(tD, copyBackCallback) {
                writeln("Should not reach this point!");
                exit(1);
            }

            if logSortKernelTime {
                timer.stop();
                writef("copy_back           %2n %10.3dr\n", here.id, timer.elapsed()*1000.0);
                try! stdout.flush();
                timer.clear();
                timer.start();
            }
            
            destroyDeviceBuffers(t, deviceBuffers);

            if logSortKernelTime {
                timer.stop();
                writef("destroy_dev_buffers %2n %10.3dr\n", here.id, timer.elapsed()*1000.0);
                try! stdout.flush();
            }
        }

        if logSortKernelTime {
            timerLoc1.stop();
            writef("GPU_sort               %10.3dr\n", timerLoc1.elapsed()*1000.0);
            try! stdout.flush();
            timerLoc1.clear();
            timerLoc1.start();
        }

        if (aD.targetLocales().size > 1) {
            mergeSortedChunks(dest);
            if logSortKernelTime {
                timerLoc1.stop();
                writef("dist_merge             %10.3dr\n", timerLoc1.elapsed()*1000.0);
                try! stdout.flush();
            }
        }
        return dest;
    }

    /* Radix Sort Least Significant Digit
       radix sort a block distributed array
       returning a permutation vector as a block distributed array */
    proc cubRadixSortLSD_ranks(e: SymEntry(?)) throws where e.GPU == true {
        ref a = e.a;
        var aD = a.domain;
        type t = e.etype;

        var ranksIn: [aD] int = [rank in aD] rank;
        var aOut: [aD] t;
        var ranksOut: [aD] int = noinit;
        var ranksEntry = createSymEntry(ranksOut, true);

        // TODO: proper lambda functions break Chapel compiler
        record Lambda {
            proc this(lo: int, hi: int, N: int) {
                var deviceId: int(32);
                GetDevice(deviceId);
                e.prefetchLocalDataToDevice(lo, hi, deviceId);
                if (cubRadixSortVerbose) {
                    var count: int(32);
                    GetDeviceCount(count);
                    writeln("In cubSortPairsCallback, launching the CUDA kernel with a range of ", lo, "..", hi, " (Size: ", N, "), GPU", deviceId, " of ", count, " @", here);
                }
                // these are temporary arrays that do not need to be cached on SymEntry
                var devAOut = new GPUArray(aOut.localSlice(lo .. hi));
                var devRanksIn = new GPUArray(ranksIn.localSlice(lo .. hi));
                cubSortPairs(t, e.c_ptrToLocalData(lo), devAOut, devRanksIn, ranksEntry.c_ptrToLocalData(lo), N);
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
            ranksEntry.a = [(_, rank) in krOut] rank;
            return ranksEntry;
        }
    }
}
