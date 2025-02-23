module GPUCollectives {
    use GPUIterator;
    use GPUAPI;
    use CTypes;

    extern proc gpuCommGetUniqueID(): c_ptr(void);
    extern proc gpuCommInitRank(numRanks: int(32), commId: c_ptr(void), rank: int(32)): c_ptr(void);
    extern proc gpuCommDestroy(comm: c_ptr(void));

    extern proc gpuReduce_sum_int32(src: c_ptr(int(32)), dst: c_ptr(int(32)), N: c_size_t, root: c_int, comm: c_ptr(void));
    extern proc gpuReduce_sum_int64(src: c_ptr(int(64)), dst: c_ptr(int(64)), N: c_size_t, root: c_int, comm: c_ptr(void));
    extern proc gpuReduce_sum_float(src: c_ptr(real(32)), dst: c_ptr(real(32)), N: c_size_t, root: c_int, comm: c_ptr(void));
    extern proc gpuReduce_sum_double(src: c_ptr(real(64)), dst: c_ptr(real(64)), N: c_size_t, root: c_int, comm: c_ptr(void));
    extern proc gpuReduce_min_int32(src: c_ptr(int(32)), dst: c_ptr(int(32)), N: c_size_t, root: c_int, comm: c_ptr(void));
    extern proc gpuReduce_min_int64(src: c_ptr(int(64)), dst: c_ptr(int(64)), N: c_size_t, root: c_int, comm: c_ptr(void));
    extern proc gpuReduce_min_float(src: c_ptr(real(32)), dst: c_ptr(real(32)), N: c_size_t, root: c_int, comm: c_ptr(void));
    extern proc gpuReduce_min_double(src: c_ptr(real(64)), dst: c_ptr(real(64)), N: c_size_t, root: c_int, comm: c_ptr(void));
    extern proc gpuReduce_max_int32(src: c_ptr(int(32)), dst: c_ptr(int(32)), N: c_size_t, root: c_int, comm: c_ptr(void));
    extern proc gpuReduce_max_int64(src: c_ptr(int(64)), dst: c_ptr(int(64)), N: c_size_t, root: c_int, comm: c_ptr(void));
    extern proc gpuReduce_max_float(src: c_ptr(real(32)), dst: c_ptr(real(32)), N: c_size_t, root: c_int, comm: c_ptr(void));
    extern proc gpuReduce_max_double(src: c_ptr(real(64)), dst: c_ptr(real(64)), N: c_size_t, root: c_int, comm: c_ptr(void));
    extern proc gpuAllReduce_sum_int32(src: c_ptr(int(32)), dst: c_ptr(int(32)), N: c_size_t, comm: c_ptr(void));
    extern proc gpuAllReduce_sum_int64(src: c_ptr(int(64)), dst: c_ptr(int(64)), N: c_size_t, comm: c_ptr(void));
    extern proc gpuAllReduce_sum_float(src: c_ptr(real(32)), dst: c_ptr(real(32)), N: c_size_t, comm: c_ptr(void));
    extern proc gpuAllReduce_sum_double(src: c_ptr(real(64)), dst: c_ptr(real(64)), N: c_size_t, comm: c_ptr(void));

    var comm: [0..#nGPUs] c_ptr(void);

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
