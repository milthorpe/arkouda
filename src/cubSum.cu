#include <cub/cub.cuh>
#include <stdio.h>

#define CUDA_ERROR_CHECK
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

using namespace cub;

template <typename T> void cubSum(const T *d_in, T *d_out, int64_t num_items) {
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

  CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));

  // Allocate temporary storage
  CudaSafeCall(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  // Compute Sum
  CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));

  CudaSafeCall(cudaFree(d_temp_storage));
}

extern "C" {
void cubSum_int32(const int32_t *d_in, int32_t *d_out, int64_t num_items) {
  cubSum(d_in, d_out, num_items);
}

void cubSum_int64(const int64_t *d_in, int64_t *d_out, int64_t num_items) {
  cubSum(d_in, d_out, num_items);
}

void cubSum_float(const float *d_in, float *d_out, int64_t num_items) {
  cubSum(d_in, d_out, num_items);
}

void cubSum_double(const double *d_in, double *d_out, int64_t num_items) {
  cubSum(d_in, d_out, num_items);
}
}
