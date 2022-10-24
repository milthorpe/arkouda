#include <cub/cub.cuh>
#include "nccl.h"
#include <stdio.h>

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

using namespace cub;

template <typename T> void cubHistogram(const T *d_samples, unsigned long long int *d_histogram, int num_levels, T lower_bound, T upper_bound, int64_t N) {
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory
  CubDebugExit(cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
      d_samples, d_histogram, num_levels, lower_bound, upper_bound, N));
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Compute histograms
  CubDebugExit(cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
    d_samples, d_histogram, num_levels, lower_bound, upper_bound, N));

  if (d_temp_storage) cudaFree(d_temp_storage);
  //if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
}

extern "C" {
void cubHistogram_int32(const int32_t *d_samples, unsigned long long int *d_histogram, int num_levels, int32_t lower_bound, int32_t upper_bound, int64_t N) {
  cubHistogram(d_samples, d_histogram, num_levels, lower_bound, upper_bound, N);
}

void cubHistogram_int64(const int64_t *d_samples, unsigned long long int *d_histogram, int num_levels, int64_t lower_bound, int64_t upper_bound, int64_t N) {
  cubHistogram(d_samples, d_histogram, num_levels, lower_bound, upper_bound, N);
}

void cubHistogram_float(const float *d_samples, unsigned long long int *d_histogram, int num_levels, float lower_bound, float upper_bound, int64_t N) {
  cubHistogram(d_samples, d_histogram, num_levels, lower_bound, upper_bound, N);
}

void cubHistogram_double(const double *d_samples, unsigned long long int *d_histogram, int num_levels, double lower_bound, double upper_bound, int64_t N) {
  cubHistogram(d_samples, d_histogram, num_levels, lower_bound, upper_bound, N);
}

void *gpuCommGetUniqueID() {
  ncclUniqueId *unique_id = new ncclUniqueId();
  NCCLCHECK(ncclGetUniqueId(unique_id));
  return unique_id;
}

void *gpuCommInitRank(int num_ranks, void *comm_id, int rank) {
  ncclUniqueId *comm_unique_id = (ncclUniqueId*) comm_id;
  ncclComm_t *comm = new ncclComm_t();
  NCCLCHECK(ncclCommInitRank(comm, num_ranks, *comm_unique_id, rank));
  return comm;
}

void gpuCommDestroy(void *comm_ptr) {
  ncclComm_t *comm = (ncclComm_t*)comm_ptr;
  ncclCommDestroy(*comm);
  delete(comm);
}

void gpuAllReduce_sum_int32(const void *sendbuff, void *recvbuff, size_t count, void *comm_ptr) {
  NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, ncclInt32, ncclSum, *(ncclComm_t*)comm_ptr, 0));
}

void gpuAllReduce_sum_in64(const void *sendbuff, void *recvbuff, size_t count, void *comm_ptr) {
  NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, ncclInt64, ncclSum, *(ncclComm_t*)comm_ptr, 0));
}

void gpuAllReduce_sum_float(const void *sendbuff, void *recvbuff, size_t count, void *comm_ptr) {
  NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, *(ncclComm_t*)comm_ptr, 0));
}

void gpuAllReduce_sum_double(const void *sendbuff, void *recvbuff, size_t count, void *comm_ptr) {
  NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, ncclDouble, ncclSum, *(ncclComm_t*)comm_ptr, 0));
}
}
