#include <cub/cub.cuh>
#include "tanasic_sort.cuh"
#include <stdio.h>

using namespace cub;

template <typename T> void cubSortKeys(const T *d_keys_in, T *d_keys_out, size_t N) {
  size_t temp_storage_bytes = 0;
  void *d_temp_storage = NULL;

  CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory
  // run SortKeys once to determine the necessary size of d_temp_storage
  CubDebugExit(DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, N));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  CubDebugExit(DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, N));
  if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
}

extern "C" {
void cubSortKeys_int32(const int32_t *d_keys_in, int32_t *d_keys_out, size_t N) {
  cubSortKeys(d_keys_in, d_keys_out, N);
}

void cubSortKeys_int64(const int64_t *d_keys_in, int64_t *d_keys_out, size_t N) {
  cubSortKeys(d_keys_in, d_keys_out, N);
}

void cubSortKeys_float(const float *d_keys_in, float *d_keys_out, size_t N) {
  cubSortKeys(d_keys_in, d_keys_out, N);
}

void cubSortKeys_double(const double *d_keys_in, double *d_keys_out, size_t N) {
  cubSortKeys(d_keys_in, d_keys_out, N);
}
}

template <typename T> void cubSortPairs(const T *d_keys_in, T *d_keys_out, const int64_t *d_values_in, int64_t *d_values_out, size_t N) {
  size_t temp_storage_bytes = 0;
  void *d_temp_storage = NULL;

  CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory
  // run SortPairs once to determine the necessary size of d_temp_storage
  CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, N));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, N));
  if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
}

extern "C" {
void cubSortPairs_int32(const int32_t *d_keys_in, int32_t *d_keys_out, const int64_t *d_values_in, int64_t *d_values_out, size_t N) {
  cubSortPairs(d_keys_in, d_keys_out, d_values_in, d_values_out, N);
}

void cubSortPairs_int64(const int64_t *d_keys_in, int64_t *d_keys_out, const int64_t *d_values_in, int64_t *d_values_out, size_t N) {
  cubSortPairs(d_keys_in, d_keys_out, d_values_in, d_values_out, N);
}

void cubSortPairs_float(const float *d_keys_in, float *d_keys_out, const int64_t *d_values_in, int64_t *d_values_out, size_t N) {
  cubSortPairs(d_keys_in, d_keys_out, d_values_in, d_values_out, N);
}

void cubSortPairs_double(const double *d_keys_in, double *d_keys_out, const int64_t *d_values_in, int64_t *d_values_out, size_t N) {
  cubSortPairs(d_keys_in, d_keys_out, d_values_in, d_values_out, N);
}
}

template <typename T>
DeviceBuffers<T>* createDeviceBuffers(const size_t num_elements, const int *gpus, const int nGPUs) {
  //printf("createDeviceBuffers nGPUs %d\n", nGPUs);
  std::vector<int> devices(gpus, gpus+nGPUs);
  size_t num_fillers = (num_elements % nGPUs != 0) ? (nGPUs - num_elements % nGPUs) : 0;
  size_t buffer_size = (num_elements + num_fillers) / nGPUs;
  DeviceBuffers<T>* device_buffers = new DeviceBuffers<T>(devices, buffer_size, 2);

  int lastDevice = gpus[nGPUs - 1];
  thrust::fill(thrust::cuda::par(*device_buffers->GetDeviceAllocator(lastDevice))
                    .on(*device_buffers->GetPrimaryStream(lastDevice)),
                device_buffers->AtPrimary(lastDevice)->end() - num_fillers, device_buffers->AtPrimary(lastDevice)->end(),
                std::numeric_limits<T>::max());
  return device_buffers;
}

template <typename T>
void updateDeviceBufferOffset(void *device_buffers_ptr, const size_t N) {
  DeviceBuffers<int32_t>* device_buffers = (DeviceBuffers<int32_t>*)device_buffers_ptr;
  int deviceId;
  CheckCudaError(cudaGetDevice(&deviceId));
  CheckCudaError(cudaStreamSynchronize(*device_buffers->GetPrimaryStream(deviceId)));
  device_buffers->GetDeviceAllocator(deviceId)->SetOffset(N * sizeof(T));
}

extern "C" {
void *createDeviceBuffers_int32(const size_t num_elements, const int *devices, const int nGPUs) {
  return createDeviceBuffers<int32_t>(num_elements, devices, nGPUs);
}

int *getDeviceBufferData_int32(void *device_buffers_ptr) {
  int deviceId;
  CheckCudaError(cudaGetDevice(&deviceId));
  DeviceBuffers<int32_t>* device_buffers = (DeviceBuffers<int32_t>*)device_buffers_ptr;
  return (int*)(device_buffers->AtPrimary(deviceId)->data().get());
}

void copyDeviceBufferToHost_int32(void *device_buffers_ptr, int *hostArray, const size_t N) {
  int deviceId;
  CheckCudaError(cudaGetDevice(&deviceId));
  DeviceBuffers<int32_t>* device_buffers = (DeviceBuffers<int32_t>*)device_buffers_ptr;
  CheckCudaError(cudaStreamSynchronize(*device_buffers->GetPrimaryStream(deviceId)));
  CheckCudaError(cudaMemcpy(hostArray,
                            thrust::raw_pointer_cast(device_buffers->AtPrimary(deviceId)->data()),
                            sizeof(int32_t) *N,
                            cudaMemcpyDeviceToHost));
}

void updateDeviceBufferOffset_int32(void *device_buffers_ptr, const size_t N) {
  updateDeviceBufferOffset<int32_t>(device_buffers_ptr, N);
}

size_t findPivot(void *device_buffers_ptr, const int *gpus, const int nGPUs) {
  DeviceBuffers<int32_t>* device_buffers = (DeviceBuffers<int32_t>*)device_buffers_ptr;
  std::vector<int> devices(gpus, gpus+nGPUs);
  int32_t pivot = FindPivot<int32_t>(device_buffers, devices);
  //printf("findPivot for GPUs: ");
  //for (int i=0; i<nGPUs; i++) printf("%d ", devices[i]);
  //printf(" = %d\n", pivot);
  return pivot;
}

void swapPartitions(void *device_buffers_ptr, size_t pivot, const int *gpus, const int nGPUs, int* devicesToMerge) {
  //printf("swapPartitions pivot = %d\n", pivot);
  DeviceBuffers<int32_t>* device_buffers = (DeviceBuffers<int32_t>*)device_buffers_ptr;
  std::vector<int> devices(gpus, gpus+nGPUs);
  std::array<int, 2> merge = SwapPartitions(device_buffers, pivot, devices);
  devicesToMerge[0] = merge[0];
  devicesToMerge[1] = merge[1];
}

void mergeLocalPartitions(void *device_buffers_ptr, size_t pivot,
                          int deviceToMerge, const int *gpus, const int nGPUs,
                          size_t num_fillers) {
  DeviceBuffers<int32_t>* device_buffers = (DeviceBuffers<int32_t>*)device_buffers_ptr;
  std::vector<int> devices(gpus, gpus+nGPUs);

  const size_t partition_size = device_buffers->GetPartitionSize();
  pivot %= partition_size;

  for (size_t i = 0; i < devices.size(); ++i) {
    const int device = devices[i];
    const size_t offset = i >= devices.size() / 2 ? pivot : partition_size - pivot;
    if (device == deviceToMerge) {
      CheckCudaError(cudaSetDevice(device));
      //printf("merging local partitions on %d partition_size %ld pivot %ld offset %ld\n", device, partition_size, pivot, offset);

      thrust::merge(
          thrust::cuda::par(*device_buffers->GetDeviceAllocator(device)).on(*device_buffers->GetPrimaryStream(device)),
          device_buffers->AtPrimary(device)->begin(), device_buffers->AtPrimary(device)->begin() + offset,
          device_buffers->AtPrimary(device)->begin() + offset, device_buffers->AtPrimary(device)->end(),
          device_buffers->AtSecondary(device)->begin());

      device_buffers->Flip(device);

      CheckCudaError(cudaStreamSynchronize(*device_buffers->GetPrimaryStream(device)));
    }
  }
}
}