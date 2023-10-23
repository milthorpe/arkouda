#include "tanasic_sort.cuh"
#include <thrust/system/hip/execution_policy.h>
#include <hipcub/hipcub.hpp>
#include <stdio.h>

#define DebugExit(x) if (HipcubDebug(x)) exit(1);

/** Create temporary devices buffers to be used in merge and swap steps */
template <typename T>
DeviceBuffers<T>* createDeviceBuffers(const size_t num_elements, const int *gpus, const int nGPUs) {
  //printf("createDeviceBuffers nGPUs %d\n", nGPUs);
  std::vector<int> devices(gpus, gpus+nGPUs);
  size_t num_fillers = (num_elements % nGPUs != 0) ? (nGPUs - num_elements % nGPUs) : 0;
  size_t buffer_size = (num_elements + num_fillers) / nGPUs;
  DeviceBuffers<T>* device_buffers = new DeviceBuffers<T>(devices, buffer_size, 2);

  if (num_fillers > 0) {
    int lastDevice = gpus[nGPUs - 1];
    CheckCudaError(hipSetDevice(lastDevice));
    thrust::fill(thrust::hip::par(*device_buffers->GetDeviceAllocator(lastDevice))
                      .on(*device_buffers->GetPrimaryStream(lastDevice)),
                  device_buffers->AtPrimary(lastDevice)->end() - num_fillers, device_buffers->AtPrimary(lastDevice)->end(),
                  std::numeric_limits<T>::max());
    CheckCudaError(hipStreamSynchronize(*device_buffers->GetPrimaryStream(lastDevice)));
  }
  return device_buffers;
}

template <typename T>
void destroyDeviceBuffers(void *device_buffers_ptr) {
  DeviceBuffers<T>* device_buffers = (DeviceBuffers<T>*)device_buffers_ptr;
  delete device_buffers;
}

template <typename T>
void copyDeviceBufferToHost(void *device_buffers_ptr, T *hostArray, const size_t N) {
  int deviceId;
  CheckCudaError(hipGetDevice(&deviceId));
  DeviceBuffers<T>* device_buffers = (DeviceBuffers<T>*)device_buffers_ptr;
  //CheckCudaError(hipStreamSynchronize(*device_buffers->GetPrimaryStream(deviceId)));
  CheckCudaError(hipMemcpyAsync(hostArray,
                            thrust::raw_pointer_cast(device_buffers->AtPrimary(deviceId)->data()),
                            sizeof(T) * N,
                            hipMemcpyDeviceToHost,
			    *device_buffers->GetPrimaryStream(deviceId)));
  CheckCudaError(hipStreamSynchronize(*device_buffers->GetPrimaryStream(deviceId)));
}

template <typename T>
size_t findPivot(void *device_buffers_ptr, const int *gpus, const int nGPUs) {
  DeviceBuffers<T>* device_buffers = (DeviceBuffers<T>*)device_buffers_ptr;
  std::vector<int> devices(gpus, gpus+nGPUs);
  //printf("findPivot for GPUs: ");
  //for (int i=0; i<nGPUs; i++) printf("%d ", devices[i]);
  //printf("\n");
  int32_t pivot = FindPivot<T>(device_buffers, devices);
  //printf("pivot = %d\n", pivot);
  return pivot;
}

template <typename T>
void swapPartitions(void *device_buffers_ptr, size_t pivot, const int *gpus, const int nGPUs, int* devicesToMerge) {
  /*
  printf("swapPartitions, devices[ ");
  for (int i=0; i<nGPUs; i++) {
    printf("%d, ", gpus[i]);
  }
  printf("] pivot = %d\n", pivot);
  */
  DeviceBuffers<T>* device_buffers = (DeviceBuffers<T>*)device_buffers_ptr;
  std::vector<int> devices(gpus, gpus+nGPUs);
  std::array<int, 2> merge = SwapPartitions(device_buffers, pivot, devices);
  devicesToMerge[0] = merge[0];
  devicesToMerge[1] = merge[1];
}

template <typename T>
void mergeLocalPartitions(void *device_buffers_ptr, size_t pivot,
                          int deviceToMerge, const int *gpus, const int nGPUs) {
  DeviceBuffers<T>* device_buffers = (DeviceBuffers<T>*)device_buffers_ptr;
  std::vector<int> devices(gpus, gpus+nGPUs);

  const size_t partition_size = device_buffers->GetPartitionSize();
  pivot %= partition_size;

  for (size_t i = 0; i < devices.size(); ++i) {
    const int device = devices[i];
    if (device == deviceToMerge) {
      const size_t offset = i >= devices.size() / 2 ? pivot : partition_size - pivot;
      CheckCudaError(hipSetDevice(device));
      //printf("merging local partitions on %d partition_size %ld pivot %ld offset %ld\n", device, partition_size, pivot, offset);

      thrust::merge(
          thrust::system::hip::par(*device_buffers->GetDeviceAllocator(device)).on(*device_buffers->GetPrimaryStream(device)),
          device_buffers->AtPrimary(device)->begin(), device_buffers->AtPrimary(device)->begin() + offset,
          device_buffers->AtPrimary(device)->begin() + offset, device_buffers->AtPrimary(device)->end(),
          device_buffers->AtSecondary(device)->begin());

      device_buffers->Flip(device);

      CheckCudaError(hipStreamSynchronize(*device_buffers->GetPrimaryStream(device)));
    }
  }
}

using namespace hipcub;

template <typename T>
void cubSortKeysOnStream(const T *d_keys_in, T *d_keys_out, size_t N, hipStream_t stream) {
  size_t temp_storage_bytes = 0;
  void *d_temp_storage = NULL;
  hipcub::CachingDeviceAllocator g_allocator;  // Caching allocator for device memory

  // run SortKeys once to determine the necessary size of d_temp_storage
  DebugExit(hipcub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, N, 0, sizeof(T)*8, stream));
  DebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes, stream));

  DebugExit(hipcub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, N, 0, sizeof(T)*8, stream));
  if (d_temp_storage) DebugExit(g_allocator.DeviceFree(d_temp_storage));
}

template <typename T>
void sortToDeviceBuffer(const T *d_keys_in, void *device_buffers_ptr, size_t N) {
  int deviceId;
  CheckCudaError(hipGetDevice(&deviceId));
  DeviceBuffers<T>* device_buffers = (DeviceBuffers<T>*)device_buffers_ptr;
  hipStream_t primaryStream = *device_buffers->GetPrimaryStream(deviceId);
  T *d_keys_out = (T*)(device_buffers->AtPrimary(deviceId)->data().get());
  cubSortKeysOnStream(d_keys_in, d_keys_out, N, primaryStream);
  CheckCudaError(hipStreamSynchronize(primaryStream));
  device_buffers->GetDeviceAllocator(deviceId)->SetOffset(N * sizeof(T));
}

extern "C" {
/** Enable peer access to the memory of all other devices */
void enablePeerAccess(const int *devices, const int nGPUs) {
  int deviceId;
  CheckCudaError(hipGetDevice(&deviceId));

  for (size_t i = 0; i < nGPUs; ++i) {
    if (devices[i] != deviceId) {
      CheckCudaError(hipDeviceEnablePeerAccess(devices[i], 0));
    }
  }
}

void *createDeviceBuffers_int32(const size_t num_elements, const int *devices, const int nGPUs) {
  return createDeviceBuffers<int32_t>(num_elements, devices, nGPUs);
}

void *createDeviceBuffers_int64(const size_t num_elements, const int *devices, const int nGPUs) {
  return createDeviceBuffers<int64_t>(num_elements, devices, nGPUs);
}

void *createDeviceBuffers_float(const size_t num_elements, const int *devices, const int nGPUs) {
  return createDeviceBuffers<float>(num_elements, devices, nGPUs);
}

void *createDeviceBuffers_double(const size_t num_elements, const int *devices, const int nGPUs) {
  return createDeviceBuffers<double>(num_elements, devices, nGPUs);
}

void destroyDeviceBuffers_int32(void *device_buffers_ptr) {
  destroyDeviceBuffers<int32_t>(device_buffers_ptr);
}

void destroyDeviceBuffers_int64(void *device_buffers_ptr) {
  destroyDeviceBuffers<int64_t>(device_buffers_ptr);
}

void destroyDeviceBuffers_float(void *device_buffers_ptr) {
  destroyDeviceBuffers<float>(device_buffers_ptr);
}
void destroyDeviceBuffers_double(void *device_buffers_ptr) {
  destroyDeviceBuffers<double>(device_buffers_ptr);
}

void copyDeviceBufferToHost_int32(void *device_buffers_ptr, int32_t *hostArray, const size_t N) {
  copyDeviceBufferToHost<int32_t>(device_buffers_ptr, hostArray, N);
}

void copyDeviceBufferToHost_int64(void *device_buffers_ptr, int64_t *hostArray, const size_t N) {
  copyDeviceBufferToHost<int64_t>(device_buffers_ptr, hostArray, N);
}

void copyDeviceBufferToHost_float(void *device_buffers_ptr, float *hostArray, const size_t N) {
  copyDeviceBufferToHost<float>(device_buffers_ptr, hostArray, N);
}

void copyDeviceBufferToHost_double(void *device_buffers_ptr, double *hostArray, const size_t N) {
  copyDeviceBufferToHost<double>(device_buffers_ptr, hostArray, N);
}

size_t findPivot_int32(void *device_buffers_ptr, const int *gpus, const int nGPUs) {
  return findPivot<int32_t>(device_buffers_ptr, gpus, nGPUs);
}

size_t findPivot_int64(void *device_buffers_ptr, const int *gpus, const int nGPUs) {
  return findPivot<int64_t>(device_buffers_ptr, gpus, nGPUs);
}

size_t findPivot_float(void *device_buffers_ptr, const int *gpus, const int nGPUs) {
  return findPivot<float>(device_buffers_ptr, gpus, nGPUs);
}

size_t findPivot_double(void *device_buffers_ptr, const int *gpus, const int nGPUs) {
  return findPivot<double>(device_buffers_ptr, gpus, nGPUs);
}

void swapPartitions_int32(void *device_buffers_ptr, size_t pivot, const int *gpus, const int nGPUs, int* devicesToMerge) {
  swapPartitions<int32_t>(device_buffers_ptr, pivot, gpus, nGPUs, devicesToMerge);
}

void swapPartitions_int64(void *device_buffers_ptr, size_t pivot, const int *gpus, const int nGPUs, int* devicesToMerge) {
  swapPartitions<int64_t>(device_buffers_ptr, pivot, gpus, nGPUs, devicesToMerge);
}

void swapPartitions_float(void *device_buffers_ptr, size_t pivot, const int *gpus, const int nGPUs, int* devicesToMerge) {
  swapPartitions<float>(device_buffers_ptr, pivot, gpus, nGPUs, devicesToMerge);
}

void swapPartitions_double(void *device_buffers_ptr, size_t pivot, const int *gpus, const int nGPUs, int* devicesToMerge) {
  swapPartitions<double>(device_buffers_ptr, pivot, gpus, nGPUs, devicesToMerge);
}

void mergeLocalPartitions_int32(void *device_buffers_ptr, size_t pivot, int deviceToMerge, const int *gpus, const int nGPUs) {
  mergeLocalPartitions<int32_t>(device_buffers_ptr, pivot, deviceToMerge, gpus, nGPUs);
}

void mergeLocalPartitions_int64(void *device_buffers_ptr, size_t pivot, int deviceToMerge, const int *gpus, const int nGPUs) {
  mergeLocalPartitions<int64_t>(device_buffers_ptr, pivot, deviceToMerge, gpus, nGPUs);
}

void mergeLocalPartitions_float(void *device_buffers_ptr, size_t pivot, int deviceToMerge, const int *gpus, const int nGPUs) {
  mergeLocalPartitions<float>(device_buffers_ptr, pivot, deviceToMerge, gpus, nGPUs);
}

void mergeLocalPartitions_double(void *device_buffers_ptr, size_t pivot, int deviceToMerge, const int *gpus, const int nGPUs) {
  mergeLocalPartitions<double>(device_buffers_ptr, pivot, deviceToMerge, gpus, nGPUs);
}

void sortToDeviceBuffer_int32(const int32_t *d_keys_in, void *device_buffers_ptr, size_t N) {
  sortToDeviceBuffer<int32_t>(d_keys_in, device_buffers_ptr, N);
}

void sortToDeviceBuffer_int64(const int64_t *d_keys_in, void *device_buffers_ptr, size_t N) {
  sortToDeviceBuffer<int64_t>(d_keys_in, device_buffers_ptr, N);
}

void sortToDeviceBuffer_float(const float *d_keys_in, void *device_buffers_ptr, size_t N) {
  sortToDeviceBuffer<float>(d_keys_in, device_buffers_ptr, N);
}

void sortToDeviceBuffer_double(const double *d_keys_in, void *device_buffers_ptr, size_t N) {
  sortToDeviceBuffer<double>(d_keys_in, device_buffers_ptr, N);
}
}
