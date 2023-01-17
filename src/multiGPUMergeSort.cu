#include "tanasic_sort.cuh"
#include <stdio.h>

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
T *getDeviceBufferData(void *device_buffers_ptr) {
  int deviceId;
  CheckCudaError(cudaGetDevice(&deviceId));
  DeviceBuffers<T>* device_buffers = (DeviceBuffers<T>*)device_buffers_ptr;
  return (T*)(device_buffers->AtPrimary(deviceId)->data().get());
}

template <typename T>
void copyDeviceBufferToHost(void *device_buffers_ptr, T *hostArray, const size_t N) {
  int deviceId;
  CheckCudaError(cudaGetDevice(&deviceId));
  DeviceBuffers<T>* device_buffers = (DeviceBuffers<T>*)device_buffers_ptr;
  CheckCudaError(cudaStreamSynchronize(*device_buffers->GetPrimaryStream(deviceId)));
  CheckCudaError(cudaMemcpy(hostArray,
                            thrust::raw_pointer_cast(device_buffers->AtPrimary(deviceId)->data()),
                            sizeof(T) * N,
                            cudaMemcpyDeviceToHost));
}

template <typename T>
void updateDeviceBufferOffset(void *device_buffers_ptr, const size_t N) {
  DeviceBuffers<T>* device_buffers = (DeviceBuffers<T>*)device_buffers_ptr;
  int deviceId;
  CheckCudaError(cudaGetDevice(&deviceId));
  //CheckCudaError(cudaStreamSynchronize(*device_buffers->GetPrimaryStream(deviceId)));
  device_buffers->GetDeviceAllocator(deviceId)->SetOffset(N * sizeof(T));
}

template <typename T>
size_t findPivot(void *device_buffers_ptr, const int *gpus, const int nGPUs) {
  DeviceBuffers<T>* device_buffers = (DeviceBuffers<T>*)device_buffers_ptr;
  std::vector<int> devices(gpus, gpus+nGPUs);
  int32_t pivot = FindPivot<T>(device_buffers, devices);
  //printf("findPivot for GPUs: ");
  //for (int i=0; i<nGPUs; i++) printf("%d ", devices[i]);
  //printf(" = %d\n", pivot);
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
                          int deviceToMerge, const int *gpus, const int nGPUs,
                          size_t num_fillers) {
  DeviceBuffers<T>* device_buffers = (DeviceBuffers<T>*)device_buffers_ptr;
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

extern "C" {
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

int32_t *getDeviceBufferData_int32(void *device_buffers_ptr) {
  return getDeviceBufferData<int32_t>(device_buffers_ptr);
}

int64_t *getDeviceBufferData_int64(void *device_buffers_ptr) {
  return getDeviceBufferData<int64_t>(device_buffers_ptr);
}

float *getDeviceBufferData_float(void *device_buffers_ptr) {
  return getDeviceBufferData<float>(device_buffers_ptr);
}

double *getDeviceBufferData_double(void *device_buffers_ptr) {
  return getDeviceBufferData<double>(device_buffers_ptr);
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

void updateDeviceBufferOffset_int32(void *device_buffers_ptr, const size_t N) {
  updateDeviceBufferOffset<int32_t>(device_buffers_ptr, N);
}

void updateDeviceBufferOffset_int64(void *device_buffers_ptr, const size_t N) {
  updateDeviceBufferOffset<int64_t>(device_buffers_ptr, N);
}

void updateDeviceBufferOffset_float(void *device_buffers_ptr, const size_t N) {
  updateDeviceBufferOffset<float>(device_buffers_ptr, N);
}

void updateDeviceBufferOffset_double(void *device_buffers_ptr, const size_t N) {
  updateDeviceBufferOffset<double>(device_buffers_ptr, N);
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

void mergeLocalPartitions_int32(void *device_buffers_ptr, size_t pivot, int deviceToMerge, const int *gpus, const int nGPUs, size_t num_fillers) {
  mergeLocalPartitions<int32_t>(device_buffers_ptr, pivot, deviceToMerge, gpus, nGPUs, num_fillers);
}

void mergeLocalPartitions_int64(void *device_buffers_ptr, size_t pivot, int deviceToMerge, const int *gpus, const int nGPUs, size_t num_fillers) {
  mergeLocalPartitions<int64_t>(device_buffers_ptr, pivot, deviceToMerge, gpus, nGPUs, num_fillers);
}

void mergeLocalPartitions_float(void *device_buffers_ptr, size_t pivot, int deviceToMerge, const int *gpus, const int nGPUs, size_t num_fillers) {
  mergeLocalPartitions<float>(device_buffers_ptr, pivot, deviceToMerge, gpus, nGPUs, num_fillers);
}

void mergeLocalPartitions_double(void *device_buffers_ptr, size_t pivot, int deviceToMerge, const int *gpus, const int nGPUs, size_t num_fillers) {
  mergeLocalPartitions<double>(device_buffers_ptr, pivot, deviceToMerge, gpus, nGPUs, num_fillers);
}
}