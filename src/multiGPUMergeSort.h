#ifndef _MULTI_GPU_MERGE_SORT_H_
#define _MULTI_GPU_MERGE_SORT_H_
#include <stddef.h>
#include <stdint.h>
void *createDeviceBuffers_int32(const size_t num_elements, const int *devices, const int nGPUs);
void *createDeviceBuffers_int64(const size_t num_elements, const int *devices, const int nGPUs);
void *createDeviceBuffers_float(const size_t num_elements, const int *devices, const int nGPUs);
void *createDeviceBuffers_double(const size_t num_elements, const int *devices, const int nGPUs);
void *destroyDeviceBuffers_int32(const size_t num_elements, const int *devices, const int nGPUs);
void *destroyDeviceBuffers_int64(const size_t num_elements, const int *devices, const int nGPUs);
void *destroyDeviceBuffers_float(const size_t num_elements, const int *devices, const int nGPUs);
void *destroyDeviceBuffers_double(const size_t num_elements, const int *devices, const int nGPUs);
int32_t *getDeviceBufferData_int32(void *device_buffers_ptr);
int64_t *getDeviceBufferData_int64(void *device_buffers_ptr);
float *getDeviceBufferData_float(void *device_buffers_ptr);
double *getDeviceBufferData_double(void *device_buffers_ptr);
void copyDeviceBufferToHost_int32(void *device_buffers_ptr, int32_t *hostArray, const size_t N);
void copyDeviceBufferToHost_int64(void *device_buffers_ptr, int64_t *hostArray, const size_t N);
void copyDeviceBufferToHost_float(void *device_buffers_ptr, float *hostArray, const size_t N);
void copyDeviceBufferToHost_double(void *device_buffers_ptr, double *hostArray, const size_t N);
void updateDeviceBufferOffset_int32(void *device_buffers_ptr, const size_t N);
void updateDeviceBufferOffset_int64(void *device_buffers_ptr, const size_t N);
void updateDeviceBufferOffset_float(void *device_buffers_ptr, const size_t N);
void updateDeviceBufferOffset_double(void *device_buffers_ptr, const size_t N);
size_t findPivot_int32(void *device_buffers_ptr, const int *gpus, const int nGPUs);
size_t findPivot_int64(void *device_buffers_ptr, const int *gpus, const int nGPUs);
size_t findPivot_float(void *device_buffers_ptr, const int *gpus, const int nGPUs);
size_t findPivot_double(void *device_buffers_ptr, const int *gpus, const int nGPUs);
void swapPartitions_int32(void *device_buffers_ptr, size_t pivot, const int *gpus, const int nGPUs, int* devicesToMerge);
void swapPartitions_int64(void *device_buffers_ptr, size_t pivot, const int *gpus, const int nGPUs, int* devicesToMerge);
void swapPartitions_float(void *device_buffers_ptr, size_t pivot, const int *gpus, const int nGPUs, int* devicesToMerge);
void swapPartitions_double(void *device_buffers_ptr, size_t pivot, const int *gpus, const int nGPUs, int* devicesToMerge);
void mergeLocalPartitions_int32(void *device_buffers_ptr, size_t pivot, int devicesToMerge, const int *gpus, const int nGPUs);
void mergeLocalPartitions_int64(void *device_buffers_ptr, size_t pivot, int devicesToMerge, const int *gpus, const int nGPUs);
void mergeLocalPartitions_float(void *device_buffers_ptr, size_t pivot, int devicesToMerge, const int *gpus, const int nGPUs);
void mergeLocalPartitions_double(void *device_buffers_ptr, size_t pivot, int devicesToMerge, const int *gpus, const int nGPUs);
void sortToDeviceBuffer_int32(const int32_t *d_keys_in, void *device_buffers_ptr, size_t N);
void sortToDeviceBuffer_int64(const int64_t *d_keys_in, void *device_buffers_ptr, size_t N);
void sortToDeviceBuffer_float(const float *d_keys_in, void *device_buffers_ptr, size_t N);
void sortToDeviceBuffer_double(const double *d_keys_in, void *device_buffers_ptr, size_t N);
#endif
