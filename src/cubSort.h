#ifndef _CUB_SORT_H_
#define _CUB_SORT_H_
#include <stddef.h>
#include <stdint.h>
void cubSortKeys_int32(int32_t *d_keys_in, int32_t *d_keys_out, size_t N);
void cubSortKeys_int64(int64_t *d_keys_in, int64_t *d_keys_out, size_t N);
void cubSortKeys_float(float *d_keys_in, float *d_keys_out, size_t N);
void cubSortKeys_double(double *d_keys_in, double *d_keys_out, size_t N);
void cubSortPairs_int32(int32_t *d_keys_in, int32_t *d_keys_out, int64_t *d_values_in, int64_t *d_values_out, size_t N);
void cubSortPairs_int64(int64_t *d_keys_in, int64_t *d_keys_out, int64_t *d_values_in, int64_t *d_values_out, size_t N);
void cubSortPairs_float(float *d_keys_in, float *d_keys_out, int64_t *d_values_in, int64_t *d_values_out, size_t N);
void cubSortPairs_double(double *d_keys_in, double *d_keys_out, int64_t *d_values_in, int64_t *d_values_out, size_t N);
void *createDeviceBuffers_int32(const size_t num_elements, const int *devices, const int nGPUs);
int *getDeviceBufferData_int32(void *device_buffers_ptr);
void copyDeviceBufferToHost_int32(void *device_buffers_ptr, int *hostArray, const size_t N);
void updateDeviceBufferOffset_int32(void *device_buffers_ptr, const size_t N);
size_t findPivot(void *device_buffers_ptr, const int *gpus, const int nGPUs);
void swapPartitions(void *device_buffers_ptr, size_t pivot, const int *gpus, const int nGPUs, int* devicesToMerge);
void mergeLocalPartitions(void *device_buffers_ptr, size_t pivot, int devicesToMerge, const int *gpus, const int nGPUs, size_t num_fillers);
#endif
