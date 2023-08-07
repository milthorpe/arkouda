#include <hipcub/hipcub.hpp>
#include <stdio.h>
#include "hip_error_check.h"

using namespace hipcub;

#define DebugExit(x) if (HipcubDebug(x)) exit(1);

template <typename T> void cubSortKeys(const T *d_keys_in, T *d_keys_out, size_t N) {
  size_t temp_storage_bytes = 0;
  void *d_temp_storage = NULL;

  CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory
  // run SortKeys once to determine the necessary size of d_temp_storage
  DebugExit(DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, N));
  DebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  DebugExit(DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, N));
  if (d_temp_storage) DebugExit(g_allocator.DeviceFree(d_temp_storage));
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
  DebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, N));
  DebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  DebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, N));
  if (d_temp_storage) DebugExit(g_allocator.DeviceFree(d_temp_storage));
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
