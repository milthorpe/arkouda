#include <cub/cub.cuh>
#include <stdio.h>

using namespace cub;

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

void cubSortPairs_int64(const int32_t *d_keys_in, int32_t *d_keys_out, const int64_t *d_values_in, int64_t *d_values_out, size_t N) {
  cubSortPairs(d_keys_in, d_keys_out, d_values_in, d_values_out, N);
}

void cubSortPairs_float(const float *d_keys_in, float *d_keys_out, const int64_t *d_values_in, int64_t *d_values_out, size_t N) {
  cubSortPairs(d_keys_in, d_keys_out, d_values_in, d_values_out, N);
}

void cubSortPairs_double(const double *d_keys_in, double *d_keys_out, const int64_t *d_values_in, int64_t *d_values_out, size_t N) {
  cubSortPairs(d_keys_in, d_keys_out, d_values_in, d_values_out, N);
}
}
