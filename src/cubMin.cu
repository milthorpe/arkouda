#include <hipcub/hipcub.hpp>

using namespace hipcub;

#define DebugExit(x) if (HipcubDebug(x)) exit(1);

template <typename T> void cubMin(const T *d_in, T *d_out, int64_t num_items) {
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  hipcub::CachingDeviceAllocator g_allocator;  // Caching allocator for device memory

  DebugExit(hipcub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));
  // Allocate temporary storage
  DebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Compute Sum
  DebugExit(hipcub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));

  if (d_temp_storage) DebugExit(g_allocator.DeviceFree(d_temp_storage));
}

extern "C" {
void cubMin_int32(const int32_t *d_in, int32_t *d_out, int64_t num_items) {
  cubMin(d_in, d_out, num_items);
}

void cubMin_int64(const int64_t *d_in, int64_t *d_out, int64_t num_items) {
  cubMin(d_in, d_out, num_items);
}

void cubMin_float(const float *d_in, float *d_out, int64_t num_items) {
  cubMin(d_in, d_out, num_items);
}

void cubMin_double(const double *d_in, double *d_out, int64_t num_items) {
  cubMin(d_in, d_out, num_items);
}
}
