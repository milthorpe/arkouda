#include <cub/cub.cuh>
#include <stdio.h>

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
}