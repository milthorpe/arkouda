#ifndef _CUB_HISTOGRAM_H_
#define _CUB_HISTOGRAM_H_
#include <stdint.h>
void cubHistogram_int32(const int32_t *d_samples, unsigned long long int *d_histogram, int num_levels, int32_t lower_bound, int32_t upper_bound, int64_t N);
void cubHistogram_int64(const int64_t *d_samples, unsigned long long int *d_histogram, int num_levels, int64_t lower_bound, int64_t upper_bound, int64_t N);
void cubHistogram_float(const float *d_samples, unsigned long long int *d_histogram, int num_levels, float lower_bound, float upper_bound, int64_t N);
void cubHistogram_double(const double *d_samples, unsigned long long int *d_histogram, int num_levels, double lower_bound, double upper_bound, int64_t N);
#endif
