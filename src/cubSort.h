#ifndef _CUB_SORT_H_
#define _CUB_SORT_H_
void cubSortPairs_int32(int32_t *d_keys_in, int32_t *d_keys_out, int64_t *d_values_in, int64_t *d_values_out, size_t N);
void cubSortPairs_int64(int64_t *d_keys_in, int64_t *d_keys_out, int64_t *d_values_in, int64_t *d_values_out, size_t N);
void cubSortPairs_float(float *d_keys_in, float *d_keys_out, int64_t *d_values_in, int64_t *d_values_out, size_t N);
void cubSortPairs_double(double *d_keys_in, double *d_keys_out, int64_t *d_values_in, int64_t *d_values_out, size_t N);
#endif
