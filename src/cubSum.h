#ifndef _CUB_SUM_H_
#define _CUB_SUM_H_

void cubSum_int32(const int32_t *d_in, int32_t *d_out, int64_t num_items);
void cubSum_int64(const int64_t *d_in, int64_t *d_out, int64_t num_items);
void cubSum_float(const float *d_in, float *d_out, int64_t num_items);
void cubSum_double(const double *d_in, double *d_out, int64_t num_items);

#endif
