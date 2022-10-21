#ifndef _CUB_MIN_H_
#define _CUB_MIN_H_

void cubMin_int32(const int32_t *d_in, int32_t *d_out, int64_t num_items);
void cubMin_int64(const int64_t *d_in, int64_t *d_out, int64_t num_items);
void cubMin_float(const float *d_in, float *d_out, int64_t num_items);
void cubMin_double(const double *d_in, double *d_out, int64_t num_items);

#endif
