#ifndef _CUB_MAX_H_
#define _CUB_MAX_H_

void cubMax_int32(const int32_t *d_in, int32_t *d_out, int64_t num_items);
void cubMax_int64(const int64_t *d_in, int64_t *d_out, int64_t num_items);
void cubMax_float(const float *d_in, float *d_out, int64_t num_items);
void cubMax_double(const double *d_in, double *d_out, int64_t num_items);

#endif
