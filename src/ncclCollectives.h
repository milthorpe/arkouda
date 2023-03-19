#ifndef _NCCL_COLLECTIVES_H_
#define _NCCL_COLLECTIVES_H_
#include <stdint.h>
void *gpuCommGetUniqueID(void);
void *gpuCommInitRank(int num_ranks, void* comm_id, int rank);
void gpuCommDestroy(void *comm_ptr);
void gpuReduce_sum_int32(const int32_t *sendbuff, int32_t *recvbuff, size_t count, int root, void *comm_ptr);
void gpuReduce_sum_int64(const int64_t *sendbuff, int64_t *recvbuff, size_t count, int root, void *comm_ptr);
void gpuReduce_sum_float(const float *sendbuff, float *recvbuff, size_t count, int root, void *comm_ptr);
void gpuReduce_sum_double(const double *sendbuff, double *recvbuff, size_t count, int root, void *comm_ptr);
void gpuReduce_min_int32(const int32_t *sendbuff, int32_t *recvbuff, size_t count, int root, void *comm_ptr);
void gpuReduce_min_int64(const int64_t *sendbuff, int64_t *recvbuff, size_t count, int root, void *comm_ptr);
void gpuReduce_min_float(const float *sendbuff, float *recvbuff, size_t count, int root, void *comm_ptr);
void gpuReduce_min_double(const double *sendbuff, double *recvbuff, size_t count, int root, void *comm_ptr);
void gpuReduce_max_int32(const int32_t *sendbuff, int32_t *recvbuff, size_t count, int root, void *comm_ptr);
void gpuReduce_max_int64(const int64_t *sendbuff, int64_t *recvbuff, size_t count, int root, void *comm_ptr);
void gpuReduce_max_float(const float *sendbuff, float *recvbuff, size_t count, int root, void *comm_ptr);
void gpuReduce_max_double(const double *sendbuff, double *recvbuff, size_t count, int root, void *comm_ptr);
void gpuAllReduce_sum_int32(const int32_t *sendbuff, int32_t *recvbuff, size_t count, void *comm_ptr);
void gpuAllReduce_sum_int64(const int64_t *sendbuff, int64_t *recvbuff, size_t count, void *comm_ptr);
void gpuAllReduce_sum_float(const float *sendbuff, float *recvbuff, size_t count, void *comm_ptr);
void gpuAllReduce_sum_double(const double *sendbuff, double *recvbuff, size_t count, void *comm_ptr);
#endif
