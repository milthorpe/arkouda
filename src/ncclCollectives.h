#ifndef _NCCL_COLLECTIVES_H_
#define _NCCL_COLLECTIVES_H_
#include <stdint.h>
void *gpuCommGetUniqueID();
void *gpuCommInitRank(int num_ranks, void* comm_id, int rank);
void gpuCommDestroy(void *comm_ptr);
void gpuReduce_sum_int32(const void *sendbuff, void *recvbuff, size_t count, int root, void *comm_ptr);
void gpuReduce_sum_int64(const void *sendbuff, void *recvbuff, size_t count, int root, void *comm_ptr);
void gpuReduce_sum_float(const void *sendbuff, void *recvbuff, size_t count, int root, void *comm_ptr);
void gpuReduce_sum_double(const void *sendbuff, void *recvbuff, size_t count, int root, void *comm_ptr);
void gpuAllReduce_sum_int32(const void *sendbuff, void *recvbuff, size_t count, void *comm_ptr);
void gpuAllReduce_sum_int64(const void *sendbuff, void *recvbuff, size_t count, void *comm_ptr);
void gpuAllReduce_sum_float(const void *sendbuff, void *recvbuff, size_t count, void *comm_ptr);
void gpuAllReduce_sum_double(const void *sendbuff, void *recvbuff, size_t count, void *comm_ptr);
#endif
