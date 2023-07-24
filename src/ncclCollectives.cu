#include "rccl/rccl.h"
#include <stdio.h>

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

extern "C" {
void *gpuCommGetUniqueID() {
  ncclUniqueId *unique_id = new ncclUniqueId();
  NCCLCHECK(ncclGetUniqueId(unique_id));
  return unique_id;
}

void *gpuCommInitRank(int num_ranks, void *comm_id, int rank) {
  ncclUniqueId *comm_unique_id = (ncclUniqueId*) comm_id;
  ncclComm_t *comm = new ncclComm_t();
  NCCLCHECK(ncclCommInitRank(comm, num_ranks, *comm_unique_id, rank));
  return comm;
}

void gpuCommDestroy(void *comm_ptr) {
  ncclComm_t *comm = (ncclComm_t*)comm_ptr;
  ncclCommDestroy(*comm);
  delete(comm);
}

void gpuReduce_sum_int32(const int32_t *sendbuff, int32_t *recvbuff, size_t count, int root, void *comm_ptr) {
  NCCLCHECK(ncclReduce(sendbuff, recvbuff, count, ncclInt32, ncclSum, root, *(ncclComm_t*)comm_ptr, 0));
}

void gpuReduce_sum_int64(const int64_t *sendbuff, int64_t *recvbuff, size_t count, int root, void *comm_ptr) {
  NCCLCHECK(ncclReduce(sendbuff, recvbuff, count, ncclInt64, ncclSum, root, *(ncclComm_t*)comm_ptr, 0));
}

void gpuReduce_sum_float(const float *sendbuff, float *recvbuff, size_t count, int root, void *comm_ptr) {
  NCCLCHECK(ncclReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, root, *(ncclComm_t*)comm_ptr, 0));
}

void gpuReduce_sum_double(const double *sendbuff, double *recvbuff, size_t count, int root, void *comm_ptr) {
  NCCLCHECK(ncclReduce(sendbuff, recvbuff, count, ncclDouble, ncclSum, root, *(ncclComm_t*)comm_ptr, 0));
}

void gpuReduce_min_int32(const int32_t *sendbuff, int32_t *recvbuff, size_t count, int root, void *comm_ptr) {
  NCCLCHECK(ncclReduce(sendbuff, recvbuff, count, ncclInt32, ncclMin, root, *(ncclComm_t*)comm_ptr, 0));
}

void gpuReduce_min_int64(const int64_t *sendbuff, int64_t *recvbuff, size_t count, int root, void *comm_ptr) {
  NCCLCHECK(ncclReduce(sendbuff, recvbuff, count, ncclInt64, ncclMin, root, *(ncclComm_t*)comm_ptr, 0));
}

void gpuReduce_min_float(const float *sendbuff, float *recvbuff, size_t count, int root, void *comm_ptr) {
  NCCLCHECK(ncclReduce(sendbuff, recvbuff, count, ncclFloat, ncclMin, root, *(ncclComm_t*)comm_ptr, 0));
}

void gpuReduce_min_double(const double *sendbuff, double *recvbuff, size_t count, int root, void *comm_ptr) {
  NCCLCHECK(ncclReduce(sendbuff, recvbuff, count, ncclDouble, ncclMin, root, *(ncclComm_t*)comm_ptr, 0));
}

void gpuReduce_max_int32(const int32_t *sendbuff, int32_t *recvbuff, size_t count, int root, void *comm_ptr) {
  NCCLCHECK(ncclReduce(sendbuff, recvbuff, count, ncclInt32, ncclMax, root, *(ncclComm_t*)comm_ptr, 0));
}

void gpuReduce_max_int64(const int64_t *sendbuff, int64_t *recvbuff, size_t count, int root, void *comm_ptr) {
  NCCLCHECK(ncclReduce(sendbuff, recvbuff, count, ncclInt64, ncclMax, root, *(ncclComm_t*)comm_ptr, 0));
}

void gpuReduce_max_float(const float *sendbuff, float *recvbuff, size_t count, int root, void *comm_ptr) {
  NCCLCHECK(ncclReduce(sendbuff, recvbuff, count, ncclFloat, ncclMax, root, *(ncclComm_t*)comm_ptr, 0));
}

void gpuReduce_max_double(const double *sendbuff, double *recvbuff, size_t count, int root, void *comm_ptr) {
  NCCLCHECK(ncclReduce(sendbuff, recvbuff, count, ncclDouble, ncclMax, root, *(ncclComm_t*)comm_ptr, 0));
}

void gpuAllReduce_sum_int32(const int32_t *sendbuff, int32_t *recvbuff, size_t count, void *comm_ptr) {
  NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, ncclInt32, ncclSum, *(ncclComm_t*)comm_ptr, 0));
}

void gpuAllReduce_sum_int64(const int64_t *sendbuff, int64_t *recvbuff, size_t count, void *comm_ptr) {
  NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, ncclInt64, ncclSum, *(ncclComm_t*)comm_ptr, 0));
}

void gpuAllReduce_sum_float(const float *sendbuff, float *recvbuff, size_t count, void *comm_ptr) {
  NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, *(ncclComm_t*)comm_ptr, 0));
}

void gpuAllReduce_sum_double(const double *sendbuff, double *recvbuff, size_t count, void *comm_ptr) {
  NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, ncclDouble, ncclSum, *(ncclComm_t*)comm_ptr, 0));
}
}
