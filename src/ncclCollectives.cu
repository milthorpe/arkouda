#include "nccl.h"
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

void gpuReduce_sum_int32(const void *sendbuff, void *recvbuff, size_t count, int root, void *comm_ptr) {
  NCCLCHECK(ncclReduce(sendbuff, recvbuff, count, ncclInt32, ncclSum, root, *(ncclComm_t*)comm_ptr, 0));
}

void gpuReduce_sum_in64(const void *sendbuff, void *recvbuff, size_t count, int root, void *comm_ptr) {
  NCCLCHECK(ncclReduce(sendbuff, recvbuff, count, ncclInt64, ncclSum, root, *(ncclComm_t*)comm_ptr, 0));
}

void gpuReduce_sum_float(const void *sendbuff, void *recvbuff, size_t count, int root, void *comm_ptr) {
  NCCLCHECK(ncclReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, root, *(ncclComm_t*)comm_ptr, 0));
}

void gpuReduce_sum_double(const void *sendbuff, void *recvbuff, size_t count, int root, void *comm_ptr) {
  NCCLCHECK(ncclReduce(sendbuff, recvbuff, count, ncclDouble, ncclSum, root, *(ncclComm_t*)comm_ptr, 0));
}

void gpuAllReduce_sum_int32(const void *sendbuff, void *recvbuff, size_t count, void *comm_ptr) {
  NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, ncclInt32, ncclSum, *(ncclComm_t*)comm_ptr, 0));
}

void gpuAllReduce_sum_in64(const void *sendbuff, void *recvbuff, size_t count, void *comm_ptr) {
  NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, ncclInt64, ncclSum, *(ncclComm_t*)comm_ptr, 0));
}

void gpuAllReduce_sum_float(const void *sendbuff, void *recvbuff, size_t count, void *comm_ptr) {
  NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, *(ncclComm_t*)comm_ptr, 0));
}

void gpuAllReduce_sum_double(const void *sendbuff, void *recvbuff, size_t count, void *comm_ptr) {
  NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, ncclDouble, ncclSum, *(ncclComm_t*)comm_ptr, 0));
}
}
