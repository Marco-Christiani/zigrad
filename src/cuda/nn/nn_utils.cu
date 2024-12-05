#ifndef __NN_UTILS_ZIG__
#define __NN_UTILS_ZIG__
#include "../nn_conflux.h"
#include "../cuda_helpers.cu"
#define CUDNN_DTYPE(id) ((id) == SINGLE) ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE

inline cudnnSoftmaxAlgorithm_t SMAX_OP_TYPE(smaxtype op) {
  switch (op) {
    case SMAX_FAST:
      return cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_FAST;
    case SMAX_MAX:
      return cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE;
    default:
      return cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_LOG;
    } 
}
  


#endif
