#ifndef __NN_UTILS_ZIG__
#define __NN_UTILS_ZIG__
#include "../nn_conflux.h"
#include "../cuda_helpers.cu"
#define CUDNN_DTYPE(id) ((id) == SINGLE) ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE
#endif
