#ifndef __NN_SMAX_VEC_ZIG__
#define __NN_SMAX_VEC_ZIG__

#include "nn_utils.cu"
#include <numeric>
#include <limits>

extern "C" void smax_vec_forward(
  dtype id,
  void* cudnn_handle,
  const void* x,
  void* y,
  len_t n,
  smaxtype op
) {
  const auto _cudnn_handle = static_cast<cudnnHandle_t>(cudnn_handle);
  const int _n = static_cast<int>(n);

  cudnnTensorDescriptor_t desc;
  cudnnCreateTensorDescriptor(&desc);
  cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DTYPE(id), 1, 1, 1, _n);

  switch (id) {
    case SINGLE: {
      const float alpha = 1.0f;
      const float beta = 0.0f;
      return CUDNN_ASSERT(cudnnSoftmaxForward(
        _cudnn_handle,
        SMAX_OP_TYPE(op),
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha, desc, x,
        &beta, desc, y
      ));
    }
    case DOUBLE: {
      const double alpha = 1.0;
      const double beta = 0.0;
      return CUDNN_ASSERT(cudnnSoftmaxForward(
        _cudnn_handle,
        SMAX_OP_TYPE(op),
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha, desc, x,
        &beta, desc, y
      ));
    }
  }
}

extern "C" void softmax_vec_reverse(
  dtype id,
  void* cudnn_handle,
  const void* y_val,
  const void* y_grd,
  void* x_grd,
  len_t n
) {
  const auto _cudnn_handle = static_cast<cudnnHandle_t>(cudnn_handle);
  const int _n = static_cast<int>(n);

  cudnnTensorDescriptor_t desc;
  cudnnCreateTensorDescriptor(&desc);
  cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DTYPE(id), 1, 1, 1, _n);
  
  switch (id) {
    case SINGLE: {
      const float alpha = 1.0f;
      const float beta = 1.0f;
      return CUDNN_ASSERT(cudnnSoftmaxBackward(
        _cudnn_handle,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha, desc, y_val, desc, y_grd,
        &beta, desc, x_grd
      ));
    }
    case DOUBLE: {
      const double alpha = 1.0;
      const double beta = 1.0;
      return CUDNN_ASSERT(cudnnSoftmaxBackward(
        _cudnn_handle,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha, desc, y_val, desc, y_grd,
        &beta, desc, x_grd
      ));
    }
  }
}

#endif