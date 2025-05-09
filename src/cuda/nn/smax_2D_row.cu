#ifndef __NN_SMAX_2D_ROW_ZIG__
#define __NN_SMAX_2D_ROW_ZIG__

#include "nn_utils.cu"

void __smax_2D_row_fwd(
  dtype id,
  CudnnWrapper w,
  const void* x,
  void* y,
  len_t m,
  len_t n
) {
  const int _m = static_cast<int>(m);
  const int _n = static_cast<int>(n);

  cudnnTensorDescriptor_t desc;
  cudnnCreateTensorDescriptor(&desc);
  cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DTYPE(id), _m, 1, 1, _n);

  switch (id) {
    case SINGLE: {
      const float alpha = 1.0f;
      const float beta = 0.0f;
      return CUDNN_ASSERT(cudnnSoftmaxForward(
        __cast_cudnn(w),
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha, desc, x,
        &beta, desc, y
      ));
    }
    case DOUBLE: {
      const double alpha = 1.0;
      const double beta = 0.0;
      return CUDNN_ASSERT(cudnnSoftmaxForward(
        __cast_cudnn(w),
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha, desc, x,
        &beta, desc, y
      ));
    }
    default:
      SYSTEM_EXIT("Unsupported data type");
  }
}


void smax_2D_row_fwd(
  dtype id,
  CudnnWrapper w,
  const void* x,
  void* y,
  len_t m,
  len_t n
) {
  __smax_2D_row_fwd(id, w, x, y, m, n);
  CUDA_ASSERT(cudaPeekAtLastError());
}

void __smax_2D_row_bwd(
  dtype id,
  CudnnWrapper w,
  const void* y_val,
  const void* y_grd,
  void* x_grd,
  len_t m,
  len_t n
) {
  const int _m = static_cast<int>(m);
  const int _n = static_cast<int>(n);

  cudnnTensorDescriptor_t desc;
  cudnnCreateTensorDescriptor(&desc);
  cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DTYPE(id), _m, 1, 1, _n);

  switch (id) {
    case SINGLE: {
      const float alpha = 1.0f;
      const float beta = 1.0f;
      return CUDNN_ASSERT(cudnnSoftmaxBackward(
        __cast_cudnn(w),
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
        __cast_cudnn(w),
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha, desc, y_val, desc, y_grd,
        &beta, desc, x_grd
      ));
    }
    default:
      SYSTEM_EXIT("Unsupported data type");
  }
}

extern "C" void smax_2D_row_bwd(
  dtype id,
  CudnnWrapper w,
  const void* y_val,
  const void* y_grd,
  void* x_grd,
  len_t m,
  len_t n
){
  __smax_2D_row_bwd(id, w, y_val, y_grd, x_grd, m, n);
  CUDA_ASSERT(cudaPeekAtLastError());
}

#endif
