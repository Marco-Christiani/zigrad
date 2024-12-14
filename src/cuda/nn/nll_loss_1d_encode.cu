#ifndef __NN_NLL_ZIG__
#define __NN_NLL_ZIG__

#include "nn_utils.cu"
#include <limits>

template <typename T>
__global__ void __nll_loss_1D_encode_reduce_kernel(
    const T* src,
    const T* trg,
    T* dst,
    len_t n,
    reduxtype reduxop
) {
  if (reduxop == RDX_NONE) {
    *dst = std::numeric_limits<T>::quiet_NaN();
    return;
  }
  
  // expects encodings to be either 0 or 1...
  *dst = -thrust::inner_product(thrust::device, src, src + n, trg, T(0));
  
  if (reduxop == RDX_MEAN) {
      *dst /= T(n);
  } 
}

template<typename T>
void __nll_loss_1D_encode_reduce(
  void* cudnn_handle,
  const void* x_val,
  const void* y_val,
  void* dst,
  len_t n,
  reduxtype reduxop
) {
  const auto stream = __cudnn_stream(cudnn_handle);
  const auto _x_val = static_cast<const T*>(x_val);
  const auto _y_val = static_cast<const T*>(y_val);
  const auto _dst = static_cast<T*>(dst);
  __nll_loss_1D_encode_reduce_kernel<T><<<1,1,0,stream>>>(_x_val, _y_val, _dst, n, reduxop);
}

extern "C" void nll_loss_1D_encode_forward(
  dtype id,
  void* cudnn_handle,
  void* src,
  const void* trg,
  void* dst,
  len_t n,
  bool inplace_smax,
  reduxtype reduxop
) {
  if (inplace_smax) {
    smax_vec_forward(id, cudnn_handle, src, src, n, SMAX_LOG);
  }
  if (id == SINGLE) {
      __nll_loss_1D_encode_reduce<f32>(cudnn_handle, src, trg, dst, n, reduxop);
  } else {
      __nll_loss_1D_encode_reduce<f64>(cudnn_handle, src, trg, dst, n, reduxop);
  }  

  CUDA_ASSERT(cudaPeekAtLastError());
}

template<typename T>
struct NLLBwdFunctor {

    T denom;

    __device__
    T operator()(T src_val, T trg_val) const {
        return (src_val - trg_val) / this->denom;
    }
};

template<typename T>
void __nll_loss_1D_encode_reverse(
  void* cudnn_handle,
  const void* x_val,
  const void* y_val,
  void *x_grd,
  len_t n,
  reduxtype reduxop
) {
  const auto stream = __cudnn_stream(cudnn_handle);
  const auto x_val_iter = static_cast<const T*>(x_val);
  const auto y_val_iter = static_cast<const T*>(y_val);
  const auto x_grd_iter = static_cast<T*>(x_grd);
  const T denom = (reduxop == RDX_SUM) ? T(1) : static_cast<T>(n);
  thrust::transform(
    thrust::cuda::par.on(stream), 
    x_val_iter,
    x_val_iter + n,
    y_val_iter,
    x_grd_iter,
    NLLBwdFunctor<T>{ .denom = denom }
  );
}

extern "C" void nll_loss_1D_encode_reverse(
  dtype id,
  void* cudnn_handle,
  const void* x_val,
  const void* y_val,
  void *x_grd,
  len_t n,
  reduxtype reduxop
) {

  if (id == SINGLE) {
      __nll_loss_1D_encode_reverse<f32>(cudnn_handle, x_val, y_val, x_grd, n, reduxop);
  } else {
      __nll_loss_1D_encode_reverse<f64>(cudnn_handle, x_val, y_val, x_grd, n, reduxop);
  }

  CUDA_ASSERT(cudaPeekAtLastError());
}

#endif
