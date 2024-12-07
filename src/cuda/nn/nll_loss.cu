#ifndef __NN_NLL_ZIG__
#define __NN_NLL_ZIG__

#include "nn_utils.cu"
#include <limits>

template <typename T>
__global__ void __nll_loss_1D_index_reduce_kernel(
    const T* src,
    len_t trg,
    T* dst,
    len_t n,
    reduxtype reduxop
) {
  if (reduxop == RDX_SUM) {
      *dst = -src[trg];
  } else {
      *dst = -src[trg] / T(n);
  }
}

template<typename T>
void __nll_loss_1D_index_reduce(
  void* cudnn_handle,
  const void* x,
  len_t trg,
  void* y,
  len_t n,
  reduxtype reduxop
) {
  const auto stream = __cudnn_stream(cudnn_handle);
  const auto _src = static_cast<const T*>(x);
  const auto _dst = static_cast<T*>(y);
  __nll_loss_1D_index_reduce_kernel<T><<<1,1,0,stream>>>(_src, trg, _dst, n, reduxop);
}

extern "C" void nll_loss_1D_index_forward(
  dtype id,
  void* cudnn_handle,
  void* src,
  len_t trg,
  void* dst,
  len_t n,
  bool inplace_smax,
  reduxtype reduxop
) {
  if (inplace_smax) {
    smax_vec_forward(id, cudnn_handle, src, src, n, SMAX_LOG);
  }
  if (id == SINGLE) {
      __nll_loss_1D_index_reduce<f32>(cudnn_handle, src, trg, dst, n, reduxop);
  } else {
      __nll_loss_1D_index_reduce<f64>(cudnn_handle, src, trg, dst, n, reduxop);
  }  
}

template<typename T>
struct NLLBwdFunctor {

    len_t trg;
    T denom;

    __device__
    T operator()(T src_val, len_t idx) const {
        const T tmp = (idx == this->trg) ? T(1) : T(0);
        return (src_val - tmp) / this->denom;
    }
};

template<typename T>
void __nll_loss_1D_index_reverse(
  void* cudnn_handle,
  const void* x_val,
  void *x_grd,
  len_t trg,
  len_t n,
  reduxtype reduxop
) {
  const auto stream = __cudnn_stream(cudnn_handle);
  const auto idx_iter = thrust::make_counting_iterator(0UL);
  const auto _x_val_iter = static_cast<const T*>(x_val);
  const auto _x_grd_iter = static_cast<T*>(x_grd);
  const T denom = (reduxop == RDX_SUM) ? T(1) : static_cast<T>(n);
  thrust::transform(
    thrust::cuda::par.on(stream), 
    _x_val_iter,
    _x_val_iter + n,
    idx_iter,
    _x_grd_iter,
    NLLBwdFunctor<T>{ .trg = trg, .denom = denom }
  );
}

extern "C" void nll_loss_1D_index_reverse(
  dtype id,
  void* cudnn_handle,
  const void* x_val,
  void *x_grd,
  len_t trg,
  len_t n,
  reduxtype reduxop
) {
  switch (id) {
    case SINGLE:
      return __nll_loss_1D_index_reverse<f32>(cudnn_handle, x_val, x_grd, trg, n, reduxop);
    case DOUBLE: 
      return __nll_loss_1D_index_reverse<f64>(cudnn_handle, x_val, x_grd, trg, n, reduxop);
  }
}

#endif
