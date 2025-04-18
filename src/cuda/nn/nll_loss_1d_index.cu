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
  CudnnWrapper w,
  const void* x,
  len_t trg,
  void* y,
  len_t n,
  reduxtype reduxop
) {
  const auto stream = __cast_stream(w);
  const auto _src = static_cast<const T*>(x);
  const auto _dst = static_cast<T*>(y);
  __nll_loss_1D_index_reduce_kernel<T><<<1,1,0,stream>>>(_src, trg, _dst, n, reduxop);
}

extern "C" void nll_loss_1D_index_fwd(
  dtype id,
  CudnnWrapper w,
  void* src,
  len_t trg,
  void* dst,
  len_t n,
  bool inplace_smax,
  reduxtype reduxop
) {
  if (inplace_smax) {
    smax_vec_fwd(id, w, src, src, n, SMAX_LOG);
  }
  if (id == SINGLE) {
      __nll_loss_1D_index_reduce<f32>(w, src, trg, dst, n, reduxop);
  } else if (id == DOUBLE){
      __nll_loss_1D_index_reduce<f64>(w, src, trg, dst, n, reduxop);
  } else {
     SYSTEM_EXIT("Unsupported data type");
  }

  CUDA_ASSERT(cudaPeekAtLastError());
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
void __nll_loss_1D_index_bwd(
  CudnnWrapper w,
  const void* x_val,
  void *x_grd,
  len_t trg,
  len_t n,
  reduxtype reduxop
) {
  const auto stream = __cast_stream(w);
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

extern "C" void nll_loss_1D_index_bwd(
  dtype id,
  CudnnWrapper w,
  const void* x_val,
  void *x_grd,
  len_t trg,
  len_t n,
  reduxtype reduxop
) {

  if (id == SINGLE) {
      __nll_loss_1D_index_bwd<f32>(w, x_val, x_grd, trg, n, reduxop);
  } else if (id == DOUBLE){
      __nll_loss_1D_index_bwd<f64>(w, x_val, x_grd, trg, n, reduxop);
  } else {
     SYSTEM_EXIT("Unsupported data type");
  }

  CUDA_ASSERT(cudaPeekAtLastError());
}

#endif
