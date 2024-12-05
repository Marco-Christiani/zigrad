
#include "nn_utils.cu"

template <typename T>
__global__ void __nll_loss_1D_index_reduce_kernel(
    const T* src,
    len_t trg,
    len_t n,
    T* total_rdx,
    int ignore_idx,
    reduxtype reduxop
) {
  if (static_cast<int>(trg) == ignore_idx) {
    *total_rdx = T(0);
    return;
  }
  // NONE means we didn't reduce at all
  // so it's not actually an option here
  if (reduxop == RDX_SUM) {
      *total_rdx = -src[trg];
  } else {
      *total_rdx = -src[trg] / T(n);
  }
}

template<typename T>
void __nll_loss_1D_index_reduce(
  void* cudnn_handle,
  void* src,
  len_t trg,
  len_t n,
  void* rdx,
  int ignore_idx,
  reduxtype reduxop
) {
  cudaStream_t stream;
  CUDNN_ASSERT(cudnnGetStream(static_cast<cudnnHandle_t>(cudnn_handle), &stream));
  const auto _src = static_cast<T*>(src);
  const auto _rdx = static_cast<T*>(rdx);
  __nll_loss_1D_index_reduce_kernel<T><<<1,1,0,stream>>>(_src, trg, n, _rdx, ignore_idx, reduxop);
  return;
}

extern "C" void nll_loss_1D_index_reduce(
  dtype id,
  void* cudnn_handle,
  void* src,
  len_t trg,
  len_t n,
  void* rdx,
  int ignore_idx,
  reduxtype reduxop
) {
  smax_vec_forward(id, cudnn_handle, src, src, n);

  switch (id) {
    case SINGLE:
      return __nll_loss_1D_index_reduce<f32>(cudnn_handle, src, trg, n, rdx, ignore_idx, reduxop);
    case DOUBLE:
      return __nll_loss_1D_index_reduce<f64>(cudnn_handle, src, trg, n, rdx, ignore_idx, reduxop);
  }  
}

template<typename T>
struct NLLBwdFunctor {

    len_t trg;
    T denom;

    __host__ __device__ 
    NLLBwdFunctor() = default;
    __host__ __device__ 
    NLLBwdFunctor(const NLLBwdFunctor& other) = default;
    __host__ __device__
    NLLBwdFunctor(len_t _trg, T _denom) : trg{_trg}, denom{_denom} {}
  
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
  cudaStream_t stream;
  CUDNN_ASSERT(cudnnGetStream(static_cast<cudnnHandle_t>(cudnn_handle), &stream));
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
    NLLBwdFunctor<T>(trg, denom)
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
