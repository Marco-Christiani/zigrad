#ifndef __BLAS_CLIP_NORM_ZIG__
#define __BLAS_CLIP_NORM_ZIG__

#include "blas_utils.cu"

// TODO: This is a dynamic kernel. Check the performance of this.

template<typename T>
__global__ void __clip_nrm2_kernel(
  void* x,
  len_t n,
  const void* cur_nrm,
  double max_nrm,
  double delta
) {
  const auto _x = static_cast<T*>(x);
  const auto _cur_nrm = *static_cast<const T*>(cur_nrm);
  const auto _max_nrm = static_cast<T>(max_nrm);

  if (_cur_nrm > _max_nrm) {
    const auto _delta = static_cast<T>(delta);
    const auto scale = _max_nrm / (_cur_nrm + _delta);
    thrust::transform(thrust::device, _x, _x + n, _x, [=](T a) -> T { return a * scale; });
  }
}

extern "C" void clip_nrm2(
  dtype id,
  CublasWrapper w,
  void* x,
  void* cur_nrm,
  len_t n,
  double max_nrm,
  double delta
) {
  nrm2(id, w, x, cur_nrm, n);
  const auto stream = __cast_stream(w);

  if (id == SINGLE) {
    __clip_nrm2_kernel<f32><<<1,1,0,stream>>>(x, n, cur_nrm, max_nrm, delta);
  } else if (id == DOUBLE){
    __clip_nrm2_kernel<f64><<<1,1,0,stream>>>(x, n, cur_nrm, max_nrm, delta);
  } else {
    SYSTEM_EXIT("Unsupported data type");
  }

  CUDA_ASSERT(cudaPeekAtLastError());
}

#endif
