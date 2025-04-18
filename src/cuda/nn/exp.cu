#ifndef __NN_EXP_ZIG__
#define __NN_EXP_ZIG__

#include "nn_utils.cu"

template <class T>
void __pow_exp(
  StreamWrapper w,
  const void* x,
  void* y,
  len_t n
) {
    const auto _stream = __cast_stream(w);
    const auto x_iter = static_cast<const T*>(x);
    const auto y_iter = static_cast<T*>(y);
    thrust::transform(
        thrust::cuda::par.on(_stream), 
        x_iter,  
        x_iter + n,
        y_iter,
        [] __device__ (f32 a) -> f32 { return std::exp(a); }
    );
}

extern "C" void pow_exp(
  dtype id,
  StreamWrapper w,
  const void* x,
  void* y,
  len_t n
) {
  switch (id) {
    case SINGLE: return __pow_exp<f32>(w, x, y, n);
    case DOUBLE: return __pow_exp<f64>(w, x, y, n);
    default: SYSTEM_EXIT("Unsupported data type");
  }
}
#endif
