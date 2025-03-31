#ifndef __NN_CLAMP_ZIG__
#define __NN_CLAMP_ZIG__

#include "nn_utils.cu"
#include <cmath>

template<typename T>
void __clamp(
  StreamWrapper w,
  const void* x,
  void* y,
  len_t n,
  T lower,
  T upper
) {
  const auto _stream = __cast_stream(w);
  const auto x_iter = static_cast<const T*>(x);
  const auto y_iter = static_cast<T*>(y);
  thrust::transform(
      thrust::cuda::par.on(_stream), 
      x_iter,  
      x_iter + n,
      y_iter,
      [=] __device__ (T a) -> T { return thrust::min(thrust::max(lower, a), upper); }
  );
}

extern "C" void clamp(
  dtype id,
  StreamWrapper w,
  const void* x,
  void* y,
  len_t n,
  double lower,
  double upper
) {
  switch (id) {
    case SINGLE: {
      return __clamp<f32>(w, x, y, n, static_cast<f32>(lower), static_cast<f32>(upper));
    }
    case DOUBLE: {
      return __clamp<f64>(w, x, y, n, lower, upper);
    }
    default: {
      SYSTEM_EXIT("Unsupported data type");
    }
  }
}

#endif
