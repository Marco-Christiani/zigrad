#ifndef __BLAS_MAX_ZIG__
#define __BLAS_MAX_ZIG__

#include "blas_utils.cu"

template <typename T>
void __global__ __max_bwd(
  const void* __restrict__ x_val,
  void const* __restrict__ y_val,
  const void* __restrict__ y_grd,
        void* __restrict__ x_grd,
  len_t n
) {
  auto y_v = *static_cast<const T*>(y_val);
  auto y_g = *static_cast<const T*>(y_grd);
  auto _x_val = static_cast<const T*>(x_val);
  auto _x_grd = static_cast<T*>(x_grd);
  thrust::transform(thrust::device, _x_val, _x_val + n, _x_grd, _x_grd, [=](T x, T g) -> T {
    return g + ((x == y_v) ? y_g : T(0));
  });
}


template <typename T>
void __global__ __max_fwd(
  const void* __restrict__ x_val,
        void* __restrict__ y_val,
  len_t n
) {  
  auto src = static_cast<const T*>(x_val);
  auto dst = static_cast<T*>(y_val);
  *dst = *thrust::max_element(thrust::device, src, src + n);
}

extern "C" void max_forward(
  dtype id,
  void* stream,
  const void* x,
  void* y,
  len_t n
) {
  const auto _stream = static_cast<cudaStream_t>(stream);
  switch (id) {
    case SINGLE: return __max_fwd<f32><<<1,1,0,_stream>>>(x, y, n);
    case DOUBLE: return __max_fwd<f64><<<1,1,0,_stream>>>(x, y, n);
    default: SYSTEM_EXIT("Unsupported data type");
  }
  CUDA_ASSERT(cudaPeekAtLastError());
}

extern "C" void max_reverse(
  dtype id,
  void * stream,
  const void* x_val,
  void const* y_val,
  const void* y_grd,
        void* x_grd,
  len_t n
) {
  const auto _stream = static_cast<cudaStream_t>(stream);
  switch (id) {
    case SINGLE: return __max_bwd<f32><<<1,1,0,_stream>>>(x_val, y_val, y_grd, x_grd, n);
    case DOUBLE: return __max_bwd<f64><<<1,1,0,_stream>>>(x_val, y_val, y_grd, x_grd, n);
    default: SYSTEM_EXIT("Unsupported data type");
  }
  CUDA_ASSERT(cudaPeekAtLastError());
}

#endif
