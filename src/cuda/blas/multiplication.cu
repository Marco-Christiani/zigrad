#ifndef __BLAS_MULTIPLY_ZIG__
#define __BLAS_MULTIPLY_ZIG__

#include "blas_utils.cu"

// we're using double because every float can cast
// up to a double and then we can go back down.

template <class T>
void __multiplication(
  void* stream,
  const void* x,
  const void* y,
  void* z,
  len_t n
) {  
  const auto _stream = static_cast<cudaStream_t>(stream);
  const auto iter_x = static_cast<const T*>(x);
  const auto iter_y = static_cast<const T*>(y);
  const auto iter_z = static_cast<T*>(z);
  thrust::transform(
    thrust::cuda::par.on(_stream), 
    iter_x,  
    iter_x + n,
    iter_y,
    iter_z,
    thrust::multiplies<T>()
  );
}

extern "C" void multiplication(
  dtype id,
  void* stream,
  const void* x,
  const void* y,
  void* z,
  len_t n
) {

  switch (id) {
    case SINGLE: {
        return __multiplication<f32>(stream, x, y, z, n);
    }
    case DOUBLE: {
        return __multiplication<f64>(stream, x, y, z, n);
    }
  }
  CUDA_ASSERT(cudaPeekAtLastError());
}

#endif
