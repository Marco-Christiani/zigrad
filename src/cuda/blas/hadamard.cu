#ifndef __BLAS_HADAMARD_ZIG__
#define __BLAS_HADAMARD_ZIG__

#include "blas_utils.cu"

// we're using double because every float can cast
// up to a double and then we can go back down.

extern "C" void hadamard(
  dtype id,
  void* stream,
  const void* x,
  const void* y,
  void* z,
  len_t n
) {
  const auto _stream = static_cast<cudaStream_t>(stream);

  switch (id) {
    case SINGLE: {
      const auto iter_x = static_cast<const float*>(x);
      const auto iter_y = static_cast<const float*>(y);
      const auto iter_z = static_cast<float*>(z);
      thrust::transform(
        thrust::cuda::par.on(_stream), 
        iter_x,  
        iter_x + n,
        iter_y,
        iter_z,
        thrust::multiplies<float>()
      );
      return;
    }
    case DOUBLE: {
      const auto iter_x = static_cast<const double*>(x);
      const auto iter_y = static_cast<const double*>(y);
      const auto iter_z = static_cast<double*>(z);
      thrust::transform(
        thrust::cuda::par.on(_stream), 
        iter_x,  
        iter_x + n,
        iter_y,
        iter_z,
        thrust::multiplies<double>()
      );
      return;
    }
  }
}

#endif
