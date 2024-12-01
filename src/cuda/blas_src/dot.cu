#ifndef __BLAS_DOT_ZIG__
#define __BLAS_DOT_ZIG__

#include "blas_utils.cu"

// we're using double because every float can cast
// up to a double and then we can go back down.

extern "C" void dot(
  dtype id,
  void* stream,
  const void* a_data,
  const void* b_data,
  void* result,
  len_t n
) {
  const auto _stream = static_cast<cudaStream_t>(stream);

  switch (id) {
    case SINGLE: {
      const auto iter_a = static_cast<const float*>(a_data);
      const auto iter_b = static_cast<const float*>(b_data);
      const auto _result = static_cast<float*>(result);
      *_result = thrust::inner_product(
        thrust::cuda::par.on(_stream), 
        iter_a,  
        iter_a + n,
        iter_b,
        static_cast<float>(0)
      );
      return;
    }
    case DOUBLE: {
      const auto iter_a = static_cast<const double*>(a_data);
      const auto iter_b = static_cast<const double*>(b_data);
      const auto _result = static_cast<double*>(result);
      *_result = thrust::inner_product(
        thrust::cuda::par.on(_stream), 
        iter_a,  
        iter_a + n,
        iter_b,
        static_cast<double>(0)
      );
      return;
    }
  }
}

#endif
