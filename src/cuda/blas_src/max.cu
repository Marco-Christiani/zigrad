#ifndef __BLAS_MAX_ZIG__
#define __BLAS_MAX_ZIG__

#include "blas_utils.cu"
#include <limits>

extern "C" void reduce_max(
  dtype id,
  void* stream,
  const void* a_data,
  void* result,
  len_t n
) {
  const auto _stream = static_cast<cudaStream_t>(stream);

  switch (id) {
    case SINGLE: {
      const auto _result = static_cast<float*>(result);
      const auto iter = static_cast<const float*>(a_data);
      *_result = thrust::reduce(
          thrust::cuda::par.on(_stream), 
          iter,  
          iter + n,
          std::numeric_limits<float>::lowest(),
          thrust::maximum<float>()
      );
      return;
    }
    case DOUBLE: {
      const auto _result = static_cast<double*>(result);
      const auto iter = static_cast<const double*>(a_data);
      *_result = thrust::reduce(
          thrust::cuda::par.on(_stream), 
          iter,  
          iter + n,
          std::numeric_limits<double>::lowest(),
          thrust::maximum<double>()
      );
      return;
    }
  }
}

#endif
