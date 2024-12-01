#ifndef __BLAS_SUM_ZIG__
#define __BLAS_SUM_ZIG__

#include "blas_utils.cu"

EXTERN_C void reduce_sum(
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
        static_cast<float>(0),
        thrust::plus<float>()
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
        static_cast<double>(0),
        thrust::plus<double>()
      );
      return;
    }
  }
}

#endif
