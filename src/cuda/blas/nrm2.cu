#ifndef __BLAS_NRM2_ZIG__
#define __BLAS_NRM2_ZIG__

#include "blas_utils.cu"

extern "C" void nrm2(
  dtype id,
  void* cublas_handle,
  const void* x,
  void* y,
  len_t n
) {
  const int _n = static_cast<int>(n);
  
  switch (id) {
    case SINGLE: {
      return CUBLAS_ASSERT(cublasSnrm2(
          get_handle(cublas_handle),  
          _n,
          static_cast<const float*>(x), 1,
          static_cast<float*>(y)
      ));
    }
    case DOUBLE: {
      return CUBLAS_ASSERT(cublasDnrm2(
          get_handle(cublas_handle),  
          _n,
          static_cast<const double*>(x), 1,
          static_cast<double*>(y)
      ));
    }
  }
}

#endif
