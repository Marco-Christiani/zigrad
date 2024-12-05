#ifndef __BLAS_AXPY_ZIG__
#define __BLAS_AXPY_ZIG__

#include "blas_utils.cu"

// we're using double because every float can cast
// up to a double and then we can go back down.

extern "C" void axpy(
  dtype id,
  void* cublas_handle,
  const void* x,
  void* y,
  len_t n, 
  const void* alpha
) {
  const int _n = static_cast<int>(n);
  
  switch (id) {
    case SINGLE: {
      const auto _alpha = static_cast<const float*>(alpha);
      return CUBLAS_ASSERT(cublasSaxpy(
          get_handle(cublas_handle),  
          _n, 
          static_cast<const float*>(alpha),
          static_cast<const float*>(x), 1,
          static_cast<float*>(y), 1
      ));
    }
    case DOUBLE: {
      return CUBLAS_ASSERT(cublasDaxpy(
          get_handle(cublas_handle),  
          _n,
          static_cast<const double*>(alpha),
          static_cast<const double*>(x), 1,
          static_cast<double*>(y), 1
      ));
    }
  }
}

#endif
