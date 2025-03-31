#ifndef __BLAS_AXPY_ZIG__
#define __BLAS_AXPY_ZIG__

#include "blas_utils.cu"

// we're using double because every float can cast
// up to a double and then we can go back down.

void __axpy(
  dtype id,
  CublasWrapper w,
  const void* x,
  void* y,
  len_t n, 
  const void* alpha
) {
  const int _n = static_cast<int>(n);
  
  switch (id) {
    case SINGLE: {
      return CUBLAS_ASSERT(cublasSaxpy(
          __cast_cublas(w),
          _n, 
          static_cast<const float*>(alpha),
          static_cast<const float*>(x), 1,
          static_cast<float*>(y), 1
      ));
    }
    case DOUBLE: {
      return CUBLAS_ASSERT(cublasDaxpy(
          __cast_cublas(w),
          _n,
          static_cast<const double*>(alpha),
          static_cast<const double*>(x), 1,
          static_cast<double*>(y), 1
      ));
    }
    default:
      SYSTEM_EXIT("Unsupported data type");
  }
}

extern "C" void axpy(
  dtype id,
  CublasWrapper w,
  const void* x,
  void* y,
  len_t n, 
  const void* alpha
) {
  __axpy(id, w, x, y, n, alpha);
  CUDA_ASSERT(cudaPeekAtLastError());
}

#endif
