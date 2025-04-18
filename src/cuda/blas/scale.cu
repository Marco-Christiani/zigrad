#ifndef __BLAS_SCALE_ZIG__
#define __BLAS_SCALE_ZIG__

#include "blas_utils.cu"

extern "C" void scale(
  dtype id,
  CublasWrapper w,
  void* a_data,
  len_t n,
  double alpha
) {
  const int _n = static_cast<int>(n);
  
  switch (id) {
    case SINGLE: {
      const float _alpha = static_cast<float>(alpha);
      return CUBLAS_ASSERT(cublasSscal(
          __cast_cublas(w),
          _n,
          &_alpha,
          static_cast<float*>(a_data), 1
      ));
    }
    case DOUBLE: {
      return CUBLAS_ASSERT(cublasDscal(
          __cast_cublas(w),
          _n,
          &alpha,
          static_cast<double*>(a_data), 1
      ));
    }
  }
}

#endif

