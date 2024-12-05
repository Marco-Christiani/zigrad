#ifndef __BLAS_SUM_ZIG__
#define __BLAS_SUM_ZIG__

#include "blas_utils.cu"

EXTERN_C void reduce_sum(
  dtype id,
  void* handle,
  const void* x,
  void* y,
  len_t n
) {
  const auto _handle = static_cast<cublasHandle_t>(handle);
  const auto _n = static_cast<int>(n);

  switch (id) {
    case SINGLE: {
      const auto _x = static_cast<const float*>(x);
      const auto _y = static_cast<float*>(y);
      return CUBLAS_ASSERT(cublasSasum(_handle, _n, _x, 1, _y));
    }
    case DOUBLE: {
      const auto _x = static_cast<const double*>(x);
      const auto _y = static_cast<double*>(y);
      return CUBLAS_ASSERT(cublasDasum(_handle, _n, _x, 1, _y));
    }
  }
}

#endif
