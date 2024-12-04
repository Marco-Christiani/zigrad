#ifndef __BLAS_DOT_ZIG__
#define __BLAS_DOT_ZIG__

#include "blas_utils.cu"

// we're using double because every float can cast
// up to a double and then we can go back down.

extern "C" void dot(
  dtype id,
  void* handle,
  const void* x,
  const void* y,
  void* z,
  len_t n
) {
  const auto _handle = static_cast<cublasHandle_t>(handle);
  const auto _n = static_cast<int>(n);

  switch (id) {
    case SINGLE: {
      const auto _x = static_cast<const float*>(x);
      const auto _y = static_cast<const float*>(y);
      const auto _z = static_cast<float*>(z);
      return CUBLAS_ASSERT(cublasSdot(_handle, _n, _x, 1, _y, 1, _z));
    }
    case DOUBLE: {
      const auto _x = static_cast<const double*>(x);
      const auto _y = static_cast<const double*>(y);
      const auto _z = static_cast<double*>(z);
      return CUBLAS_ASSERT(cublasDdot(_handle, _n, _x, 1, _y, 1, _z));
    }
  }
}

#endif
