#ifndef __BLAS_GER_ZIG__
#define __BLAS_GER_ZIG__

#include "blas_utils.cu"

// we're using double because every float can cast
// up to a double and then we can go back down.

extern "C" void ger(
  dtype id,
  void* cublas_handle,
  const void* a_data,
  const void* b_data,
  void* c_data,
  len_t m,
  len_t n, 
  len_t ldc,
  double alpha
) {
  const int _m = static_cast<int>(m);
  const int _n = static_cast<int>(n);
  const int _ldc = static_cast<int>(ldc);
  
  switch (id) {
    case SINGLE: {
      const float _alpha = static_cast<float>(alpha);
      return CUBLAS_ASSERT(cublasSger(
          get_handle(cublas_handle),  
          _m, _n,
          &_alpha,
          static_cast<const float*>(a_data), 1,
          static_cast<const float*>(b_data), 1,
          static_cast<float*>(c_data), _ldc
      ));
    }
    case DOUBLE: {
      return CUBLAS_ASSERT(cublasDger(
          get_handle(cublas_handle),  
          _m, _n,
          &alpha,
          static_cast<const double*>(a_data), 1,
          static_cast<const double*>(b_data), 1,
          static_cast<double*>(c_data), _ldc
      ));
    }
  }
}

#endif
