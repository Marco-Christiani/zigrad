#ifndef __BLAS_AXPY_ZIG__
#define __BLAS_AXPY_ZIG__

#include "blas_utils.cu"

// we're using double because every float can cast
// up to a double and then we can go back down.

extern "C" void axpy(
  dtype id,
  void* cublas_handle,
  const void* a_data,
  void* b_data,
  len_t n, 
  len_t inca,
  len_t incb,
  double alpha
) {
  const int _n = static_cast<int>(n);
  const int _inca = static_cast<int>(inca);
  const int _incb = static_cast<int>(incb);
  
  switch (id) {
    case SINGLE: {
      const float _alpha = static_cast<float>(alpha);
      return CUBLAS_ASSERT(cublasSaxpy(
          get_handle(cublas_handle),  
          _n,
          &_alpha,
          static_cast<const float*>(a_data), _inca,
          static_cast<float*>(b_data), _incb
      ));
    }
    case DOUBLE: {
      return CUBLAS_ASSERT(cublasDaxpy(
          get_handle(cublas_handle),  
          _n,
          &alpha,
          static_cast<const double*>(a_data), _inca,
          static_cast<double*>(b_data), _incb
      ));
    }
  }
}

#endif
