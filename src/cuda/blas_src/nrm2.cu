#ifndef __BLAS_NRM2_ZIG__
#define __BLAS_NRM2_ZIG__

#include "blas_utils.cu"

extern "C" void nrm2(
  dtype id,
  void* cublas_handle,
  void* a_data,
  len_t n,
  len_t stride
) {
  const int _n = static_cast<int>(n);
  const int _stride = static_cast<int>(stride);
  
  switch (id) {
    case SINGLE: {
      return CUBLAS_ASSERT(cublasSnrm2(
          get_handle(cublas_handle),  
          _n,
          static_cast<const float*>(a_data), _stride,
          static_cast<float*>(a_data)
      ));
    }
    case DOUBLE: {
      return CUBLAS_ASSERT(cublasDnrm2(
          get_handle(cublas_handle),  
          _n,
          static_cast<const double*>(a_data), _stride,
          static_cast<double*>(a_data)
      ));
    }
  }
}

#endif
