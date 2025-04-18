#ifndef __BLAS_NRM2_ZIG__
#define __BLAS_NRM2_ZIG__

#include "blas_utils.cu"

void __nrm2(
  dtype id,
  CublasWrapper w,
  const void* x,
  void* y,
  len_t n
) {
  const int _n = static_cast<int>(n);
  
  switch (id) {
    case SINGLE: {
      return CUBLAS_ASSERT(cublasSnrm2(
          __cast_cublas(w),
          _n,
          static_cast<const float*>(x), 1,
          static_cast<float*>(y)
      ));
    }
    case DOUBLE: {
      return CUBLAS_ASSERT(cublasDnrm2(
          __cast_cublas(w),
          _n,
          static_cast<const double*>(x), 1,
          static_cast<double*>(y)
      ));
    }
    default:
      SYSTEM_EXIT("Unsupported data type");
  }
}


extern "C" void nrm2(
  dtype id,
  CublasWrapper w,
  const void* x,
  void* y,
  len_t n
) {
  __nrm2(id, w, x, y, n);
  CUDA_ASSERT(cudaPeekAtLastError());
}

#endif
