#ifndef __BLAS_GEMV_ZIG__
#define __BLAS_GEMV_ZIG__

#include "blas_utils.cu"

void __gemv(
    dtype id, 
    CublasWrapper w,
    const void* A, 
    const void* x, 
    void* y, 
    len_t m,
    len_t n,
    bool trans_a,
    double alpha,
    double beta
) {
  const int _m = static_cast<int>(m);
  const int _n = static_cast<int>(n);
  const auto _trans_a = (trans_a) ? CUBLAS_OP_N : CUBLAS_OP_T;
  
  switch (id) {
    case SINGLE: {
      const float _alpha = static_cast<float>(alpha);
      const float _beta = static_cast<float>(beta);
      return CUBLAS_ASSERT(cublasSgemv(
        __cast_cublas(w),
        _trans_a,
        _n, _m,
        &_alpha,
        static_cast<const float*>(A), _n,
        static_cast<const float*>(x), 1,
        &_beta,
        static_cast<float*>(y), 1
      ));
    }
    case DOUBLE: {
      return CUBLAS_ASSERT(cublasDgemv(
        __cast_cublas(w),
        _trans_a,
        _n, _m,
        &alpha,
        static_cast<const double*>(x), 1,
        static_cast<const double*>(A), _n,
        &beta,
        static_cast<double*>(y), 1
      ));
    }
    default:
      SYSTEM_EXIT("Unsupported data type");
  }
}

extern "C" void gemv(
    dtype id, 
    CublasWrapper w,
    const void* A, 
    const void* x, 
    void* y, 
    len_t m,
    len_t n,
    bool trans_a,
    double alpha,
    double beta
) {
  __gemv(id, w, A, x, y, m, n, trans_a, alpha, beta);
  CUDA_ASSERT(cudaPeekAtLastError());
}
#endif
