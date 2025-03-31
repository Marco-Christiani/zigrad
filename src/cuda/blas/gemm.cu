#ifndef __BLAS_GEMM_ZIG__
#define __BLAS_GEMM_ZIG__

#include "blas_utils.cu"

// we're using double because every float can cast
// up to a double and then we can go back down.

void __gemm(
  dtype id,
  CublasWrapper w,
  const void* a_data,
  const void* b_data,
  void* c_data,
  len_t m, 
  len_t n, 
  len_t k,
  bool trans_a,
  bool trans_b, 
  len_t lda,
  len_t ldb,
  len_t ldc,
  double alpha,
  double beta
) {
  const int _m = static_cast<int>(m);
  const int _n = static_cast<int>(n);
  const int _k = static_cast<int>(k);
  
  switch (id) {
    case SINGLE: {
      const float _alpha = static_cast<float>(alpha);
      const float _beta = static_cast<float>(beta);
      return CUBLAS_ASSERT(cublasSgemm(
          __cast_cublas(w),
          (trans_b) ? CUBLAS_OP_T : CUBLAS_OP_N,
          (trans_a) ? CUBLAS_OP_T : CUBLAS_OP_N,
          _k, _m, _n,
          &_alpha, 
          static_cast<const float*>(b_data), ldb,
          static_cast<const float*>(a_data), lda,
          &_beta, 
          static_cast<float*>(c_data), ldc
      ));
    }
    case DOUBLE: {
      return CUBLAS_ASSERT(cublasDgemm(
          __cast_cublas(w),
          (trans_b) ? CUBLAS_OP_T : CUBLAS_OP_N,
          (trans_a) ? CUBLAS_OP_T : CUBLAS_OP_N,
          _k, _m, _n,
          &alpha, 
          static_cast<const double*>(b_data), ldb,
          static_cast<const double*>(a_data), lda,
          &beta, 
          static_cast<double*>(c_data), ldc
      ));
    }
    default:
      SYSTEM_EXIT("Unsupported data type");
  }
}

extern "C" void gemm(
  dtype id,
  CublasWrapper w,
  const void* a_data,
  const void* b_data,
  void* c_data,
  len_t m, 
  len_t n, 
  len_t k,
  bool trans_a,
  bool trans_b, 
  len_t lda,
  len_t ldb,
  len_t ldc,
  double alpha,
  double beta
) {
  __gemm(id, w, a_data, b_data, c_data, m, n, k, trans_a, trans_b, lda, ldb, ldc, alpha, beta);
  CUDA_ASSERT(cudaPeekAtLastError());
}

#endif
