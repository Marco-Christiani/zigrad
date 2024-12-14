#ifndef __BLAS_CONFLUX_ZIG_H__
#define __BLAS_CONFLUX_ZIG_H__

#include "decls.h"

EXTERN_C void dot(
  dtype id,
  void* stream,
  const void* a_data,
  const void* b_data,
  void* result,
  len_t n
);

EXTERN_C void gemv(
    dtype id, 
    void* cublas_handle,
    const void* a_data, 
    const void* b_data, 
    void* c_data, 
    len_t m,
    len_t n,
    bool trans_a,
    double alpha,
    double beta
);

EXTERN_C void gemm(
  dtype id,
  void* cublas_handle,
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
);

EXTERN_C void ger(
  dtype id,
  void* cublas_handle,
  const void* a_data,
  const void* b_data,
  void* c_data,
  len_t m,
  len_t n, 
  len_t ldc,
  double alpha
);

EXTERN_C void nrm2(
  dtype id,
  void* cublas_handle,
  const void* x,
  void* y,
  len_t n
);

EXTERN_C void max_forward(
  dtype id,
  void* cublas_handle,
  const void* x,
  void* y,
  int* idx,
  len_t n
);

EXTERN_C void max_reverse(
  dtype id,
  void* cublas_handle,
  const void* y_grd,
  void* x_grd,
  int* idx
);

EXTERN_C void reduce_sum(
  dtype id,
  void* stream,
  const void* a_data,
  void* result,
  len_t n
);

EXTERN_C void scale(
  dtype id,
  void* cublas_handle,
  void* a_data,
  len_t n,
  double alpha
);

EXTERN_C void axpy(
  dtype id,
  void* cublas_handle,
  const void* x,
  void* y,
  len_t n, 
  const void* alpha
);

EXTERN_C void addition(
  dtype id,
  void* stream,
  const void* x,
  const void* y,
  void* z,
  len_t n
);

EXTERN_C void subtraction(
  dtype id,
  void* stream,
  const void* x,
  const void* y,
  void* z,
  len_t n
);

EXTERN_C void multiplication(
  dtype id,
  void* stream,
  const void* x,
  const void* y,
  void* z,
  len_t n
);

EXTERN_C void division(
  dtype id,
  void* stream,
  const void* x,
  const void* y,
  void* z,
  len_t n
);

#endif
