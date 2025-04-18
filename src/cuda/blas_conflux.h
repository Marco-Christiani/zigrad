#ifndef __BLAS_CONFLUX_ZIG_H__
#define __BLAS_CONFLUX_ZIG_H__

#include "decls.h"

EXTERN_C void dot(
  dtype id,
  CublasWrapper w,
  const void* a_data,
  const void* b_data,
  void* result,
  len_t n
);

EXTERN_C void gemv(
    dtype id, 
    CublasWrapper w,
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
);

EXTERN_C void ger(
  dtype id,
  CublasWrapper w,
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
  CublasWrapper w,
  const void* x,
  void* y,
  len_t n
);

EXTERN_C void clip_nrm2(
  dtype id,
  CublasWrapper w,
  void* x,
  void* cur_nrm2,
  len_t n,
  double max_nrm2,
  double delta
);

EXTERN_C void max_fwd(
  dtype id,
  StreamWrapper w,
  const void* x,
  void* y,
  len_t n
);

EXTERN_C void max_bwd(
  dtype id,
  StreamWrapper w,
  const void* x_val,
  void const* y_val,
  const void* y_grd,
        void* x_grd,
  len_t n
);

EXTERN_C void reduce_sum(
  dtype id,
  CublasWrapper w,
  const void* a_data,
  void* result,
  len_t n
);

EXTERN_C void scale(
  dtype id,
  CublasWrapper w,
  void* a_data,
  len_t n,
  double alpha
);

EXTERN_C void axpy(
  dtype id,
  CublasWrapper w,
  const void* x,
  void* y,
  len_t n, 
  const void* alpha
);

EXTERN_C void addition(
  dtype id,
  StreamWrapper w,
  const void* x,
  const void* y,
  void* z,
  len_t x_len,
  len_t y_len,
  len_t z_len
);

EXTERN_C void subtraction(
  dtype id,
  StreamWrapper w,
  const void* x,
  const void* y,
  void* z,
  len_t x_len,
  len_t y_len,
  len_t z_len
);

EXTERN_C void multiplication(
  dtype id,
  StreamWrapper w,
  const void* x,
  const void* y,
  void* z,
  len_t x_len,
  len_t y_len,
  len_t z_len
);

EXTERN_C void division(
  dtype id,
  StreamWrapper w,
  const void* x,
  const void* y,
  void* z,
  len_t x_len,
  len_t y_len,
  len_t z_len
);

EXTERN_C void reduce(
    dtype id,
    CutensorWrapper wrapper,
    // x_tensor //
    const void* x_vals,
    const len_t* x_dims,
    len_t x_dims_len,
    // y_tensor //
    void* y_vals,
    // scratch //
    len_t* scratch,
    len_t* scratch_len,
    // reduce idxs //
    const len_t* rdx_idxs,
    len_t rdx_idxs_len,
    const void* alpha,
    const void* beta,
    BINARY_OP op
);

EXTERN_C void permutate(
  dtype id,
  CutensorWrapper wrapper,
  const void* x_vals, const len_t* x_dims, const unsigned char* x_syms, len_t x_dims_len,
        void* y_vals, const len_t* y_dims, const unsigned char* y_syms, len_t y_dims_len,
  len_t* scratch, len_t* scratch_len,
  const void* alpha
);

EXTERN_C void contraction(
  dtype id,
  CutensorWrapper wrapper,
  // A tensor //
  const void* A_vals,
  const len_t* A_dims,
  const unsigned char* A_syms,
  len_t A_dims_len,
  // x tensor //
  const void* x_vals,
  const len_t* x_dims,
  const unsigned char* x_syms,
  len_t x_dims_len,
  // y tensor //
         void* y_vals,
  const len_t* y_dims,
  const unsigned char* y_syms,
  len_t y_dims_len,
  // scratch //
  len_t* scratch,
  len_t* scratch_len,
  const void* alpha,
  const void* gamma
);

#endif
