#ifndef __NN_CONFLUX_ZIG_H__
#define __NN_CONFLUX_ZIG_H__

#include "decls.h"

EXTERN_C void relu_fwd(
  dtype id,
  StreamWrapper w,
  const void* x,
  void* y,
  len_t n
);

EXTERN_C void relu_bwd(
  dtype id,
  StreamWrapper w,
  const void* x,
  const void* y_grd,
  void* x_grd,
  len_t n
);

EXTERN_C void smax_vec_fwd(
  dtype id,
  CudnnWrapper w,
  const void* x,
  void* y,
  len_t n,
  smaxtype smaxop
);

EXTERN_C void smax_vec_bwd(
  dtype id,
  CudnnWrapper w,
  const void* y_val,
  const void* y_grd,
  void* x_grd,
  len_t n
);

EXTERN_C void smax_2D_row_fwd(
  dtype id,
  CudnnWrapper w,
  const void* x,
  void* y,
  len_t m,
  len_t n
);

EXTERN_C void smax_2D_row_bwd(
  dtype id,
  CudnnWrapper w,
  const void* y_val,
  const void* y_grd,
  void* x_grd,
  len_t m,
  len_t n
);

EXTERN_C void clamp(
  dtype id,
  StreamWrapper w,
  const void* x,
  void* y,
  len_t n,
  double lower,
  double upper
);

EXTERN_C void nll_loss_1D_index_fwd(
  dtype id,
  CudnnWrapper w,
  void* src,
  len_t trg,
  void* dst,
  len_t n,
  bool inplace_smax,
  reduxtype reduxop
);

EXTERN_C void nll_loss_1D_index_bwd(
  dtype id,
  CudnnWrapper w,
  const void* x_val,
  void *x_grd,
  len_t trg,
  len_t n,
  reduxtype reduxop
);

EXTERN_C void nll_loss_1D_encode_fwd(
  dtype id,
  CudnnWrapper w,
  void* src,
  const void* trg,
  void* dst,
  len_t n,
  bool inplace_smax,
  reduxtype reduxop
);

EXTERN_C void nll_loss_1D_encode_bwd(
  dtype id,
  CudnnWrapper w,
  const void* x_val,
  const void* y_val,
  void *x_grd,
  len_t n,
  reduxtype reduxop
);

EXTERN_C void pow_exp(
  dtype id,
  StreamWrapper w,
  const void* x,
  void* y,
  len_t n
);

#endif
