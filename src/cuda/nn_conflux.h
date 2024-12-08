#ifndef __NN_CONFLUX_ZIG_H__
#define __NN_CONFLUX_ZIG_H__

#include "decls.h"

EXTERN_C void relu_forward(
  dtype id,
  void* stream,
  const void* x,
  void* y,
  len_t n
);

EXTERN_C void relu_reverse(
  dtype id,
  void* stream,
  const void* x,
  const void* y_grd,
  void* x_grd,
  len_t n
);

EXTERN_C void smax_vec_forward(
  dtype id,
  void* cudnn_handle,
  const void* x,
  void* y,
  len_t n,
  smaxtype smaxop
);

EXTERN_C void smax_vec_reverse(
  dtype id,
  void* cudnn_handle,
  const void* y_val,
  const void* y_grd,
  void* x_grd,
  len_t n
);

EXTERN_C void smax_2D_row_forward(
  dtype id,
  void* cudnn_handle,
  const void* x,
  void* y,
  len_t m,
  len_t n
);

EXTERN_C void softmax_2D_row_reverse(
  dtype id,
  void* cudnn_handle,
  const void* y_val,
  const void* y_grd,
  void* x_grd,
  len_t m,
  len_t n
);

EXTERN_C void clamp(
  dtype id,
  void* stream,
  const void* x,
  void* y,
  len_t n,
  double lower,
  double upper
);

EXTERN_C void nll_loss_1D_index_forward(
  dtype id,
  void* cudnn_handle,
  void* src,
  len_t trg,
  void* dst,
  len_t n,
  bool inplace_smax,
  reduxtype reduxop
);

EXTERN_C void nll_loss_1D_index_reverse(
  dtype id,
  void* cudnn_handle,
  const void* x_val,
  void *x_grd,
  len_t trg,
  len_t n,
  reduxtype reduxop
);

EXTERN_C void clip_norm(
  dtype id,
  void* cublas_handle,
  void* x,
  len_t n,
  void* cur_nrm2,
  double max_nrm2,
  double delta
);

#endif
