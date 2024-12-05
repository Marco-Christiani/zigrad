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
  len_t n
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

#endif
