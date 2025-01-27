#ifndef __BLAS_CONTRACT_ZIG__
#define __BLAS_CONTRACT_ZIG__

#include "blas_utils.cu"

// we're using double because every float can cast
// up to a double and then we can go back down.

// syms and dims must be same length
EXTERN_C void contraction(
  dtype id,
  CutensorWrapper wrapper,
  // x tensor //
  const void* A_vals,
  const len_t* A_dims,
  const u8* A_syms,
  len_t A_dims_len,
  // y tensor //
  const void* x_vals,
  const len_t* x_dims,
  const u8* x_syms,
  len_t x_dims_len,
  // z tensor //
        void* y_vals,
  const len_t* y_dims,
  const u8* y_syms,
  len_t y_dims_len,
  // scratch //
  len_t* scratch,
  len_t* scratch_len,
  const void* alpha,
  const void* gamma
) {
  CutensorBackend* backend = CutensorBackend::unwrap(wrapper);

  auto plan = backend->get_contraction_plan(
    id,
    A_dims, A_syms, A_dims_len,
    x_dims, x_syms, x_dims_len,
    y_dims, y_syms, y_dims_len,
    scratch, scratch_len
  );

  const auto workspace = reinterpret_cast<void*>(*scratch);

  CUTENSOR_ASSERT(cutensorContract(
      backend->handle,
      plan, 
      alpha,
      A_vals,
      x_vals,
      gamma,
      y_vals,
      y_vals,
      workspace,
      *scratch_len, 
      backend->stream
  ));
}

#endif
