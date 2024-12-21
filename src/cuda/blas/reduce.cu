#ifndef __BLAS_REDUCE_ZIG__
#define __BLAS_REDUCE_ZIG__

#include "blas_utils.cu"

// we're using double because every float can cast
// up to a double and then we can go back down.

extern "C" void reduce(
    dtype id,
    CutensorWrapper wrapper,
    const void* x_vals,
    const len_t* x_dims,
    len_t x_dims_len,
    void* y_vals,
    const len_t* y_dims,
    len_t y_dims_len,
    len_t* scratch,
    len_t* scratch_len,
    const len_t* rdx_idxs,
    len_t rdx_idxs_len,
    const void* alpha,
    const void* beta,
    BINARY_OP op
) {
  CutensorBackend* backend = CutensorBackend::unwrap(wrapper);

  auto plan = backend->get_reduce_plan(
    id, x_dims, x_dims_len, rdx_idxs, rdx_idxs_len, scratch, scratch_len, op
  );

  const auto workspace = reinterpret_cast<void*>(*scratch);

  CUTENSOR_ASSERT(cutensorReduce(
      backend->handle,
      plan, 
      alpha, x_vals,
      beta, y_vals,
      y_vals,
      workspace,
      *scratch_len,
      backend->stream
  ));
}

#endif
