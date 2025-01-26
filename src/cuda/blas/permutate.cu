#ifndef __BLAS_PERMUTATE_ZIG__
#define __BLAS_PERMUTATE_ZIG__

#include "blas_utils.cu"

// we're using double because every float can cast
// up to a double and then we can go back down.

extern "C" void permutate(
    dtype id,
    CutensorWrapper wrapper,
    const void* x_vals, const len_t* x_dims, const u8* x_syms, len_t x_dims_len,
          void* y_vals, const len_t* y_dims, const u8* y_syms, len_t y_dims_len,
    len_t* scratch, len_t* scratch_len,
    const void* alpha
) {
  CutensorBackend* backend = CutensorBackend::unwrap(wrapper);

  // TODO: Do we need the scratch memory at all? We don't use it.
  auto plan = backend->get_permutate_plan(
    id,
    x_dims, x_syms, x_dims_len,
    y_dims, y_syms, y_dims_len,
    scratch, scratch_len
  );

  CUTENSOR_ASSERT(cutensorPermute(
      backend->handle, plan, alpha, x_vals, y_vals, backend->stream
  ));
}

extern "C" void permutate_backwards(
    dtype id,
    CutensorWrapper wrapper,
          void* x_grad, const len_t* x_dims, const u8* x_syms, len_t x_dims_len,
    const void* y_grad, const len_t* y_dims, const u8* y_syms, len_t y_dims_len,
    len_t* scratch, len_t* scratch_len,
    const void* alpha, const void* beta
) {
  CutensorBackend* backend = CutensorBackend::unwrap(wrapper);

  if (x_dims_len == y_dims_len) {
      // TODO: Do we need the scratch memory at all? We don't use it.
      auto plan = backend->get_binary_plan(
        id,
        x_dims, x_syms, x_dims_len,
        y_dims, y_syms, y_dims_len,
        BINARY_OP::ADD
      );

    CUTENSOR_ASSERT(cutensorElementwiseBinaryExecute(
        backend->handle, plan, alpha, x_grad, beta, y_grad, x_grad, backend->stream
    ));

  } else {
    auto plan = backend->get_reduce_plan(
      id,
      y_dims, y_syms, y_dims_len,
      x_dims, x_syms, x_dims_len,
      scratch, scratch_len,
      BINARY_OP::ADD
    );

    CUTENSOR_ASSERT(cutensorReduce(
        backend->handle, plan, alpha, y_grad, beta, x_grad, x_grad, scratch, *scratch_len, backend->stream
    ));
  }


  //CUTENSOR_ASSERT(cutensorPermute(
  //    backend->handle, plan, alpha, x_vals, y_vals, backend->stream
  //));
}

#endif
