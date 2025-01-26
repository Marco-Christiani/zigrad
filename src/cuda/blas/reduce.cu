#ifndef __BLAS_REDUCE_ZIG__
#define __BLAS_REDUCE_ZIG__

#include "blas_utils.cu"

// we're using double because every float can cast
// up to a double and then we can go back down.

extern "C" void reduce(
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

template<class T>
void __reduce_backwards(
    dtype id,
    CutensorBackend* backend,
    // x_tensor //
    const void* x_vals,
          void* x_grad,
    const len_t* x_dims,
    len_t x_dims_len,
    // y_tensor //
    const void* y_vals,
    const void* y_grad,
    // scratch //
    len_t* scratch,
    len_t* scratch_len,
    // reduce idxs //
    const len_t* rdx_idxs,
    len_t rdx_idxs_len,
    BINARY_OP op
) {
  if (op == BINARY_OP::ADD) {
    auto binary_plan = backend->get_reduce_bwds_binary(
      id, x_dims, x_dims_len, rdx_idxs, rdx_idxs_len, BINARY_OP::ADD
    );

    const T alpha = 1.0;
    const T gamma = 1.0;
    CUTENSOR_ASSERT(cutensorElementwiseBinaryExecute(
        backend->handle,
        binary_plan, 
        &alpha, y_grad,
        &gamma, x_grad,
        x_grad,
        backend->stream
    ));

  } else {
    ENSURE_SCRATCH(id, backend->stream, scratch, scratch_len, __product(x_dims, x_dims_len));
    auto s_vals = reinterpret_cast<T*>(*scratch);
    auto s_size = *scratch_len;
    
    auto add_plan = backend->get_reduce_bwds_binary(
      id, x_dims, x_dims_len, rdx_idxs, rdx_idxs_len, BINARY_OP::ADD
    );
    auto mul_plan = backend->get_reduce_bwds_binary(
      id, x_dims, x_dims_len, rdx_idxs, rdx_idxs_len, BINARY_OP::MUL
    );

    T alpha = 1.0;
    T gamma = 0.0;
    CUTENSOR_ASSERT(cutensorElementwiseBinaryExecute(
        backend->handle,
        add_plan, 
        &alpha,
        y_vals,
        &gamma,
        scratch,
        scratch,
        backend->stream
    ));

    thrust::transform(
      thrust::cuda::par.on(backend->stream), 
      s_vals,  
      s_vals + s_size,
      static_cast<const T*>(x_vals),
      s_vals,
      [=] __device__ (T x, T y) -> T { return (x == y) ? T(1) : T(0); }
    );

    // TODO: remove both of these and replace with trinary

    gamma = 1;
    CUTENSOR_ASSERT(cutensorElementwiseBinaryExecute(
        backend->handle,
        mul_plan, 
        &alpha,
        y_grad,
        &gamma,
        scratch,
        scratch,
        backend->stream
    ));

    CUTENSOR_ASSERT(cutensorElementwiseBinaryExecute(
        backend->handle,
        add_plan, 
        &alpha,
        scratch,
        &gamma,
        x_grad,
        x_grad,
        backend->stream
    ));
  }
}

extern "C" void reduce_backwards(
    dtype id,
    CutensorWrapper wrapper,
    // x_tensor //
    const void* x_vals,
          void* x_grad,
    const len_t* x_dims,
    len_t x_dims_len,
    // y_tensor //
    const void* y_vals,
    const void* y_grad,
    // scratch //
    len_t* scratch,
    len_t* scratch_len,
    // reduce idxs //
    const len_t* rdx_idxs,
    len_t rdx_idxs_len,
    BINARY_OP op
) {
    CutensorBackend* backend = CutensorBackend::unwrap(wrapper);

    if (op == BINARY_OP::MUL) {
      SYSTEM_EXIT("Unimplemented binary op in reduce backwards");
    }

    switch (id) {
      case SINGLE: return __reduce_backwards<f32>(
          id, backend, x_vals, x_grad, x_dims, x_dims_len, y_vals, y_grad, scratch, scratch_len, rdx_idxs, rdx_idxs_len, op
      );
      case DOUBLE: return __reduce_backwards<f64>(
          id, backend, x_vals, x_grad, x_dims, x_dims_len, y_vals, y_grad, scratch, scratch_len, rdx_idxs, rdx_idxs_len, op
      );
      default: SYSTEM_EXIT("Unsupported datatype");
    }
}

#endif
