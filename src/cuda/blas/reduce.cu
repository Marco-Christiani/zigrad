#ifndef __BLAS_REDUCE_ZIG__
#define __BLAS_REDUCE_ZIG__

#include "blas_utils.cu"

// we're using double because every float can cast
// up to a double and then we can go back down.

extern "C" CutensorPlanWrapper get_reduce_plan(
  CutensorWrapper wrapper,
  dtype id,
  const len_t* src_dims,
  const u8* src_syms,
  len_t src_dims_len,
  const len_t* dst_dims,
  const u8* dst_syms,
  len_t dst_dims_len,
  BINARY_OP op
) {
  CHECK_INVARIANT(0 < dst_dims_len, "Zero length dimensions passed to reduce");
  CHECK_INVARIANT(src_dims_len > dst_dims_len, "Reduction dimension out of bounds");

  auto ct = CutensorBackend::unwrap(wrapper);
  
  const auto data_type = cutensor_data_type(id);
  const auto op_type = cutensor_op_type(op);

  auto key = ct->manager.make_key<ReducePlan>(
    data_type,
    {
      __seq_hash(src_dims, src_dims_len),
      __seq_hash(src_syms, src_dims_len),
      __seq_hash(dst_dims, src_dims_len),
      __seq_hash(dst_syms, src_dims_len),
      static_cast<len_t>(op_type),
    }
  );

  if (auto entry = ct->manager.find<ReducePlan>(key); entry)
    return { .plan = entry->plan.ptr, .scratch_len = entry->scratch_len };

  BoundedArray<i64> a_dims(src_dims, src_dims_len, true);
  BoundedArray<i32> a_syms(src_syms, src_dims_len, true);
  BoundedArray<i64> b_dims(dst_dims, dst_dims_len, true);
  BoundedArray<i32> b_syms(dst_syms, dst_dims_len, true);

  cutensorTensorDescriptor_t x_desc;
  CUTENSOR_ASSERT(cutensorCreateTensorDescriptor(
              ct->handle,
              &x_desc,
              a_dims.size,
              a_dims.data,
              NULL,/*stride*/
              data_type, cutensor_alignment));
  
  cutensorTensorDescriptor_t y_desc;
  CUTENSOR_ASSERT(cutensorCreateTensorDescriptor(
              ct->handle,
              &y_desc,
              b_dims.size,
              b_dims.data,
              NULL,/*stride*/
              data_type, cutensor_alignment));
  
  cutensorOperationDescriptor_t op_desc;
  CUTENSOR_ASSERT(cutensorCreateReduction(
              ct->handle, &op_desc,
              x_desc, a_syms.data, CUTENSOR_OP_IDENTITY,
              y_desc, b_syms.data, CUTENSOR_OP_IDENTITY,
              y_desc, b_syms.data, op_type,
              cutensor_compute_type(id)));
  
  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
  
  cutensorPlanPreference_t plan_pref;
  CUTENSOR_ASSERT(cutensorCreatePlanPreference(
              ct->handle,
              &plan_pref,
              algo,
              CUTENSOR_JIT_MODE_NONE));
  
  len_t scratch_len;
  CUTENSOR_ASSERT(cutensorEstimateWorkspaceSize(
              ct->handle,
              op_desc,
              plan_pref,
              CUTENSOR_WORKSPACE_DEFAULT,
              &scratch_len));

  cutensorPlan_t plan;
  CUTENSOR_ASSERT(cutensorCreatePlan(
              ct->handle,
              &plan,
              op_desc,
              plan_pref,
              scratch_len));
  
  ct->manager.insert(
    key,
    ReducePlan{
      .plan = plan,
      .plan_pref = plan_pref,
      .x_desc = x_desc,
      .y_desc = y_desc,
      .scratch_len = scratch_len,
    }
  );
  
  return { .plan = plan, .scratch_len = scratch_len };
} 

extern "C" void reduce(
    CutensorWrapper c_wrap,
    void* plan,
    const void* x_vals,
          void* y_vals,
    void* scratch,
    len_t scratch_len,
    const void* alpha,
    const void* beta
) {
  CutensorBackend* backend = CutensorBackend::unwrap(c_wrap);

  CUTENSOR_ASSERT(cutensorReduce(
      backend->handle,
      reinterpret_cast<cutensorPlan_t>(plan), 
      alpha, x_vals,
      beta, y_vals,
      y_vals,
      scratch,
      scratch_len,
      backend->stream
  ));
}

#endif
