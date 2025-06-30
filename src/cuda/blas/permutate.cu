#ifndef __BLAS_PERMUTATE_ZIG__
#define __BLAS_PERMUTATE_ZIG__

#include "blas_utils.cu"

// syms and dims must be same length
extern "C" CutensorPlanWrapper get_permutate_plan(
    CutensorWrapper wrapper,
    dtype id,
    const len_t* src_dims,
    const u8* src_syms,
    len_t src_dims_len,
    const len_t* dst_dims,
    const u8* dst_syms,
    len_t dst_dims_len
) {
  CHECK_INVARIANT(0 < src_dims_len, "Zero length dimensions passed to permutate");
  CHECK_INVARIANT(src_dims_len <= dst_dims_len, "Source dimensions length greater than destination");

  auto ct = CutensorBackend::unwrap(wrapper);

  const auto data_type = cutensor_data_type(id);

  auto key = ct->manager.make_key<PermutatePlan>(
    data_type,
    {
      __seq_hash(src_dims, src_dims_len),
      __seq_hash(src_syms, src_dims_len),
      __seq_hash(dst_dims, src_dims_len),
      __seq_hash(dst_syms, src_dims_len),
    }
  );

  if (auto entry = ct->manager.find<PermutatePlan>(key); entry)
    return { .plan = entry->plan.ptr, .scratch_len = entry->scratch_len };

  BoundedArray<i64> a_dims(src_dims, src_dims_len, true);
  BoundedArray<i32> a_syms(src_syms, src_dims_len, true);
  BoundedArray<i64> b_dims(dst_dims, src_dims_len, true);
  BoundedArray<i32> b_syms(dst_syms, src_dims_len, true);

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
  CUTENSOR_ASSERT(cutensorCreatePermutation(
              ct->handle, &op_desc,
              x_desc, a_syms.data, CUTENSOR_OP_IDENTITY,
              y_desc, b_syms.data, 
              cutensor_compute_type(id)));

  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

  cutensorPlanPreference_t plan_pref;
  CUTENSOR_ASSERT(cutensorCreatePlanPreference(
              ct->handle,
              &plan_pref,
              algo,
              CUTENSOR_JIT_MODE_NONE));

  len_t scratch_len = 0;
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
    PermutatePlan{
      .plan = plan,
      .plan_pref = plan_pref,
      .x_desc = x_desc,
      .y_desc = y_desc,
      .scratch_len = scratch_len,
    }
  );

  return { .plan = plan, .scratch_len = scratch_len };
}

extern "C" void permutate(
    CutensorWrapper wrapper,
    void* plan,
    const void* x_vals,
          void* y_vals,
    void* scratch,
    len_t scratch_len,
    const void* alpha
) {
  CutensorBackend* backend = CutensorBackend::unwrap(wrapper);

  CUTENSOR_ASSERT(cutensorPermute(
      backend->handle,
      reinterpret_cast<cutensorPlan_t>(plan),
      alpha,
      x_vals,
      y_vals,
      backend->stream
  ));
}

#endif
