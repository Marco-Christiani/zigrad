#ifndef __BLAS_CONTRACT_ZIG__
#define __BLAS_CONTRACT_ZIG__

#include "blas_utils.cu"

// we're using double because every float can cast
// up to a double and then we can go back down.
  // syms and dims must be same length
extern "C" CutensorPlanWrapper get_contraction_plan(
  CutensorWrapper wrapper,
  dtype id,
  // x tensor //
  const len_t* x_dims,
  const u8* x_syms,
  len_t x_dims_len,
  // y tensor //
  const len_t* y_dims,
  const u8* y_syms,
  len_t y_dims_len,
  // z tensor //
  const len_t* z_dims,
  const u8* z_syms,
  len_t z_dims_len
) {
  CHECK_INVARIANT(0 < x_dims_len, "Zero length dimensions passed to contraction.");
  CHECK_INVARIANT(0 < y_dims_len, "Zero length dimensions passed to contraction.");

  auto ct = CutensorBackend::unwrap(wrapper);

  const auto data_type = cutensor_data_type(id);

  auto key = ct->manager.make_key<PermutatePlan>(
    data_type,
    {
      __seq_hash(x_dims, x_dims_len),
      __seq_hash(x_syms, x_dims_len),
      __seq_hash(y_dims, y_dims_len),
      __seq_hash(y_syms, y_dims_len),
      __seq_hash(z_dims, z_dims_len),
      __seq_hash(z_syms, z_dims_len),
    }
  );

  if (auto entry = ct->manager.find<ContractionPlan>(key); entry)
    return { .plan = entry->plan.ptr, .scratch_len = entry->scratch_len };

  BoundedArray<i64> a_dims(x_dims, x_dims_len, true);
  BoundedArray<i32> a_syms(x_syms, x_dims_len, true);

  BoundedArray<i64> b_dims(y_dims, y_dims_len, true);
  BoundedArray<i32> b_syms(y_syms, y_dims_len, true);

  BoundedArray<i64> c_dims(z_dims, z_dims_len, true);
  BoundedArray<i32> c_syms(z_syms, z_dims_len, true);

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

  cutensorTensorDescriptor_t z_desc;
  CUTENSOR_ASSERT(cutensorCreateTensorDescriptor(
              ct->handle,
              &z_desc,
              c_dims.size,
              c_dims.data,
              NULL,/*stride*/
              data_type, cutensor_alignment));

  cutensorOperationDescriptor_t op_desc;
  CUTENSOR_ASSERT(cutensorCreateContraction(
              ct->handle, &op_desc,
              x_desc, a_syms.data, CUTENSOR_OP_IDENTITY,
              y_desc, b_syms.data, CUTENSOR_OP_IDENTITY,
              z_desc, c_syms.data, CUTENSOR_OP_IDENTITY,
              z_desc, c_syms.data,
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
    ContractionPlan{
      .plan = plan,
      .plan_pref = plan_pref,
      .x_desc = x_desc,
      .y_desc = y_desc,
      .z_desc = z_desc,
      .scratch_len = scratch_len,
    }
  );

  return { .plan = plan, .scratch_len = scratch_len };
}


// syms and dims must be same length
extern "C" void contraction(
  CutensorWrapper wrapper,
  void* plan,
  // x tensor //
  const void* A_vals,
  // y tensor //
  const void* x_vals,
  // z tensor //
        void* y_vals,
  // scratch //
  void* scratch,
  len_t scratch_len,
  const void* alpha,
  const void* gamma
) {
  CutensorBackend* backend = CutensorBackend::unwrap(wrapper);

  CUTENSOR_ASSERT(cutensorContract(
      backend->handle,
      reinterpret_cast<cutensorPlan_t>(plan),
      alpha,
      A_vals,
      x_vals,
      gamma,
      y_vals,
      y_vals,
      scratch,
      scratch_len, 
      backend->stream
  ));
}

#endif
