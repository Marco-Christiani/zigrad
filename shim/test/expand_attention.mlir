// RUN-PIPELINE: builtin.module(func.func(zg-kernel-call-expand))
// Test: zigrad.kernel_call pattern=attention -> full StableHLO attention chain
//   Expands: attention(Q, K, V) -> dot_general(softmax(scale * dot_general(Q, K)), V)
func.func @expand_attention(%q: tensor<2x4x8x16xf32>, %k: tensor<2x4x8x16xf32>, %v: tensor<2x4x8x16xf32>) -> tensor<2x4x8x16xf32> {
  %0 = "zigrad.kernel_call"(%q, %k, %v) {
    api_version = 4 : i32,
    call_target_name = "zigrad.kernel.dispatch",
    has_side_effect = false,
    backend_config = {
      "zigrad.kernel_key" = "mk_0",
      "zigrad.provider" = "mirage",
      "zigrad.pattern" = "attention",
      "zigrad.scale" = 2.500000e-01 : f32,
      "zigrad.reduction_dim" = 3 : i32,
      "zigrad.reduction_factor" = 8 : i32,
      "zigrad.score_dot_dims" = #stablehlo.dot<
        lhs_batching_dimensions = [0, 1],
        rhs_batching_dimensions = [0, 1],
        lhs_contracting_dimensions = [3],
        rhs_contracting_dimensions = [3]
      >,
      "zigrad.value_dot_dims" = #stablehlo.dot<
        lhs_batching_dimensions = [0, 1],
        rhs_batching_dimensions = [0, 1],
        lhs_contracting_dimensions = [3],
        rhs_contracting_dimensions = [2]
      >,
      "zigrad.scores_shape" = array<i64: 2, 4, 8, 8>
    }
  } : (tensor<2x4x8x16xf32>, tensor<2x4x8x16xf32>, tensor<2x4x8x16xf32>) -> tensor<2x4x8x16xf32>
  return %0 : tensor<2x4x8x16xf32>
}
