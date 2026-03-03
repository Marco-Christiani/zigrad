// Test: attention with masking between scale and softmax.
// AttentionPattern should NOT match (select op breaks the chain).
// SoftmaxMatmulPattern should match the shifted_scores -> softmax -> @V part.
func.func @attention_masked(%q: tensor<2x4x8x16xf32>, %k: tensor<2x4x8x16xf32>, %v: tensor<2x4x8x16xf32>, %mask: tensor<2x4x8x8xf32>) -> tensor<2x4x8x16xf32> {
  // raw_scores = Q @ K^T
  %raw_scores = "stablehlo.dot_general"(%q, %k) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [3]
    >
  } : (tensor<2x4x8x16xf32>, tensor<2x4x8x16xf32>) -> tensor<2x4x8x8xf32>

  // scaled = raw_scores * scale
  %scale = stablehlo.constant dense<2.500000e-01> : tensor<f32>
  %scale_bc = "stablehlo.broadcast_in_dim"(%scale) {
    broadcast_dimensions = array<i64>
  } : (tensor<f32>) -> tensor<2x4x8x8xf32>
  %scaled = stablehlo.multiply %raw_scores, %scale_bc : tensor<2x4x8x8xf32>

  // Masking: select(mask, scaled, -inf) — this breaks AttentionPattern
  %neg_inf_fill = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %neg_inf_bc = "stablehlo.broadcast_in_dim"(%neg_inf_fill) {
    broadcast_dimensions = array<i64>
  } : (tensor<f32>) -> tensor<2x4x8x8xf32>
  %pred = "stablehlo.compare"(%mask, %neg_inf_bc) {
    comparison_direction = #stablehlo<comparison_direction GT>
  } : (tensor<2x4x8x8xf32>, tensor<2x4x8x8xf32>) -> tensor<2x4x8x8xi1>
  %masked = "stablehlo.select"(%pred, %scaled, %neg_inf_bc) : (tensor<2x4x8x8xi1>, tensor<2x4x8x8xf32>, tensor<2x4x8x8xf32>) -> tensor<2x4x8x8xf32>

  // Softmax on masked scores (SoftmaxMatmulPattern should match from here)
  %exp = stablehlo.exponential %masked : tensor<2x4x8x8xf32>

  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %sum = "stablehlo.reduce"(%exp, %zero) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %a = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %a : tensor<f32>
  }) {dimensions = array<i64: 3>} : (tensor<2x4x8x8xf32>, tensor<f32>) -> tensor<2x4x8xf32>

  %sum_bc = "stablehlo.broadcast_in_dim"(%sum) {
    broadcast_dimensions = array<i64: 0, 1, 2>
  } : (tensor<2x4x8xf32>) -> tensor<2x4x8x8xf32>
  %probs = stablehlo.divide %exp, %sum_bc : tensor<2x4x8x8xf32>

  // probs @ V
  %result = "stablehlo.dot_general"(%probs, %v) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [2]
    >
  } : (tensor<2x4x8x8xf32>, tensor<2x4x8x16xf32>) -> tensor<2x4x8x16xf32>

  return %result : tensor<2x4x8x16xf32>
}
