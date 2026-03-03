// Test: full unmasked attention (Q@K -> scale -> stable softmax -> @V) -> zigrad.kernel_call pattern=attention
// Q=[2,4,8,16], K=[2,4,8,16], V=[2,4,8,16] — batched attention with 2 batch, 4 heads, 8 seq, 16 head_dim
func.func @attention(%q: tensor<2x4x8x16xf32>, %k: tensor<2x4x8x16xf32>, %v: tensor<2x4x8x16xf32>) -> tensor<2x4x8x16xf32> {
  // raw_scores = Q @ K^T: [2,4,8,16] x [2,4,8,16] -> [2,4,8,8]
  %raw_scores = "stablehlo.dot_general"(%q, %k) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [3]
    >
  } : (tensor<2x4x8x16xf32>, tensor<2x4x8x16xf32>) -> tensor<2x4x8x8xf32>

  // scaled = raw_scores * (1/sqrt(16)) = raw_scores * 0.25
  %scale = stablehlo.constant dense<2.500000e-01> : tensor<f32>
  %scale_bc = "stablehlo.broadcast_in_dim"(%scale) {
    broadcast_dimensions = array<i64>
  } : (tensor<f32>) -> tensor<2x4x8x8xf32>
  %scaled = stablehlo.multiply %raw_scores, %scale_bc : tensor<2x4x8x8xf32>

  // max = reduce_max(scaled, dim=3)
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %max = "stablehlo.reduce"(%scaled, %neg_inf) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %m = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %m : tensor<f32>
  }) {dimensions = array<i64: 3>} : (tensor<2x4x8x8xf32>, tensor<f32>) -> tensor<2x4x8xf32>

  // shifted = scaled - broadcast(max)
  %max_bc = "stablehlo.broadcast_in_dim"(%max) {
    broadcast_dimensions = array<i64: 0, 1, 2>
  } : (tensor<2x4x8xf32>) -> tensor<2x4x8x8xf32>
  %shifted = stablehlo.subtract %scaled, %max_bc : tensor<2x4x8x8xf32>

  // exp = exp(shifted)
  %exp = stablehlo.exponential %shifted : tensor<2x4x8x8xf32>

  // sum = reduce_sum(exp, dim=3)
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %sum = "stablehlo.reduce"(%exp, %zero) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %a = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %a : tensor<f32>
  }) {dimensions = array<i64: 3>} : (tensor<2x4x8x8xf32>, tensor<f32>) -> tensor<2x4x8xf32>

  // probs = exp / broadcast(sum)
  %sum_bc = "stablehlo.broadcast_in_dim"(%sum) {
    broadcast_dimensions = array<i64: 0, 1, 2>
  } : (tensor<2x4x8xf32>) -> tensor<2x4x8x8xf32>
  %probs = stablehlo.divide %exp, %sum_bc : tensor<2x4x8x8xf32>

  // result = probs @ V: [2,4,8,8] x [2,4,8,16] -> [2,4,8,16]
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
