// Test: dot_general(softmax(scores), V) -> zigrad.kernel_call pattern=softmax_matmul
// CHECK-LABEL: func.func @softmax_matmul
// CHECK: zigrad.kernel_call
// CHECK-SAME: zigrad.pattern = "softmax_matmul"
func.func @softmax_matmul(%scores: tensor<4x8xf32>, %v: tensor<8x4xf32>) -> tensor<4x4xf32> {
  %exp = stablehlo.exponential %scores : tensor<4x8xf32>

  %init = stablehlo.constant dense<0.0> : tensor<f32>
  %sum = "stablehlo.reduce"(%exp, %init) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %add = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %add : tensor<f32>
  }) {dimensions = array<i64: 1>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>

  %sum_broadcast = "stablehlo.broadcast_in_dim"(%sum) {
    broadcast_dimensions = array<i64: 0>
  } : (tensor<4xf32>) -> tensor<4x8xf32>

  %attn_probs = stablehlo.divide %exp, %sum_broadcast : tensor<4x8xf32>

  %result = "stablehlo.dot_general"(%attn_probs, %v) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >
  } : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>

  return %result : tensor<4x4xf32>
}
