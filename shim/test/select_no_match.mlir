// Test: multi-use dot result prevents pattern matching (no zigrad.kernel_call expected)
func.func @no_match(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>, %c: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  %dot = "stablehlo.dot_general"(%a, %b) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >
  } : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
  %add = stablehlo.add %dot, %c : tensor<4x4xf32>
  return %dot, %add : tensor<4x4xf32>, tensor<4x4xf32>
}
