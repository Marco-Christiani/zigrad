// Test: mul(add(dot_general(A,B), C), C) -> zigrad.kernel_call pattern=dot_add_mul
func.func @dot_add_mul(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>, %c: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = "stablehlo.dot_general"(%a, %b) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >
  } : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
  %1 = stablehlo.add %0, %c : tensor<4x4xf32>
  %2 = stablehlo.multiply %1, %c : tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}
