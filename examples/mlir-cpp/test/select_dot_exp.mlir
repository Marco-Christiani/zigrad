// Test: exp(dot_general(A,B)) -> zigrad.kernel_call pattern=dot_exp
// CHECK-LABEL: func.func @dot_exp
// CHECK: zigrad.kernel_call
// CHECK-SAME: zigrad.pattern = "dot_exp"
func.func @dot_exp(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
  %0 = "stablehlo.dot_general"(%a, %b) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >
  } : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
  %1 = stablehlo.exponential %0 : tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}
