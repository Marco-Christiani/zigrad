// RUN-PIPELINE: builtin.module(func.func(zg-kernel-call-expand))
// Test: zigrad.kernel_call pattern=dot_add -> stablehlo.dot_general + stablehlo.add
// CHECK-LABEL: func.func @expand_dot_add
// CHECK: stablehlo.dot_general
// CHECK: stablehlo.add
// CHECK-NOT: zigrad.kernel_call
func.func @expand_dot_add(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>, %c: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = "zigrad.kernel_call"(%a, %b, %c) {
    api_version = 4 : i32,
    call_target_name = "zigrad.kernel.dispatch",
    has_side_effect = false,
    backend_config = {
      "zigrad.kernel_key" = "mk_0",
      "zigrad.provider" = "mirage",
      "zigrad.pattern" = "dot_add"
    }
  } : (tensor<4x8xf32>, tensor<8x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
