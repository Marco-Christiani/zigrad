// RUN-PIPELINE: builtin.module(func.func(zg-kernel-legalize))
// Test: zigrad.kernel_call -> stablehlo.custom_call
func.func @legalize(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>, %c: tensor<4x4xf32>) -> tensor<4x4xf32> {
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
