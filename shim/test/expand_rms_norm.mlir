// RUN-PIPELINE: builtin.module(func.func(zg-kernel-call-expand))
// Test: zigrad.kernel_call pattern=rms_norm -> stablehlo ops
//   Expands: rms_norm(x) -> multiply(x, broadcast(rsqrt(add(multiply(reduce_sum(multiply(x,x)), scale), eps))))
func.func @expand_rms_norm(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = "zigrad.kernel_call"(%x) {
    api_version = 4 : i32,
    call_target_name = "zigrad.kernel.dispatch",
    has_side_effect = false,
    backend_config = {
      "zigrad.kernel_key" = "mk_0",
      "zigrad.provider" = "mirage",
      "zigrad.pattern" = "rms_norm",
      "zigrad.normalized_size" = 4 : i32
    }
  } : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
