// RUN-PIPELINE: builtin.module(func.func(zg-kernel-call-expand))
// Test: zigrad.kernel_call pattern=rms_norm_matmul -> rms_norm chain + dot_general
//   Expands: rms_norm_matmul(X, W') -> dot_general(rms_norm(X), W')
func.func @expand_rms_norm_matmul(%x: tensor<4x2048xf32>, %w: tensor<2048x256xf32>) -> tensor<4x256xf32> {
  %0 = "zigrad.kernel_call"(%x, %w) {
    api_version = 4 : i32,
    call_target_name = "zigrad.kernel.dispatch",
    has_side_effect = false,
    backend_config = {
      "zigrad.kernel_key" = "mk_0",
      "zigrad.provider" = "mirage",
      "zigrad.pattern" = "rms_norm_matmul",
      "zigrad.normalized_size" = 2048 : i32
    }
  } : (tensor<4x2048xf32>, tensor<2048x256xf32>) -> tensor<4x256xf32>
  return %0 : tensor<4x256xf32>
}
