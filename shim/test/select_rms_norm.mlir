// Test: rsqrt-based RMSNorm chain -> zigrad.kernel_call pattern=rms_norm
// Pattern: multiply(x, broadcast(rsqrt(add(multiply(reduce_sum(multiply(x,x)), scale), eps))))
//   followed by weight multiply (which stays outside the kernel boundary).
//
// NOTE: RmsNormPattern is currently disabled in the pass (Mirage assertion).
//   To test: temporarily enable the pattern or run with a custom pass config.
//   This file documents the expected input shape for when it is re-enabled.
func.func @rms_norm(%x: tensor<2x4xf32>, %weight: tensor<2x4xf32>) -> tensor<2x4xf32> {
  // x_sq = x * x
  %x_sq = stablehlo.multiply %x, %x : tensor<2x4xf32>

  // sum = reduce_sum(x_sq, dim=1) -> tensor<2xf32>
  %init = stablehlo.constant dense<0.0> : tensor<f32>
  %sum = "stablehlo.reduce"(%x_sq, %init) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %add = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %add : tensor<f32>
  }) {dimensions = array<i64: 1>} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>

  // mean = sum * (1/4) via broadcast scalar
  %scale = stablehlo.constant dense<2.500000e-01> : tensor<f32>
  %scale_bc = "stablehlo.broadcast_in_dim"(%scale) {
    broadcast_dimensions = array<i64>
  } : (tensor<f32>) -> tensor<2xf32>
  %mean = stablehlo.multiply %sum, %scale_bc : tensor<2xf32>

  // denom = mean + eps
  %eps = stablehlo.constant dense<1.000000e-05> : tensor<f32>
  %eps_bc = "stablehlo.broadcast_in_dim"(%eps) {
    broadcast_dimensions = array<i64>
  } : (tensor<f32>) -> tensor<2xf32>
  %denom = stablehlo.add %mean, %eps_bc : tensor<2xf32>

  // inv = rsqrt(denom) -> tensor<2xf32>
  %inv = stablehlo.rsqrt %denom : tensor<2xf32>

  // broadcast inv to x shape -> tensor<2x4xf32>
  %inv_bc = "stablehlo.broadcast_in_dim"(%inv) {
    broadcast_dimensions = array<i64: 0>
  } : (tensor<2xf32>) -> tensor<2x4xf32>

  // normed = x * inv_bc  (this is the kernel boundary output)
  %normed = stablehlo.multiply %x, %inv_bc : tensor<2x4xf32>

  // weight multiply stays outside kernel: XLA fuses it trivially
  %result = stablehlo.multiply %normed, %weight : tensor<2x4xf32>

  return %result : tensor<2x4xf32>
}
