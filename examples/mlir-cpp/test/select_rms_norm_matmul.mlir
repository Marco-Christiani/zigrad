// RUN-PIPELINE: func.func(zg-mirage-kernel-select)
// Test: rms_norm(X)*gamma -> convert -> reshape -> dot_general(_, W)
//   => kernel_call{rms_norm_matmul}(X_bf16_2D, W') with pre-multiplied W'.
//
// The bf16->f32 converts for precision, rms_norm chain runs in f32,
// gamma multiply, convert back to bf16, reshape to 2D, then matmul.
// CHECK-LABEL: func.func @rms_norm_matmul
// CHECK: zigrad.kernel_call
// CHECK-SAME: zigrad.pattern = "rms_norm_matmul"
func.func @rms_norm_matmul(
    %x_bf16: tensor<4x2048xbf16>,
    %gamma_bf16: tensor<2048xbf16>,
    %w: tensor<2048x256xbf16>
) -> tensor<4x256xbf16> {
  // bf16 -> f32 for rms_norm precision
  %x_f32 = stablehlo.convert %x_bf16 : (tensor<4x2048xbf16>) -> tensor<4x2048xf32>
  %gamma_f32 = stablehlo.convert %gamma_bf16 : (tensor<2048xbf16>) -> tensor<2048xf32>

  // x_sq = x * x
  %x_sq = stablehlo.multiply %x_f32, %x_f32 : tensor<4x2048xf32>

  // sum = reduce_sum(x_sq, dim=1)
  %init = stablehlo.constant dense<0.0> : tensor<f32>
  %sum = "stablehlo.reduce"(%x_sq, %init) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %add = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %add : tensor<f32>
  }) {dimensions = array<i64: 1>} : (tensor<4x2048xf32>, tensor<f32>) -> tensor<4xf32>

  // mean = sum * (1/2048)
  %scale = stablehlo.constant dense<4.882812e-04> : tensor<f32>
  %scale_bc = "stablehlo.broadcast_in_dim"(%scale) {
    broadcast_dimensions = array<i64>
  } : (tensor<f32>) -> tensor<4xf32>
  %mean = stablehlo.multiply %sum, %scale_bc : tensor<4xf32>

  // denom = mean + eps
  %eps = stablehlo.constant dense<1.000000e-05> : tensor<f32>
  %eps_bc = "stablehlo.broadcast_in_dim"(%eps) {
    broadcast_dimensions = array<i64>
  } : (tensor<f32>) -> tensor<4xf32>
  %denom = stablehlo.add %mean, %eps_bc : tensor<4xf32>

  // inv = rsqrt(denom)
  %inv = stablehlo.rsqrt %denom : tensor<4xf32>

  // broadcast inv to x shape
  %inv_bc = "stablehlo.broadcast_in_dim"(%inv) {
    broadcast_dimensions = array<i64: 0>
  } : (tensor<4xf32>) -> tensor<4x2048xf32>

  // normed = x * inv
  %normed = stablehlo.multiply %x_f32, %inv_bc : tensor<4x2048xf32>

  // gamma multiply
  %gamma_bc = "stablehlo.broadcast_in_dim"(%gamma_f32) {
    broadcast_dimensions = array<i64: 1>
  } : (tensor<2048xf32>) -> tensor<4x2048xf32>
  %gamma_mul = stablehlo.multiply %normed, %gamma_bc : tensor<4x2048xf32>

  // convert back to bf16
  %converted = stablehlo.convert %gamma_mul : (tensor<4x2048xf32>) -> tensor<4x2048xbf16>

  // matmul
  %result = "stablehlo.dot_general"(%converted, %w) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >
  } : (tensor<4x2048xbf16>, tensor<2048x256xbf16>) -> tensor<4x256xbf16>

  return %result : tensor<4x256xbf16>
}
