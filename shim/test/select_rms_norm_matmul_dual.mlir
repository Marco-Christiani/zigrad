// RUN-PIPELINE: func.func(zg-mirage-kernel-select)
// Test: Same rms_norm chain feeding TWO dot_general ops (gate + up projections).
//   => Two separate kernel_call{rms_norm_matmul} ops; shared chain erased.
func.func @rms_norm_matmul_dual(
    %x_bf16: tensor<4x2048xbf16>,
    %gamma_bf16: tensor<2048xbf16>,
    %w_gate: tensor<2048x256xbf16>,
    %w_up: tensor<2048x256xbf16>
) -> (tensor<4x256xbf16>, tensor<4x256xbf16>) {
  // bf16 -> f32
  %x_f32 = stablehlo.convert %x_bf16 : (tensor<4x2048xbf16>) -> tensor<4x2048xf32>
  %gamma_f32 = stablehlo.convert %gamma_bf16 : (tensor<2048xbf16>) -> tensor<2048xf32>

  // rms_norm chain (shared)
  %x_sq = stablehlo.multiply %x_f32, %x_f32 : tensor<4x2048xf32>
  %init = stablehlo.constant dense<0.0> : tensor<f32>
  %sum = "stablehlo.reduce"(%x_sq, %init) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %add = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %add : tensor<f32>
  }) {dimensions = array<i64: 1>} : (tensor<4x2048xf32>, tensor<f32>) -> tensor<4xf32>

  %scale = stablehlo.constant dense<4.882812e-04> : tensor<f32>
  %scale_bc = "stablehlo.broadcast_in_dim"(%scale) {
    broadcast_dimensions = array<i64>
  } : (tensor<f32>) -> tensor<4xf32>
  %mean = stablehlo.multiply %sum, %scale_bc : tensor<4xf32>

  %eps = stablehlo.constant dense<1.000000e-05> : tensor<f32>
  %eps_bc = "stablehlo.broadcast_in_dim"(%eps) {
    broadcast_dimensions = array<i64>
  } : (tensor<f32>) -> tensor<4xf32>
  %denom = stablehlo.add %mean, %eps_bc : tensor<4xf32>
  %inv = stablehlo.rsqrt %denom : tensor<4xf32>
  %inv_bc = "stablehlo.broadcast_in_dim"(%inv) {
    broadcast_dimensions = array<i64: 0>
  } : (tensor<4xf32>) -> tensor<4x2048xf32>
  %normed = stablehlo.multiply %x_f32, %inv_bc : tensor<4x2048xf32>

  %gamma_bc = "stablehlo.broadcast_in_dim"(%gamma_f32) {
    broadcast_dimensions = array<i64: 1>
  } : (tensor<2048xf32>) -> tensor<4x2048xf32>
  %gamma_mul = stablehlo.multiply %normed, %gamma_bc : tensor<4x2048xf32>

  %converted = stablehlo.convert %gamma_mul : (tensor<4x2048xf32>) -> tensor<4x2048xbf16>

  // Gate matmul
  %gate = "stablehlo.dot_general"(%converted, %w_gate) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >
  } : (tensor<4x2048xbf16>, tensor<2048x256xbf16>) -> tensor<4x256xbf16>

  // Up matmul
  %up = "stablehlo.dot_general"(%converted, %w_up) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >
  } : (tensor<4x2048xbf16>, tensor<2048x256xbf16>) -> tensor<4x256xbf16>

  return %gate, %up : tensor<4x256xbf16>, tensor<4x256xbf16>
}
