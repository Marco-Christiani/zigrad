//! Optimizer building blocks for traced Tensor math.
//!
//! These are convenience functions that emit standard optimizer update
//!  equations during tracing. Users can always write their own update
//!  logic using Tensor ops directly.
const Tensor = @import("../tensor.zig");

/// Apply a single SGD update: `param - lr * grad`.
///
/// Both `param_tensor` and `grad` must be traced-mode Tensors from the same
///  builder.
///
/// Returns the updated parameter tensor.
pub fn sgd_update(param_tensor: Tensor, grad: Tensor, lr: f32) !Tensor {
    const lr_tensor = try scalar_broadcast_like(param_tensor, lr);
    const scaled = try grad.mul(lr_tensor);
    return param_tensor.sub(scaled);
}

/// Create a scalar and broadcast to match `target`'s shape.
fn scalar_broadcast_like(target: Tensor, value: f32) !Tensor {
    const ops = @import("../pr/ops/ops.zig");
    const b = target.mode.traced.builder;
    const lit = ops.types.scalar_literal(target.dtype, value);
    const scalar_id = try b.literal_scalar(lit);
    const scalar = try Tensor.from_id(b, scalar_id);
    if (target.rank() == 0) return scalar;
    return scalar.broadcast_in_dim(target.dims(), &.{});
}
