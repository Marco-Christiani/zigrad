//! Optimizer building blocks for traced Tensor math.
//!
//! These are convenience functions that emit standard optimizer update
//!  equations during tracing. Users can always write their own update
//!  logic using Tensor ops directly.
const Tensor = @import("../tensor.zig");

pub const SGD = struct {
    const Self = @This();
    lr: f32,

    /// Apply a single SGD update: `param - lr * grad`.
    ///
    /// Both `param` and `grad` must be traced-mode Tensors from the same
    ///  builder.
    ///
    /// Returns the updated parameter tensor.
    pub fn update(self: Self, param: Tensor, grad: Tensor) !Tensor {
        const scaled = try grad.mul(try Tensor.constant_like(grad, self.lr));
        return try param.sub(scaled);
    }
};
