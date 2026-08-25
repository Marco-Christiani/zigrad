//! Context types for op implementations and helpers.
const std = @import("std");
const pr = @import("../pr.zig");

/// Context passed to forward-mode derivative rules.
///
/// Maps are indexed by `Var.id` and sized by the source function's
///  `var_count`. An absent tangent represents structural zero.
pub const JvpContext = struct {
    /// Builder receiving generated tangent operations.
    builder: *pr.FunctionBuilder,
    /// Rebuilt primal values indexed by source `Var.id`.
    primals: []?*pr.Var,
    /// Propagated tangents indexed by source `Var.id`.
    tangents: []?*pr.Var,
    /// Scratch allocator available to derivative rules.
    allocator: std.mem.Allocator,

    /// Return the mapped primal value.
    pub fn primal(self: JvpContext, v: *const pr.Var) error{MissingPrimal}!*pr.Var {
        std.debug.assert(v.id < self.primals.len);
        return self.primals[v.id] orelse error.MissingPrimal;
    }

    pub fn set_primal(self: JvpContext, v: *const pr.Var, value: *pr.Var) void {
        std.debug.assert(v.id < self.primals.len);
        self.primals[v.id] = value;
    }

    pub fn tangent(self: JvpContext, v: *const pr.Var) ?*pr.Var {
        std.debug.assert(v.id < self.tangents.len);
        return self.tangents[v.id];
    }

    pub fn set_tangent(self: JvpContext, v: *const pr.Var, value: *pr.Var) void {
        std.debug.assert(v.id < self.tangents.len);
        self.tangents[v.id] = value;
    }

    /// Return `v`'s tangent, materializing zero when it is structurally absent.
    pub fn tangent_or_zero(self: JvpContext, v: *const pr.Var) pr.BuildError!*pr.Var {
        return self.tangent(v) orelse try self.zero_like(v.as_tensor());
    }

    /// Emit a zero with `tensor`'s dtype and shape.
    pub fn zero_like(self: JvpContext, tensor: pr.Tensor) pr.BuildError!*pr.Var {
        const zero = try self.builder.scalar(tensor.dtype, 0.0);
        if (tensor.shape.rank() == 0) return zero;
        return try self.builder.broadcast_in_dim(zero, tensor.shape.dims, &.{});
    }

    /// Add one tangent contribution to `v`.
    pub fn add_tangent(self: JvpContext, v: *const pr.Var, contribution: *pr.Var) pr.BuildError!void {
        std.debug.assert(v.id < self.tangents.len);
        if (self.tangents[v.id]) |existing| {
            self.tangents[v.id] = try self.builder.add(existing, contribution);
        } else {
            self.tangents[v.id] = contribution;
        }
    }
};

/// Context passed to reverse-mode derivative rules.
///
/// Maps are indexed by `Var.id` and sized by the source function's
///  `var_count`. An absent cotangent represents structural zero.
pub const VjpContext = struct {
    /// Builder receiving generated cotangent operations.
    builder: *pr.FunctionBuilder,
    /// Rebuilt primal values indexed by source `Var.id`.
    primals: []?*pr.Var,
    /// Accumulated cotangents indexed by source `Var.id`.
    cotangents: []?*pr.Var,
    /// Scratch allocator available to derivative rules.
    allocator: std.mem.Allocator,

    /// Return the mapped primal value.
    pub fn primal(self: VjpContext, v: *const pr.Var) error{MissingPrimal}!*pr.Var {
        std.debug.assert(v.id < self.primals.len);
        return self.primals[v.id] orelse error.MissingPrimal;
    }

    pub fn cotangent(self: VjpContext, v: *const pr.Var) ?*pr.Var {
        std.debug.assert(v.id < self.cotangents.len);
        return self.cotangents[v.id];
    }

    /// Add one cotangent contribution to `v`.
    pub fn add_cotangent(self: VjpContext, v: *const pr.Var, contribution: *pr.Var) pr.BuildError!void {
        std.debug.assert(v.id < self.cotangents.len);
        if (self.cotangents[v.id]) |existing| {
            self.cotangents[v.id] = try self.builder.add(existing, contribution);
        } else {
            self.cotangents[v.id] = contribution;
        }
    }

    /// Emit a zero with `tensor`'s dtype and shape.
    pub fn zero_like(self: VjpContext, tensor: pr.Tensor) pr.BuildError!*pr.Var {
        const zero = try self.builder.scalar(tensor.dtype, 0.0);
        if (tensor.shape.rank() == 0) return zero;
        return try self.builder.broadcast_in_dim(zero, tensor.shape.dims, &.{});
    }
};

pub const AdError = pr.BuildError || error{
    /// No local derivative rule is registered for an active operation.
    MissingDerivativeRule,
    /// A local derivative rule does not support the operation's configuration.
    UnsupportedDerivative,
    /// A derivative rule requires a primal value absent from the transform map.
    MissingPrimal,
    /// Generated derivative functions or their index maps disagree.
    InvalidLinearization,
    /// An explicitly selected value has no standard dual.
    NonDifferentiableSelection,
    /// An index in `VjpOpts.wrt` is out of range for the source function.
    WrtIndexOutOfRange,
    /// An index in `VjpOpts.of` is out of range for the source function.
    OfIndexOutOfRange,
    /// `VjpOpts.of` provides no output cotangent seeds.
    EmptyOutputSelection,
};

/// Reports whether standard PR AD defines dual values for `aval`.
pub fn is_differentiable(aval: pr.Aval) bool {
    return switch (aval) {
        .tensor => |tensor| switch (tensor.dtype) {
            .f16, .bf16, .f32, .f64 => true,
            else => false,
        },
    };
}

pub const Writer = std.Io.Writer;
pub const FormatError = Writer.Error;

pub fn same_tensor_type(a: pr.Tensor, b: pr.Tensor) bool {
    if (a.dtype != b.dtype) return false;
    if (a.shape.rank() != b.shape.rank()) return false;
    return std.mem.eql(i64, a.shape.dims, b.shape.dims);
}

test "AD contexts distinguish missing primals from structural zeros" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "context");
    defer builder.deinit();
    const source = try builder.param_tensor(.f32, &.{4});

    var primals = [_]?*pr.Var{null};
    var tangents = [_]?*pr.Var{null};
    const jvp_ctx = JvpContext{
        .builder = &builder,
        .primals = &primals,
        .tangents = &tangents,
        .allocator = std.testing.allocator,
    };
    var cotangents = [_]?*pr.Var{null};
    const vjp_ctx = VjpContext{
        .builder = &builder,
        .primals = &primals,
        .cotangents = &cotangents,
        .allocator = std.testing.allocator,
    };

    try std.testing.expectError(error.MissingPrimal, jvp_ctx.primal(source));
    try std.testing.expectEqual(@as(?*pr.Var, null), jvp_ctx.tangent(source));
    try std.testing.expectError(error.MissingPrimal, vjp_ctx.primal(source));
    try std.testing.expectEqual(@as(?*pr.Var, null), vjp_ctx.cotangent(source));
    try std.testing.expect(!@hasDecl(JvpContext, "cotangent"));
    try std.testing.expect(!@hasDecl(VjpContext, "tangent"));
}
