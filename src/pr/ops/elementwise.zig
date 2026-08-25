//! Elementwise Operations
//! Binary ops that operate element-by-element on tensors of the same shape.
const std = @import("std");
const types = @import("types.zig");
const pr = @import("../pr.zig");
const Tensor = pr.Tensor;
const Aval = pr.Aval;

fn validate_binary_elementwise(comptime err: pr.ValidationError, op: *const pr.Op) pr.ValidationError!void {
    if (op.inputs.len != 2 or op.outputs.len != 1) return error.InvalidOpArity;

    const lhs = op.operand(0).as_tensor();
    const rhs = op.operand(1).as_tensor();
    const out = op.result(0).as_tensor();

    if (!types.same_tensor_type(lhs, rhs) or !types.same_tensor_type(lhs, out)) return err;
}

fn infer_binary_elementwise(comptime err: pr.ValidationError, inputs: []const *pr.Var) pr.BuildError!Aval {
    if (inputs.len != 2) return error.InvalidOpArity;
    const lhs = inputs[0].as_tensor();
    const rhs = inputs[1].as_tensor();
    if (!types.same_tensor_type(lhs, rhs)) return err;
    return .{ .tensor = lhs };
}

fn format_binary_elementwise(writer: *types.Writer, op: *const pr.Op, _: void) types.FormatError!void {
    if (op.inputs.len > 0) {
        const t = op.operand(0).as_tensor();
        try writer.print("dtype={s}", .{@tagName(t.dtype)});
    }
}

// Add

pub const add = struct {
    pub const arity = .{ .in = 2, .out = 1 };

    pub fn validate(op: *const pr.Op, _: void) pr.ValidationError!void {
        return try validate_binary_elementwise(error.AddTypeMismatch, op);
    }

    pub fn infer_output(_: std.mem.Allocator, inputs: []const *pr.Var, _: void) pr.BuildError!Aval {
        return try infer_binary_elementwise(error.AddTypeMismatch, inputs);
    }

    pub fn vjp(ctx: types.VjpContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.InvalidOpArity;

        const out_cot = ctx.cotangent(op.result(0)) orelse return;
        try ctx.add_cotangent(op.operand(0), out_cot);
        try ctx.add_cotangent(op.operand(1), out_cot);
    }

    /// JVP: \(\mathrm{d}(x + y) = \mathrm{d}x + \mathrm{d}y\).
    pub fn jvp(ctx: types.JvpContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.InvalidOpArity;

        const dx = ctx.tangent(op.operand(0));
        const dy = ctx.tangent(op.operand(1));
        if (dx) |value| try ctx.add_tangent(op.result(0), value);
        if (dy) |value| try ctx.add_tangent(op.result(0), value);
    }

    pub const format = format_binary_elementwise;
};

// Subtract

pub const subtract = struct {
    pub const arity = .{ .in = 2, .out = 1 };

    pub fn validate(op: *const pr.Op, _: void) pr.ValidationError!void {
        return try validate_binary_elementwise(error.SubtractTypeMismatch, op);
    }

    pub fn infer_output(_: std.mem.Allocator, inputs: []const *pr.Var, _: void) pr.BuildError!Aval {
        return try infer_binary_elementwise(error.SubtractTypeMismatch, inputs);
    }

    pub fn vjp(ctx: types.VjpContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.InvalidOpArity;

        const out_cot = ctx.cotangent(op.result(0)) orelse return;
        const rhs_tensor = op.operand(1).as_tensor();

        try ctx.add_cotangent(op.operand(0), out_cot);
        const neg = try negate_like(ctx.builder, rhs_tensor, out_cot);
        try ctx.add_cotangent(op.operand(1), neg);
    }

    /// JVP: \(\mathrm{d}(x - y) = \mathrm{d}x - \mathrm{d}y\).
    pub fn jvp(ctx: types.JvpContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.InvalidOpArity;

        const dx = ctx.tangent(op.operand(0));
        const dy = ctx.tangent(op.operand(1));
        if (dx) |value| try ctx.add_tangent(op.result(0), value);
        if (dy) |value| {
            const negated = try negate_like(ctx.builder, op.operand(1).as_tensor(), value);
            try ctx.add_tangent(op.result(0), negated);
        }
    }

    pub const format = format_binary_elementwise;
};

// Multiply

pub const multiply = struct {
    pub const arity = .{ .in = 2, .out = 1 };

    pub fn validate(op: *const pr.Op, _: void) pr.ValidationError!void {
        return try validate_binary_elementwise(error.MultiplyTypeMismatch, op);
    }

    pub fn infer_output(_: std.mem.Allocator, inputs: []const *pr.Var, _: void) pr.BuildError!Aval {
        return try infer_binary_elementwise(error.MultiplyTypeMismatch, inputs);
    }

    pub fn vjp(ctx: types.VjpContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.InvalidOpArity;

        const out_cot = ctx.cotangent(op.result(0)) orelse return;
        const lhs_primal = try ctx.primal(op.operand(0));
        const rhs_primal = try ctx.primal(op.operand(1));

        const lhs_contrib = try ctx.builder.multiply(out_cot, rhs_primal);
        const rhs_contrib = try ctx.builder.multiply(out_cot, lhs_primal);

        try ctx.add_cotangent(op.operand(0), lhs_contrib);
        try ctx.add_cotangent(op.operand(1), rhs_contrib);
    }

    /// JVP: \(\mathrm{d}(x y) = (\mathrm{d}x) y + x (\mathrm{d}y)\) (product rule).
    pub fn jvp(ctx: types.JvpContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.InvalidOpArity;

        const x = try ctx.primal(op.operand(0));
        const y = try ctx.primal(op.operand(1));
        if (ctx.tangent(op.operand(0))) |dx| {
            try ctx.add_tangent(op.result(0), try ctx.builder.multiply(dx, y));
        }
        if (ctx.tangent(op.operand(1))) |dy| {
            try ctx.add_tangent(op.result(0), try ctx.builder.multiply(x, dy));
        }
    }

    pub const format = format_binary_elementwise;
};

// Divide

pub const divide = struct {
    pub const arity = .{ .in = 2, .out = 1 };

    pub fn validate(op: *const pr.Op, _: void) pr.ValidationError!void {
        return try validate_binary_elementwise(error.DivideTypeMismatch, op);
    }

    pub fn infer_output(_: std.mem.Allocator, inputs: []const *pr.Var, _: void) pr.BuildError!Aval {
        return try infer_binary_elementwise(error.DivideTypeMismatch, inputs);
    }

    /// VJP: \(\mathrm{d}_{\mathrm{lhs}}(\mathrm{lhs}/\mathrm{rhs}) =
    ///  \frac{\mathrm{dout}}{\mathrm{rhs}}\) and
    /// \(\mathrm{d}_{\mathrm{rhs}}(\mathrm{lhs}/\mathrm{rhs}) =
    ///  -\frac{\mathrm{dout}\,\mathrm{lhs}}{\mathrm{rhs}^2}\) (quotient rule).
    pub fn vjp(ctx: types.VjpContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.InvalidOpArity;

        const out_cot = ctx.cotangent(op.result(0)) orelse return;
        const lhs_primal = try ctx.primal(op.operand(0));
        const rhs_primal = try ctx.primal(op.operand(1));

        // d_lhs = dout / rhs
        const lhs_contrib = try ctx.builder.divide(out_cot, rhs_primal);
        try ctx.add_cotangent(op.operand(0), lhs_contrib);

        // d_rhs = -dout * lhs / rhs^2
        const rhs_sq = try ctx.builder.multiply(rhs_primal, rhs_primal);
        const lhs_over_rhs_sq = try ctx.builder.divide(lhs_primal, rhs_sq);
        const rhs_contrib = try ctx.builder.multiply(out_cot, lhs_over_rhs_sq);
        const rhs_tensor = op.operand(1).as_tensor();
        const neg = try negate_like(ctx.builder, rhs_tensor, rhs_contrib);
        try ctx.add_cotangent(op.operand(1), neg);
    }

    /// JVP: \(\mathrm{d}(x/y) = \frac{\mathrm{d}x}{y} -
    ///  \frac{x\,\mathrm{d}y}{y^2}\) (quotient rule).
    pub fn jvp(ctx: types.JvpContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.InvalidOpArity;

        const x = try ctx.primal(op.operand(0));
        const y = try ctx.primal(op.operand(1));
        if (ctx.tangent(op.operand(0))) |dx| {
            try ctx.add_tangent(op.result(0), try ctx.builder.divide(dx, y));
        }
        if (ctx.tangent(op.operand(1))) |dy| {
            const y_sq = try ctx.builder.multiply(y, y);
            const x_dy = try ctx.builder.multiply(x, dy);
            const term = try ctx.builder.divide(x_dy, y_sq);
            const negated = try negate_like(ctx.builder, op.operand(1).as_tensor(), term);
            try ctx.add_tangent(op.result(0), negated);
        }
    }

    pub const format = format_binary_elementwise;
};

// Maximum

pub const maximum = struct {
    pub const arity = .{ .in = 2, .out = 1 };

    pub fn validate(op: *const pr.Op, _: void) pr.ValidationError!void {
        return try validate_binary_elementwise(error.MaximumTypeMismatch, op);
    }

    pub fn infer_output(_: std.mem.Allocator, inputs: []const *pr.Var, _: void) pr.BuildError!Aval {
        return try infer_binary_elementwise(error.MaximumTypeMismatch, inputs);
    }

    /// VJP: gradient routes to whichever operand was selected.
    ///  d_lhs = select(lhs >= rhs, cot, 0), d_rhs = select(lhs >= rhs, 0, cot).
    pub fn vjp(ctx: types.VjpContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.InvalidOpArity;

        const out_cot = ctx.cotangent(op.result(0)) orelse return;
        const lhs_primal = try ctx.primal(op.operand(0));
        const rhs_primal = try ctx.primal(op.operand(1));

        const lhs_tensor = op.operand(0).as_tensor();
        const cmp = try ctx.builder.compare(lhs_primal, rhs_primal, .{
            .direction = .GE,
            .compare_type = .FLOAT,
        });
        const zero = try ctx.zero_like(lhs_tensor);

        try ctx.add_cotangent(op.operand(0), try ctx.builder.select(cmp, out_cot, zero));
        try ctx.add_cotangent(op.operand(1), try ctx.builder.select(cmp, zero, out_cot));
    }

    /// JVP: route the tangent through the selected operand, with ties using `lhs`.
    pub fn jvp(ctx: types.JvpContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.InvalidOpArity;

        const lhs = try ctx.primal(op.operand(0));
        const rhs = try ctx.primal(op.operand(1));
        const lhs_tangent = try ctx.tangent_or_zero(op.operand(0));
        const rhs_tangent = try ctx.tangent_or_zero(op.operand(1));
        const select_lhs = try ctx.builder.compare(lhs, rhs, .{
            .direction = .GE,
            .compare_type = .FLOAT,
        });
        ctx.set_tangent(
            op.result(0),
            try ctx.builder.select(select_lhs, lhs_tangent, rhs_tangent),
        );
    }

    pub const format = format_binary_elementwise;
};

// Helpers.

/// Negate a value by multiplying with a -1 scalar broadcast to match `tensor`'s shape.
/// Handles rank-0 (scalar) tensors by skipping the broadcast.
fn negate_like(bld: *pr.FunctionBuilder, tensor: Tensor, value: *pr.Var) pr.BuildError!*pr.Var {
    const minus_one = try bld.scalar(tensor.dtype, -1.0);
    const minus_one_full = if (tensor.shape.rank() == 0)
        minus_one
    else
        try bld.broadcast_in_dim(minus_one, tensor.shape.dims, &.{});
    return try bld.multiply(value, minus_one_full);
}
