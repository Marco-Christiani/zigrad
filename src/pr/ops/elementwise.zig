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

    pub fn emit_primal(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.UnsupportedEqn;

        const lhs = ctx.get_primal(op.operand(0)) orelse return error.UnsupportedEqn;
        const rhs = ctx.get_primal(op.operand(1)) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.add(lhs, rhs);
        ctx.set_primal(op.result(0), out);
    }

    pub fn vjp(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(op.result(0)) orelse return;
        try ctx.add_cot(op.operand(0), out_cot);
        try ctx.add_cot(op.operand(1), out_cot);
    }

    /// JVP: \(\mathrm{d}(x + y) = \mathrm{d}x + \mathrm{d}y\).
    pub fn jvp(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.UnsupportedEqn;

        const dx = ctx.get_tangent(op.operand(0)) orelse return error.UnsupportedEqn;
        const dy = ctx.get_tangent(op.operand(1)) orelse return error.UnsupportedEqn;
        ctx.set_tangent(op.result(0), try ctx.builder.add(dx, dy));
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

    pub fn emit_primal(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.UnsupportedEqn;

        const lhs = ctx.get_primal(op.operand(0)) orelse return error.UnsupportedEqn;
        const rhs = ctx.get_primal(op.operand(1)) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.subtract(lhs, rhs);
        ctx.set_primal(op.result(0), out);
    }

    pub fn vjp(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(op.result(0)) orelse return;
        const rhs_tensor = op.operand(1).as_tensor();

        try ctx.add_cot(op.operand(0), out_cot);
        const neg = try negate_like(ctx.builder, rhs_tensor, out_cot);
        try ctx.add_cot(op.operand(1), neg);
    }

    /// JVP: \(\mathrm{d}(x - y) = \mathrm{d}x - \mathrm{d}y\).
    pub fn jvp(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.UnsupportedEqn;

        const dx = ctx.get_tangent(op.operand(0)) orelse return error.UnsupportedEqn;
        const dy = ctx.get_tangent(op.operand(1)) orelse return error.UnsupportedEqn;
        ctx.set_tangent(op.result(0), try ctx.builder.subtract(dx, dy));
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

    pub fn emit_primal(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.UnsupportedEqn;

        const lhs = ctx.get_primal(op.operand(0)) orelse return error.UnsupportedEqn;
        const rhs = ctx.get_primal(op.operand(1)) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.multiply(lhs, rhs);
        ctx.set_primal(op.result(0), out);
    }

    pub fn vjp(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(op.result(0)) orelse return;
        const lhs_primal = ctx.get_primal(op.operand(0)) orelse return error.UnsupportedEqn;
        const rhs_primal = ctx.get_primal(op.operand(1)) orelse return error.UnsupportedEqn;

        const lhs_contrib = try ctx.builder.multiply(out_cot, rhs_primal);
        const rhs_contrib = try ctx.builder.multiply(out_cot, lhs_primal);

        try ctx.add_cot(op.operand(0), lhs_contrib);
        try ctx.add_cot(op.operand(1), rhs_contrib);
    }

    /// JVP: \(\mathrm{d}(x y) = (\mathrm{d}x) y + x (\mathrm{d}y)\) (product rule).
    pub fn jvp(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.UnsupportedEqn;

        const x = ctx.get_primal(op.operand(0)) orelse return error.UnsupportedEqn;
        const y = ctx.get_primal(op.operand(1)) orelse return error.UnsupportedEqn;
        const dx = ctx.get_tangent(op.operand(0)) orelse return error.UnsupportedEqn;
        const dy = ctx.get_tangent(op.operand(1)) orelse return error.UnsupportedEqn;

        const term1 = try ctx.builder.multiply(dx, y);
        const term2 = try ctx.builder.multiply(x, dy);
        ctx.set_tangent(op.result(0), try ctx.builder.add(term1, term2));
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

    pub fn emit_primal(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.UnsupportedEqn;

        const lhs = ctx.get_primal(op.operand(0)) orelse return error.UnsupportedEqn;
        const rhs = ctx.get_primal(op.operand(1)) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.divide(lhs, rhs);
        ctx.set_primal(op.result(0), out);
    }

    /// VJP: \(\mathrm{d}_{\mathrm{lhs}}(\mathrm{lhs}/\mathrm{rhs}) =
    ///  \frac{\mathrm{dout}}{\mathrm{rhs}}\) and
    /// \(\mathrm{d}_{\mathrm{rhs}}(\mathrm{lhs}/\mathrm{rhs}) =
    ///  -\frac{\mathrm{dout}\,\mathrm{lhs}}{\mathrm{rhs}^2}\) (quotient rule).
    pub fn vjp(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(op.result(0)) orelse return;
        const lhs_primal = ctx.get_primal(op.operand(0)) orelse return error.UnsupportedEqn;
        const rhs_primal = ctx.get_primal(op.operand(1)) orelse return error.UnsupportedEqn;

        // d_lhs = dout / rhs
        const lhs_contrib = try ctx.builder.divide(out_cot, rhs_primal);
        try ctx.add_cot(op.operand(0), lhs_contrib);

        // d_rhs = -dout * lhs / rhs^2
        const rhs_sq = try ctx.builder.multiply(rhs_primal, rhs_primal);
        const lhs_over_rhs_sq = try ctx.builder.divide(lhs_primal, rhs_sq);
        const rhs_contrib = try ctx.builder.multiply(out_cot, lhs_over_rhs_sq);
        const rhs_tensor = op.operand(1).as_tensor();
        const neg = try negate_like(ctx.builder, rhs_tensor, rhs_contrib);
        try ctx.add_cot(op.operand(1), neg);
    }

    /// JVP: \(\mathrm{d}(x/y) = \frac{\mathrm{d}x}{y} -
    ///  \frac{x\,\mathrm{d}y}{y^2}\) (quotient rule).
    pub fn jvp(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.UnsupportedEqn;

        const x = ctx.get_primal(op.operand(0)) orelse return error.UnsupportedEqn;
        const y = ctx.get_primal(op.operand(1)) orelse return error.UnsupportedEqn;
        const dx = ctx.get_tangent(op.operand(0)) orelse return error.UnsupportedEqn;
        const dy = ctx.get_tangent(op.operand(1)) orelse return error.UnsupportedEqn;

        const term1 = try ctx.builder.divide(dx, y);
        const y_sq = try ctx.builder.multiply(y, y);
        const x_dy = try ctx.builder.multiply(x, dy);
        const term2 = try ctx.builder.divide(x_dy, y_sq);
        ctx.set_tangent(op.result(0), try ctx.builder.subtract(term1, term2));
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

    pub fn emit_primal(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.UnsupportedEqn;

        const lhs = ctx.get_primal(op.operand(0)) orelse return error.UnsupportedEqn;
        const rhs = ctx.get_primal(op.operand(1)) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.maximum(lhs, rhs);
        ctx.set_primal(op.result(0), out);
    }

    /// VJP: gradient routes to whichever operand was selected.
    ///  d_lhs = select(lhs >= rhs, cot, 0), d_rhs = select(lhs >= rhs, 0, cot).
    pub fn vjp(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(op.result(0)) orelse return;
        const lhs_primal = ctx.get_primal(op.operand(0)) orelse return error.UnsupportedEqn;
        const rhs_primal = ctx.get_primal(op.operand(1)) orelse return error.UnsupportedEqn;

        const lhs_tensor = op.operand(0).as_tensor();
        const cmp = try ctx.builder.compare(lhs_primal, rhs_primal, .{
            .direction = .GE,
            .compare_type = .FLOAT,
        });
        const zero = try zero_like(ctx.builder, lhs_tensor);

        try ctx.add_cot(op.operand(0), try ctx.builder.select(cmp, out_cot, zero));
        try ctx.add_cot(op.operand(1), try ctx.builder.select(cmp, zero, out_cot));
    }

    pub const format = format_binary_elementwise;
};

// Helpers.

/// Zero scalar broadcast to match `tensor`'s shape and dtype.
/// Handles rank-0 (scalar) tensors by skipping the broadcast.
fn zero_like(bld: *pr.FunctionBuilder, tensor: Tensor) pr.BuildError!*pr.Var {
    const zero = try bld.scalar(tensor.dtype, 0.0);
    if (tensor.shape.rank() == 0) return zero;
    return try bld.broadcast_in_dim(zero, tensor.shape.dims, &.{});
}

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
