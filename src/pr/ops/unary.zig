//! Unary elementwise operations.
const std = @import("std");
const types = @import("types.zig");
const pr = @import("../pr.zig");
const Aval = pr.Aval;

fn validate_unary_elementwise(comptime err: pr.ValidationError, op: *const pr.Op) pr.ValidationError!void {
    if (op.inputs.len != 1 or op.outputs.len != 1) return error.InvalidOpArity;

    const operand = op.operand(0).as_tensor();
    const out = op.result(0).as_tensor();
    if (!types.same_tensor_type(operand, out)) return err;
}

fn infer_unary_elementwise(comptime _: pr.ValidationError, inputs: []const *pr.Var) pr.BuildError!Aval {
    if (inputs.len != 1) return error.InvalidOpArity;
    const operand = inputs[0].as_tensor();
    return .{ .tensor = operand };
}

fn format_unary_elementwise(writer: *types.Writer, op: *const pr.Op, _: void) types.FormatError!void {
    if (op.inputs.len > 0) {
        const t = op.operand(0).as_tensor();
        try writer.print("dtype={s}", .{@tagName(t.dtype)});
    }
}

// Exp

pub const exp = struct {
    pub const arity = .{ .in = 1, .out = 1 };

    pub fn validate(op: *const pr.Op, _: void) pr.ValidationError!void {
        return try validate_unary_elementwise(error.ExpTypeMismatch, op);
    }

    pub fn infer_output(_: std.mem.Allocator, inputs: []const *pr.Var, _: void) pr.BuildError!Aval {
        return try infer_unary_elementwise(error.ExpTypeMismatch, inputs);
    }

    pub fn emit_primal(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 1) return error.UnsupportedEqn;

        const operand = ctx.get_primal(op.operand(0)) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.exp(operand);
        ctx.set_primal(op.result(0), out);
    }

    pub fn vjp(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 1) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(op.result(0)) orelse return;
        const out_primal = ctx.get_primal(op.result(0)) orelse return error.UnsupportedEqn;
        const contrib = try ctx.builder.multiply(out_cot, out_primal);
        try ctx.add_cot(op.operand(0), contrib);
    }

    /// JVP: \(\mathrm{d}(\exp(x)) = \exp(x)\,\mathrm{d}x\).
    pub fn jvp(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 1) return error.UnsupportedEqn;

        const dx = ctx.get_tangent(op.operand(0)) orelse return error.UnsupportedEqn;
        const out_primal = ctx.get_primal(op.result(0)) orelse return error.UnsupportedEqn;
        ctx.set_tangent(op.result(0), try ctx.builder.multiply(out_primal, dx));
    }

    pub const format = format_unary_elementwise;
};

// Log

pub const log = struct {
    pub const arity = .{ .in = 1, .out = 1 };

    pub fn validate(op: *const pr.Op, _: void) pr.ValidationError!void {
        return try validate_unary_elementwise(error.LogTypeMismatch, op);
    }

    pub fn infer_output(_: std.mem.Allocator, inputs: []const *pr.Var, _: void) pr.BuildError!Aval {
        return try infer_unary_elementwise(error.LogTypeMismatch, inputs);
    }

    pub fn emit_primal(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 1) return error.UnsupportedEqn;

        const operand = ctx.get_primal(op.operand(0)) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.log(operand);
        ctx.set_primal(op.result(0), out);
    }

    pub fn vjp(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 1) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(op.result(0)) orelse return;
        const operand = ctx.get_primal(op.operand(0)) orelse return error.UnsupportedEqn;
        const contrib = try ctx.builder.divide(out_cot, operand);
        try ctx.add_cot(op.operand(0), contrib);
    }

    /// JVP: \(\mathrm{d}(\log(x)) = \mathrm{d}x / x\).
    pub fn jvp(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 1) return error.UnsupportedEqn;

        const dx = ctx.get_tangent(op.operand(0)) orelse return error.UnsupportedEqn;
        const x = ctx.get_primal(op.operand(0)) orelse return error.UnsupportedEqn;
        ctx.set_tangent(op.result(0), try ctx.builder.divide(dx, x));
    }

    pub const format = format_unary_elementwise;
};

// Conversion operation.

pub const convert = struct {
    pub const arity = .{ .in = 1, .out = 1 };

    pub fn validate(op: *const pr.Op, out_dtype: pr.DType) pr.ValidationError!void {
        if (op.inputs.len != 1 or op.outputs.len != 1) return error.InvalidOpArity;

        const operand = op.operand(0).as_tensor();
        const out = op.result(0).as_tensor();
        if (out.dtype != out_dtype) return error.ConvertTypeMismatch;
        if (!std.mem.eql(i64, operand.shape.dims, out.shape.dims)) return error.ConvertTypeMismatch;
    }

    pub fn infer_output(_: std.mem.Allocator, inputs: []const *pr.Var, out_dtype: pr.DType) pr.BuildError!Aval {
        if (inputs.len != 1) return error.InvalidOpArity;
        const operand = inputs[0].as_tensor();
        return .{ .tensor = .{ .dtype = out_dtype, .shape = operand.shape } };
    }

    pub fn emit_primal(ctx: types.AdContext, op: *const pr.Op, out_dtype: pr.DType) types.AdError!void {
        if (op.inputs.len != 1) return error.UnsupportedEqn;

        const operand = ctx.get_primal(op.operand(0)) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.convert(operand, out_dtype);
        ctx.set_primal(op.result(0), out);
    }

    pub fn vjp(ctx: types.AdContext, op: *const pr.Op, _: pr.DType) types.AdError!void {
        if (op.inputs.len != 1) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(op.result(0)) orelse return;
        const in_tensor = op.operand(0).as_tensor();
        const out_tensor = op.result(0).as_tensor();
        const cot = if (out_tensor.dtype == in_tensor.dtype)
            out_cot
        else
            try ctx.builder.convert(out_cot, in_tensor.dtype);
        try ctx.add_cot(op.operand(0), cot);
    }

    /// JVP: \(\mathrm{d}(\operatorname{convert}(x, T)) =
    ///  \operatorname{convert}(\mathrm{d}x, T)\).
    pub fn jvp(ctx: types.AdContext, op: *const pr.Op, out_dtype: pr.DType) types.AdError!void {
        if (op.inputs.len != 1) return error.UnsupportedEqn;

        const dx = ctx.get_tangent(op.operand(0)) orelse return error.UnsupportedEqn;
        const dx_tensor = dx.as_tensor();
        const result = if (dx_tensor.dtype == out_dtype) dx else try ctx.builder.convert(dx, out_dtype);
        ctx.set_tangent(op.result(0), result);
    }
};

// Rsqrt

pub const rsqrt = struct {
    pub const arity = .{ .in = 1, .out = 1 };

    pub fn validate(op: *const pr.Op, _: void) pr.ValidationError!void {
        return try validate_unary_elementwise(error.RsqrtTypeMismatch, op);
    }

    pub fn infer_output(_: std.mem.Allocator, inputs: []const *pr.Var, _: void) pr.BuildError!Aval {
        return try infer_unary_elementwise(error.RsqrtTypeMismatch, inputs);
    }

    pub fn emit_primal(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 1) return error.UnsupportedEqn;

        const operand = ctx.get_primal(op.operand(0)) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.rsqrt(operand);
        ctx.set_primal(op.result(0), out);
    }

    /// VJP: \(\mathrm{d}(\operatorname{rsqrt}(x)) =
    ///  -0.5\,\operatorname{rsqrt}(x)^3\,\mathrm{d}x\).
    /// Uses the forward output \(y = \operatorname{rsqrt}(x)\) directly:
    /// \(\mathrm{scale} = -0.5\,y^3\).
    pub fn vjp(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 1) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(op.result(0)) orelse return;
        // y = rsqrt(x), reuse from forward pass to avoid recomputing sqrt.
        const out_primal = ctx.get_primal(op.result(0)) orelse return error.UnsupportedEqn;
        const y2 = try ctx.builder.multiply(out_primal, out_primal); // y^2
        const y3 = try ctx.builder.multiply(y2, out_primal); // y^3

        // scale = y^3 * (-0.5), then contrib = dout * scale
        const tensor = op.operand(0).as_tensor();
        const neg_half = try ctx.builder.scalar_broadcast(tensor.dtype, tensor.shape.dims, -0.5);
        const scale = try ctx.builder.multiply(y3, neg_half);
        const contrib = try ctx.builder.multiply(out_cot, scale);
        try ctx.add_cot(op.operand(0), contrib);
    }

    /// JVP: \(\mathrm{d}(\operatorname{rsqrt}(x)) =
    ///  -0.5\,\operatorname{rsqrt}(x)^3\,\mathrm{d}x\).
    pub fn jvp(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 1) return error.UnsupportedEqn;

        const dx = ctx.get_tangent(op.operand(0)) orelse return error.UnsupportedEqn;
        const out_primal = ctx.get_primal(op.result(0)) orelse return error.UnsupportedEqn;
        const y2 = try ctx.builder.multiply(out_primal, out_primal); // y^2
        const y3 = try ctx.builder.multiply(y2, out_primal); // y^3

        const tensor = op.operand(0).as_tensor();
        const neg_half = try ctx.builder.scalar_broadcast(tensor.dtype, tensor.shape.dims, -0.5);
        const scale = try ctx.builder.multiply(y3, neg_half);
        ctx.set_tangent(op.result(0), try ctx.builder.multiply(scale, dx));
    }

    pub const format = format_unary_elementwise;
};

// Logistic (sigmoid)

pub const logistic = struct {
    pub const arity = .{ .in = 1, .out = 1 };

    pub fn validate(op: *const pr.Op, _: void) pr.ValidationError!void {
        return try validate_unary_elementwise(error.LogisticTypeMismatch, op);
    }

    pub fn infer_output(_: std.mem.Allocator, inputs: []const *pr.Var, _: void) pr.BuildError!Aval {
        return try infer_unary_elementwise(error.LogisticTypeMismatch, inputs);
    }

    pub fn emit_primal(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 1) return error.UnsupportedEqn;

        const operand = ctx.get_primal(op.operand(0)) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.logistic(operand);
        ctx.set_primal(op.result(0), out);
    }

    /// VJP: \(\mathrm{d}(\operatorname{sigmoid}(x)) =
    ///  \operatorname{sigmoid}(x)(1 - \operatorname{sigmoid}(x))\,\mathrm{d}x\).
    /// Uses the forward output directly to avoid recomputing the sigmoid.
    pub fn vjp(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 1) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(op.result(0)) orelse return;
        // y = sigmoid(x), reuse from forward pass.
        const out_primal = ctx.get_primal(op.result(0)) orelse return error.UnsupportedEqn;

        // slope = y * (1 - y), the sigmoid derivative.
        const tensor = op.operand(0).as_tensor();
        const ones = try ctx.builder.scalar_broadcast(tensor.dtype, tensor.shape.dims, 1.0);
        const one_minus = try ctx.builder.subtract(ones, out_primal);
        const slope = try ctx.builder.multiply(out_primal, one_minus);
        const contrib = try ctx.builder.multiply(out_cot, slope);
        try ctx.add_cot(op.operand(0), contrib);
    }

    /// JVP: \(\mathrm{d}(\operatorname{sigmoid}(x)) =
    ///  \operatorname{sigmoid}(x)(1 - \operatorname{sigmoid}(x))\,\mathrm{d}x\).
    pub fn jvp(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 1) return error.UnsupportedEqn;

        const dx = ctx.get_tangent(op.operand(0)) orelse return error.UnsupportedEqn;
        const out_primal = ctx.get_primal(op.result(0)) orelse return error.UnsupportedEqn;

        const tensor = op.operand(0).as_tensor();
        const ones = try ctx.builder.scalar_broadcast(tensor.dtype, tensor.shape.dims, 1.0);
        const one_minus = try ctx.builder.subtract(ones, out_primal);
        const slope = try ctx.builder.multiply(out_primal, one_minus);
        ctx.set_tangent(op.result(0), try ctx.builder.multiply(slope, dx));
    }

    pub const format = format_unary_elementwise;
};
