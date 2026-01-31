/// Elementwise Operations
/// Binary ops that operate element-by-element on tensors of the same shape.
const std = @import("std");
const types = @import("types.zig");
const pr = @import("../pr.zig");
const mlir = @import("../../ffi/mlir/mlir.zig");
const stablehlo = @import("../../ffi/mlir/dialects/stablehlo.zig");

// Shared implementation for binary elementwise ops

fn validate_binary_elementwise(comptime err: pr.ValidationError, ctx: types.ValidateContext) pr.ValidationError!void {
    const inputs = ctx.inputs();
    const outputs = ctx.outputs();
    if (inputs.len != 2 or outputs.len != 1) return error.InvalidEqnArity;

    const lhs = try ctx.tensor_of(inputs[0]);
    const rhs = try ctx.tensor_of(inputs[1]);
    const out = try ctx.tensor_of(outputs[0]);

    if (!types.same_tensor_type(lhs, rhs) or !types.same_tensor_type(lhs, out)) return err;
}

fn infer_binary_elementwise(comptime err: pr.ValidationError, ctx: types.InferContext) pr.BuildError!types.Aval {
    if (ctx.inputs.len != 2) return error.InvalidEqnArity;
    const lhs = try ctx.tensor_of(ctx.inputs[0]);
    const rhs = try ctx.tensor_of(ctx.inputs[1]);
    if (!types.same_tensor_type(lhs, rhs)) return err;
    return .{ .tensor = lhs };
}

fn lower_binary_elementwise(
    comptime lowerFn: fn (mlir.Context, mlir.Value, mlir.Value, mlir.Location) mlir.Operation,
    ctx: types.LowerContext,
    eqn: pr.Eqn,
) types.LowerError!void {
    const inputs = ctx.inputs(eqn);
    const outputs = ctx.outputs(eqn);

    const lhs = ctx.get_value(inputs[0]) orelse return error.InvalidProgram;
    const rhs = ctx.get_value(inputs[1]) orelse return error.InvalidProgram;

    const op = lowerFn(ctx.mlir_ctx, lhs, rhs, ctx.loc);
    ctx.block.append_operation(op);
    ctx.set_value(outputs[0], op.result(0));
}

// AD helpers

fn vjp_forward_binary_elementwise(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
    const inputs = ctx.inputs(eqn);
    const outputs = ctx.outputs(eqn);
    if (inputs.len != 2) return error.UnsupportedEqn;

    const lhs = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
    const rhs = ctx.get_primal(inputs[1]) orelse return error.UnsupportedEqn;
    const out = try ctx.builder.add(lhs, rhs);
    ctx.set_primal(outputs[0], out);
}

// Format helper

fn format_binary_elementwise(writer: *types.Writer, ctx: types.FormatContext) types.FormatError!void {
    if (ctx.input_tensor(0)) |t| {
        try writer.print("dtype={s}", .{@tagName(t.dtype)});
    }
}

// ============================================================================
// Add
// ============================================================================

pub const add = struct {
    pub const arity = .{ .in = 2, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        return validate_binary_elementwise(error.AddTypeMismatch, ctx);
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        return infer_binary_elementwise(error.AddTypeMismatch, ctx);
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        return lower_binary_elementwise(stablehlo.add, ctx, eqn);
    }

    pub fn vjp_forward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 2) return error.UnsupportedEqn;

        const lhs = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const rhs = ctx.get_primal(inputs[1]) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.add(lhs, rhs);
        ctx.set_primal(outputs[0], out);
    }

    pub fn vjp_backward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 2) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(outputs[0]) orelse return;
        // d/d(lhs) = 1, d/d(rhs) = 1
        try ctx.add_cot(inputs[0], out_cot);
        try ctx.add_cot(inputs[1], out_cot);
    }

    pub const format = format_binary_elementwise;
};

// ============================================================================
// Subtract
// ============================================================================

pub const subtract = struct {
    pub const arity = .{ .in = 2, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        return validate_binary_elementwise(error.SubtractTypeMismatch, ctx);
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        return infer_binary_elementwise(error.SubtractTypeMismatch, ctx);
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        return lower_binary_elementwise(stablehlo.subtract, ctx, eqn);
    }

    pub fn vjp_forward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 2) return error.UnsupportedEqn;

        const lhs = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const rhs = ctx.get_primal(inputs[1]) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.subtract(lhs, rhs);
        ctx.set_primal(outputs[0], out);
    }

    pub fn vjp_backward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 2) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(outputs[0]) orelse return;
        const rhs_tensor = ctx.tensor_of(inputs[1]);

        // d/d(lhs) = 1, d/d(rhs) = -1
        try ctx.add_cot(inputs[0], out_cot);
        const neg = try negate_like(ctx.builder, rhs_tensor, out_cot);
        try ctx.add_cot(inputs[1], neg);
    }

    pub const format = format_binary_elementwise;
};

// ============================================================================
// Multiply
// ============================================================================

pub const multiply = struct {
    pub const arity = .{ .in = 2, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        return validate_binary_elementwise(error.MultiplyTypeMismatch, ctx);
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        return infer_binary_elementwise(error.MultiplyTypeMismatch, ctx);
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        return lower_binary_elementwise(stablehlo.multiply, ctx, eqn);
    }

    pub fn vjp_forward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 2) return error.UnsupportedEqn;

        const lhs = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const rhs = ctx.get_primal(inputs[1]) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.multiply(lhs, rhs);
        ctx.set_primal(outputs[0], out);
    }

    pub fn vjp_backward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 2) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(outputs[0]) orelse return;
        const lhs_primal = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const rhs_primal = ctx.get_primal(inputs[1]) orelse return error.UnsupportedEqn;

        // d/d(lhs) = rhs, d/d(rhs) = lhs
        const lhs_contrib = try ctx.builder.multiply(out_cot, rhs_primal);
        const rhs_contrib = try ctx.builder.multiply(out_cot, lhs_primal);

        try ctx.add_cot(inputs[0], lhs_contrib);
        try ctx.add_cot(inputs[1], rhs_contrib);
    }

    pub const format = format_binary_elementwise;
};

// =========================================================================
// Divide
// =========================================================================

pub const divide = struct {
    pub const arity = .{ .in = 2, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        return validate_binary_elementwise(error.DivideTypeMismatch, ctx);
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        return infer_binary_elementwise(error.DivideTypeMismatch, ctx);
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        return lower_binary_elementwise(stablehlo.divide, ctx, eqn);
    }

    pub fn vjp_forward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 2) return error.UnsupportedEqn;

        const lhs = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const rhs = ctx.get_primal(inputs[1]) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.divide(lhs, rhs);
        ctx.set_primal(outputs[0], out);
    }

    pub fn vjp_backward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 2) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(outputs[0]) orelse return;
        const lhs_primal = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const rhs_primal = ctx.get_primal(inputs[1]) orelse return error.UnsupportedEqn;

        const rhs_sq = try ctx.builder.multiply(rhs_primal, rhs_primal);
        const lhs_over_rhs_sq = try ctx.builder.divide(lhs_primal, rhs_sq);
        const rhs_contrib = try ctx.builder.multiply(out_cot, lhs_over_rhs_sq);

        const lhs_contrib = try ctx.builder.divide(out_cot, rhs_primal);
        try ctx.add_cot(inputs[0], lhs_contrib);

        const rhs_tensor = ctx.tensor_of(inputs[1]);
        const neg = try negate_like(ctx.builder, rhs_tensor, rhs_contrib);
        try ctx.add_cot(inputs[1], neg);
    }

    pub const format = format_binary_elementwise;
};

// ============================================================================
// Maximum
// ============================================================================

pub const maximum = struct {
    pub const arity = .{ .in = 2, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        return validate_binary_elementwise(error.MaximumTypeMismatch, ctx);
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        return infer_binary_elementwise(error.MaximumTypeMismatch, ctx);
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        return lower_binary_elementwise(stablehlo.maximum, ctx, eqn);
    }

    // No vjp_forward/vjp_backward - maximum AD not yet supported

    pub const format = format_binary_elementwise;
};

// ============================================================================
// Helpers
// ============================================================================

fn negate_like(bld: *pr.FunctionBuilder, tensor: types.Tensor, value: types.VarId) pr.BuildError!types.VarId {
    const minus_one = try bld.literal_scalar(types.scalar_literal(tensor.dtype, -1.0));
    const minus_one_full = if (tensor.shape.rank() == 0)
        minus_one
    else
        try bld.broadcast_in_dim(minus_one, tensor.shape.dims, &.{});
    return try bld.multiply(value, minus_one_full);
}
