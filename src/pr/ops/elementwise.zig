/// Elementwise Operations
/// Binary ops that operate element-by-element on tensors of the same shape.
const std = @import("std");
const types = @import("types.zig");
const pr = @import("../pr.zig");
const mlir = @import("../../ffi/mlir/mlir.zig");
const stablehlo = @import("../../ffi/mlir/dialects/stablehlo.zig");

// Shared implementation for binary elementwise ops

fn validateBinaryElementwise(comptime err: pr.ValidationError, ctx: types.ValidateContext) pr.ValidationError!void {
    const inputs = ctx.inputs();
    const outputs = ctx.outputs();
    if (inputs.len != 2 or outputs.len != 1) return error.InvalidEqnArity;

    const lhs = try ctx.tensorOf(inputs[0]);
    const rhs = try ctx.tensorOf(inputs[1]);
    const out = try ctx.tensorOf(outputs[0]);

    if (!types.sameTensorType(lhs, rhs) or !types.sameTensorType(lhs, out)) return err;
}

fn inferBinaryElementwise(comptime err: pr.ValidationError, ctx: types.InferContext) pr.BuildError!types.Aval {
    if (ctx.inputs.len != 2) return error.InvalidEqnArity;
    const lhs = try ctx.tensorOf(ctx.inputs[0]);
    const rhs = try ctx.tensorOf(ctx.inputs[1]);
    if (!types.sameTensorType(lhs, rhs)) return err;
    return .{ .tensor = lhs };
}

fn lowerBinaryElementwise(
    comptime lowerFn: fn (mlir.Context, mlir.Value, mlir.Value, mlir.Location) mlir.Operation,
    ctx: types.LowerContext,
    eqn: pr.Eqn,
) types.LowerError!void {
    const inputs = ctx.inputs(eqn);
    const outputs = ctx.outputs(eqn);

    const lhs = ctx.getValue(inputs[0]) orelse return error.InvalidProgram;
    const rhs = ctx.getValue(inputs[1]) orelse return error.InvalidProgram;

    const op = lowerFn(ctx.mlir_ctx, lhs, rhs, ctx.loc);
    ctx.block.appendOperation(op);
    ctx.setValue(outputs[0], op.result(0));
}

// AD helpers

fn vjpForwardBinaryElementwise(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
    const inputs = ctx.inputs(eqn);
    const outputs = ctx.outputs(eqn);
    if (inputs.len != 2) return error.UnsupportedEqn;

    const lhs = ctx.getPrimal(inputs[0]) orelse return error.UnsupportedEqn;
    const rhs = ctx.getPrimal(inputs[1]) orelse return error.UnsupportedEqn;
    const out = try ctx.builder.add(lhs, rhs);
    ctx.setPrimal(outputs[0], out);
}

// ============================================================================
// Add
// ============================================================================

pub const add = struct {
    pub const arity = .{ .in = 2, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        return validateBinaryElementwise(error.AddTypeMismatch, ctx);
    }

    pub fn inferOutput(ctx: types.InferContext) pr.BuildError!types.Aval {
        return inferBinaryElementwise(error.AddTypeMismatch, ctx);
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        return lowerBinaryElementwise(stablehlo.add, ctx, eqn);
    }

    pub fn vjpForward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 2) return error.UnsupportedEqn;

        const lhs = ctx.getPrimal(inputs[0]) orelse return error.UnsupportedEqn;
        const rhs = ctx.getPrimal(inputs[1]) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.add(lhs, rhs);
        ctx.setPrimal(outputs[0], out);
    }

    pub fn vjpBackward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 2) return error.UnsupportedEqn;

        const out_cot = ctx.getCot(outputs[0]) orelse return;
        // d/d(lhs) = 1, d/d(rhs) = 1
        try ctx.addCot(inputs[0], out_cot);
        try ctx.addCot(inputs[1], out_cot);
    }
};

// ============================================================================
// Subtract
// ============================================================================

pub const subtract = struct {
    pub const arity = .{ .in = 2, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        return validateBinaryElementwise(error.SubtractTypeMismatch, ctx);
    }

    pub fn inferOutput(ctx: types.InferContext) pr.BuildError!types.Aval {
        return inferBinaryElementwise(error.SubtractTypeMismatch, ctx);
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        return lowerBinaryElementwise(stablehlo.subtract, ctx, eqn);
    }

    pub fn vjpForward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 2) return error.UnsupportedEqn;

        const lhs = ctx.getPrimal(inputs[0]) orelse return error.UnsupportedEqn;
        const rhs = ctx.getPrimal(inputs[1]) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.subtract(lhs, rhs);
        ctx.setPrimal(outputs[0], out);
    }

    pub fn vjpBackward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 2) return error.UnsupportedEqn;

        const out_cot = ctx.getCot(outputs[0]) orelse return;
        const rhs_tensor = ctx.tensorOf(inputs[1]);

        // d/d(lhs) = 1, d/d(rhs) = -1
        try ctx.addCot(inputs[0], out_cot);
        const neg = try negateLike(ctx.builder, rhs_tensor, out_cot);
        try ctx.addCot(inputs[1], neg);
    }
};

// ============================================================================
// Multiply
// ============================================================================

pub const multiply = struct {
    pub const arity = .{ .in = 2, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        return validateBinaryElementwise(error.MultiplyTypeMismatch, ctx);
    }

    pub fn inferOutput(ctx: types.InferContext) pr.BuildError!types.Aval {
        return inferBinaryElementwise(error.MultiplyTypeMismatch, ctx);
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        return lowerBinaryElementwise(stablehlo.multiply, ctx, eqn);
    }

    pub fn vjpForward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 2) return error.UnsupportedEqn;

        const lhs = ctx.getPrimal(inputs[0]) orelse return error.UnsupportedEqn;
        const rhs = ctx.getPrimal(inputs[1]) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.multiply(lhs, rhs);
        ctx.setPrimal(outputs[0], out);
    }

    pub fn vjpBackward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 2) return error.UnsupportedEqn;

        const out_cot = ctx.getCot(outputs[0]) orelse return;
        const lhs_primal = ctx.getPrimal(inputs[0]) orelse return error.UnsupportedEqn;
        const rhs_primal = ctx.getPrimal(inputs[1]) orelse return error.UnsupportedEqn;

        // d/d(lhs) = rhs, d/d(rhs) = lhs
        const lhs_contrib = try ctx.builder.multiply(out_cot, rhs_primal);
        const rhs_contrib = try ctx.builder.multiply(out_cot, lhs_primal);

        try ctx.addCot(inputs[0], lhs_contrib);
        try ctx.addCot(inputs[1], rhs_contrib);
    }
};

// ============================================================================
// Maximum
// ============================================================================

pub const maximum = struct {
    pub const arity = .{ .in = 2, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        return validateBinaryElementwise(error.MaximumTypeMismatch, ctx);
    }

    pub fn inferOutput(ctx: types.InferContext) pr.BuildError!types.Aval {
        return inferBinaryElementwise(error.MaximumTypeMismatch, ctx);
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        return lowerBinaryElementwise(stablehlo.maximum, ctx, eqn);
    }

    // No vjpForward/vjpBackward - maximum AD not yet supported
};

// ============================================================================
// Helpers
// ============================================================================

fn negateLike(bld: *pr.FunctionBuilder, tensor: types.Tensor, value: types.VarId) pr.BuildError!types.VarId {
    const minus_one = try bld.literalScalar(types.scalarLiteral(tensor.dtype, -1.0));
    const minus_one_full = if (tensor.shape.rank() == 0)
        minus_one
    else
        try bld.broadcastInDim(minus_one, tensor.shape.dims, &.{});
    return try bld.multiply(value, minus_one_full);
}
