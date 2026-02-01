/// Unary elementwise operations.
const std = @import("std");
const types = @import("types.zig");
const pr = @import("../pr.zig");
const mlir = @import("../../ffi/mlir/mlir.zig");
const stablehlo = @import("../../ffi/mlir/dialects/stablehlo.zig");

fn validate_unary_elementwise(comptime err: pr.ValidationError, ctx: types.ValidateContext) pr.ValidationError!void {
    const inputs = ctx.inputs();
    const outputs = ctx.outputs();
    if (inputs.len != 1 or outputs.len != 1) return error.InvalidEqnArity;

    const operand = try ctx.tensor_of(inputs[0]);
    const out = try ctx.tensor_of(outputs[0]);
    if (!types.same_tensor_type(operand, out)) return err;
}

fn infer_unary_elementwise(comptime _: pr.ValidationError, ctx: types.InferContext) pr.BuildError!types.Aval {
    if (ctx.inputs.len != 1) return error.InvalidEqnArity;
    const operand = try ctx.tensor_of(ctx.inputs[0]);
    return .{ .tensor = operand };
}

fn lower_unary_elementwise(
    comptime lower_fn: fn (mlir.Context, mlir.Value, mlir.Location) mlir.Operation,
    ctx: types.LowerContext,
    eqn: pr.Eqn,
) types.LowerError!void {
    const inputs = ctx.inputs(eqn);
    const outputs = ctx.outputs(eqn);
    if (inputs.len != 1 or outputs.len != 1) return error.InvalidProgram;

    const operand = ctx.get_value(inputs[0]) orelse return error.InvalidProgram;
    const op = lower_fn(ctx.mlir_ctx, operand, ctx.loc);
    ctx.block.append_operation(op);
    ctx.set_value(outputs[0], op.result(0));
}

fn format_unary_elementwise(writer: *types.Writer, ctx: types.FormatContext) types.FormatError!void {
    if (ctx.input_tensor(0)) |t| {
        try writer.print("dtype={s}", .{@tagName(t.dtype)});
    }
}

fn broadcast_scalar_like(bld: *pr.FunctionBuilder, tensor: pr.Tensor, value: f64) pr.BuildError!pr.VarId {
    const lit = try bld.literal_scalar(types.scalar_literal(tensor.dtype, value));
    return bld.broadcast_in_dim(lit, tensor.shape.dims, &.{});
}

// =========================================================================
// Exp
// =========================================================================

pub const exp = struct {
    pub const arity = .{ .in = 1, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        return validate_unary_elementwise(error.ExpTypeMismatch, ctx);
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        return infer_unary_elementwise(error.ExpTypeMismatch, ctx);
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        return lower_unary_elementwise(stablehlo.exponential, ctx, eqn);
    }

    pub fn vjp_forward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 1) return error.UnsupportedEqn;

        const operand = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.exp(operand);
        ctx.set_primal(outputs[0], out);
    }

    pub fn vjp_backward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 1) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(outputs[0]) orelse return;
        const out_primal = ctx.get_primal(outputs[0]) orelse return error.UnsupportedEqn;
        const contrib = try ctx.builder.multiply(out_cot, out_primal);
        try ctx.add_cot(inputs[0], contrib);
    }

    pub const format = format_unary_elementwise;
};

// =========================================================================
// Log
// =========================================================================

pub const log = struct {
    pub const arity = .{ .in = 1, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        return validate_unary_elementwise(error.LogTypeMismatch, ctx);
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        return infer_unary_elementwise(error.LogTypeMismatch, ctx);
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        return lower_unary_elementwise(stablehlo.log, ctx, eqn);
    }

    pub fn vjp_forward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 1) return error.UnsupportedEqn;

        const operand = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.log(operand);
        ctx.set_primal(outputs[0], out);
    }

    pub fn vjp_backward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 1) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(outputs[0]) orelse return;
        const operand = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const contrib = try ctx.builder.divide(out_cot, operand);
        try ctx.add_cot(inputs[0], contrib);
    }

    pub const format = format_unary_elementwise;
};

// =========================================================================
// Convert
// =========================================================================

pub const convert = struct {
    pub const arity = .{ .in = 1, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        const inputs = ctx.inputs();
        const outputs = ctx.outputs();
        const params = ctx.params();
        if (inputs.len != 1 or outputs.len != 1) return error.InvalidEqnArity;
        const out_dtype = pr.param_out_dtype(params) orelse return error.InvalidParams;

        const operand = try ctx.tensor_of(inputs[0]);
        const out = try ctx.tensor_of(outputs[0]);
        if (out.dtype != out_dtype) return error.ConvertTypeMismatch;
        if (!std.mem.eql(usize, operand.shape.dims, out.shape.dims)) return error.ConvertTypeMismatch;
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        if (ctx.inputs.len != 1) return error.InvalidEqnArity;
        const out_dtype = pr.param_out_dtype(ctx.params) orelse return error.InvalidParams;
        const operand = try ctx.tensor_of(ctx.inputs[0]);
        return .{ .tensor = .{ .dtype = out_dtype, .shape = operand.shape } };
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 1 or outputs.len != 1) return error.InvalidProgram;

        const operand = ctx.get_value(inputs[0]) orelse return error.InvalidProgram;
        const out_tensor = try ctx.tensor_of(outputs[0]);
        const out_type = try ctx.tensor_to_mlir_type(out_tensor);
        const op = stablehlo.convert(ctx.mlir_ctx, operand, out_type, ctx.loc);
        ctx.block.append_operation(op);
        ctx.set_value(outputs[0], op.result(0));
    }

    pub fn vjp_forward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);
        if (inputs.len != 1) return error.UnsupportedEqn;
        const out_dtype = pr.param_out_dtype(params) orelse return error.UnsupportedEqn;

        const operand = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.convert(operand, out_dtype);
        ctx.set_primal(outputs[0], out);
    }

    pub fn vjp_backward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 1) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(outputs[0]) orelse return;
        const in_tensor = ctx.tensor_of(inputs[0]);
        const out_tensor = ctx.tensor_of(outputs[0]);
        const cot = if (out_tensor.dtype == in_tensor.dtype)
            out_cot
        else
            try ctx.builder.convert(out_cot, in_tensor.dtype);
        try ctx.add_cot(inputs[0], cot);
    }
};

// =========================================================================
// Rsqrt
// =========================================================================

pub const rsqrt = struct {
    pub const arity = .{ .in = 1, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        return validate_unary_elementwise(error.RsqrtTypeMismatch, ctx);
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        return infer_unary_elementwise(error.RsqrtTypeMismatch, ctx);
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        return lower_unary_elementwise(stablehlo.rsqrt, ctx, eqn);
    }

    pub fn vjp_forward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 1) return error.UnsupportedEqn;

        const operand = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.rsqrt(operand);
        ctx.set_primal(outputs[0], out);
    }

    pub fn vjp_backward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 1) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(outputs[0]) orelse return;
        const out_primal = ctx.get_primal(outputs[0]) orelse return error.UnsupportedEqn;
        const y2 = try ctx.builder.multiply(out_primal, out_primal);
        const y3 = try ctx.builder.multiply(y2, out_primal);

        const tensor = ctx.tensor_of(inputs[0]);
        const neg_half = try broadcast_scalar_like(ctx.builder, tensor, -0.5);
        const scale = try ctx.builder.multiply(y3, neg_half);
        const contrib = try ctx.builder.multiply(out_cot, scale);
        try ctx.add_cot(inputs[0], contrib);
    }

    pub const format = format_unary_elementwise;
};

// =========================================================================
// Logistic (sigmoid)
// =========================================================================

pub const logistic = struct {
    pub const arity = .{ .in = 1, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        return validate_unary_elementwise(error.LogisticTypeMismatch, ctx);
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        return infer_unary_elementwise(error.LogisticTypeMismatch, ctx);
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        return lower_unary_elementwise(stablehlo.logistic, ctx, eqn);
    }

    pub fn vjp_forward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 1) return error.UnsupportedEqn;

        const operand = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.logistic(operand);
        ctx.set_primal(outputs[0], out);
    }

    pub fn vjp_backward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 1) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(outputs[0]) orelse return;
        const out_primal = ctx.get_primal(outputs[0]) orelse return error.UnsupportedEqn;

        const tensor = ctx.tensor_of(inputs[0]);
        const ones = try broadcast_scalar_like(ctx.builder, tensor, 1.0);
        const one_minus = try ctx.builder.subtract(ones, out_primal);
        const slope = try ctx.builder.multiply(out_primal, one_minus);
        const contrib = try ctx.builder.multiply(out_cot, slope);
        try ctx.add_cot(inputs[0], contrib);
    }

    pub const format = format_unary_elementwise;
};
