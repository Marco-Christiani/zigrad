const std = @import("std");
const types = @import("types.zig");
const pr = @import("../pr.zig");
const mlir = @import("../../ffi/mlir/mlir.zig");
const stablehlo = @import("../../ffi/mlir/dialects/stablehlo.zig");

fn format_compare(writer: *types.Writer, params: pr.CompareParams) types.FormatError!void {
    try writer.print("dir={s} type={s}", .{ @tagName(params.direction), @tagName(params.compare_type) });
}

fn broadcast_scalar_like(bld: *pr.FunctionBuilder, tensor: pr.Tensor, value: f64) pr.BuildError!pr.VarId {
    const lit = try bld.literal_scalar(types.scalar_literal(tensor.dtype, value));
    return bld.broadcast_in_dim(lit, tensor.shape.dims, &.{});
}

fn map_direction(dir: pr.CompareDirection) stablehlo.ComparisonDirection.Direction {
    return switch (dir) {
        .EQ => .EQ,
        .NE => .NE,
        .GE => .GE,
        .GT => .GT,
        .LE => .LE,
        .LT => .LT,
    };
}

fn map_compare_type(ctype: pr.CompareType) stablehlo.CompareType.Type {
    return switch (ctype) {
        .SIGNED => .SIGNED,
        .UNSIGNED => .UNSIGNED,
        .FLOAT => .FLOAT,
        .TOTALORDER => .TOTALORDER,
    };
}

// =========================================================================
// Compare
// =========================================================================

pub const compare = struct {
    pub const arity = .{ .in = 2, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        const inputs = ctx.inputs();
        const outputs = ctx.outputs();
        const params = ctx.params();
        if (inputs.len != 2 or outputs.len != 1) return error.InvalidEqnArity;

        const cparams = pr.param_compare(params) orelse return error.InvalidParams;
        const lhs = try ctx.tensor_of(inputs[0]);
        const rhs = try ctx.tensor_of(inputs[1]);
        const out = try ctx.tensor_of(outputs[0]);

        if (!types.same_tensor_type(lhs, rhs)) return error.CompareTypeMismatch;
        if (out.dtype != .bool) return error.CompareTypeMismatch;
        if (!std.mem.eql(usize, lhs.shape.dims, out.shape.dims)) return error.CompareTypeMismatch;

        switch (lhs.dtype) {
            .bf16, .f32, .f64 => {
                if (cparams.compare_type != .FLOAT and cparams.compare_type != .TOTALORDER) return error.CompareTypeMismatch;
            },
            .i32, .i64 => if (cparams.compare_type != .SIGNED) return error.CompareTypeMismatch,
            .u32, .u64 => if (cparams.compare_type != .UNSIGNED) return error.CompareTypeMismatch,
            .bool => if (cparams.compare_type != .UNSIGNED) return error.CompareTypeMismatch,
        }
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        if (ctx.inputs.len != 2) return error.InvalidEqnArity;
        const cparams = pr.param_compare(ctx.params) orelse return error.InvalidParams;
        const lhs = try ctx.tensor_of(ctx.inputs[0]);
        const rhs = try ctx.tensor_of(ctx.inputs[1]);
        if (!types.same_tensor_type(lhs, rhs)) return error.CompareTypeMismatch;

        switch (lhs.dtype) {
            .bf16, .f32, .f64 => {
                if (cparams.compare_type != .FLOAT and cparams.compare_type != .TOTALORDER) return error.CompareTypeMismatch;
            },
            .i32, .i64 => if (cparams.compare_type != .SIGNED) return error.CompareTypeMismatch,
            .u32, .u64 => if (cparams.compare_type != .UNSIGNED) return error.CompareTypeMismatch,
            .bool => if (cparams.compare_type != .UNSIGNED) return error.CompareTypeMismatch,
        }
        return .{ .tensor = .{ .dtype = .bool, .shape = lhs.shape } };
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);
        if (inputs.len != 2 or outputs.len != 1) return error.InvalidProgram;

        const cparams = pr.param_compare(params) orelse return error.InvalidProgram;
        const lhs = ctx.get_value(inputs[0]) orelse return error.InvalidProgram;
        const rhs = ctx.get_value(inputs[1]) orelse return error.InvalidProgram;
        const op = stablehlo.compare(
            ctx.mlir_ctx,
            lhs,
            rhs,
            stablehlo.ComparisonDirection.init(ctx.mlir_ctx, map_direction(cparams.direction)),
            stablehlo.CompareType.init(ctx.mlir_ctx, map_compare_type(cparams.compare_type)),
            ctx.loc,
        );
        ctx.block.append_operation(op);
        ctx.set_value(outputs[0], op.result(0));
    }

    pub fn vjp_forward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);
        if (inputs.len != 2) return error.UnsupportedEqn;
        const cparams = pr.param_compare(params) orelse return error.UnsupportedEqn;

        const lhs = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const rhs = ctx.get_primal(inputs[1]) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.compare(lhs, rhs, cparams);
        ctx.set_primal(outputs[0], out);
    }

    pub const format = struct {
        pub fn call(writer: *types.Writer, ctx: types.FormatContext) types.FormatError!void {
            if (pr.param_compare(ctx.params())) |params| {
                try format_compare(writer, params);
            }
        }
    }.call;
};

// =========================================================================
// Select
// =========================================================================

pub const select = struct {
    pub const arity = .{ .in = 3, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        const inputs = ctx.inputs();
        const outputs = ctx.outputs();
        if (inputs.len != 3 or outputs.len != 1) return error.InvalidEqnArity;

        const cond = try ctx.tensor_of(inputs[0]);
        const on_true = try ctx.tensor_of(inputs[1]);
        const on_false = try ctx.tensor_of(inputs[2]);
        const out = try ctx.tensor_of(outputs[0]);

        if (cond.dtype != .bool) return error.SelectTypeMismatch;
        if (!types.same_tensor_type(on_true, on_false)) return error.SelectTypeMismatch;
        if (!types.same_tensor_type(on_true, out)) return error.SelectTypeMismatch;
        if (!std.mem.eql(usize, cond.shape.dims, out.shape.dims)) return error.SelectTypeMismatch;
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        if (ctx.inputs.len != 3) return error.InvalidEqnArity;
        const cond = try ctx.tensor_of(ctx.inputs[0]);
        if (cond.dtype != .bool) return error.SelectTypeMismatch;
        const on_true = try ctx.tensor_of(ctx.inputs[1]);
        const on_false = try ctx.tensor_of(ctx.inputs[2]);
        if (!types.same_tensor_type(on_true, on_false)) return error.SelectTypeMismatch;
        if (!std.mem.eql(usize, cond.shape.dims, on_true.shape.dims)) return error.SelectTypeMismatch;
        return .{ .tensor = on_true };
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 3 or outputs.len != 1) return error.InvalidProgram;

        const cond = ctx.get_value(inputs[0]) orelse return error.InvalidProgram;
        const on_true = ctx.get_value(inputs[1]) orelse return error.InvalidProgram;
        const on_false = ctx.get_value(inputs[2]) orelse return error.InvalidProgram;
        const op = stablehlo.select(ctx.mlir_ctx, cond, on_true, on_false, ctx.loc);
        ctx.block.append_operation(op);
        ctx.set_value(outputs[0], op.result(0));
    }

    pub fn vjp_forward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 3) return error.UnsupportedEqn;

        const cond = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const on_true = ctx.get_primal(inputs[1]) orelse return error.UnsupportedEqn;
        const on_false = ctx.get_primal(inputs[2]) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.select(cond, on_true, on_false);
        ctx.set_primal(outputs[0], out);
    }

    pub fn vjp_backward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        if (inputs.len != 3) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(outputs[0]) orelse return;
        const cond = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const on_true_tensor = ctx.tensor_of(inputs[1]);

        const zeros = try broadcast_scalar_like(ctx.builder, on_true_tensor, 0.0);
        const true_contrib = try ctx.builder.select(cond, out_cot, zeros);
        const false_contrib = try ctx.builder.select(cond, zeros, out_cot);

        try ctx.add_cot(inputs[1], true_contrib);
        try ctx.add_cot(inputs[2], false_contrib);
    }
};
