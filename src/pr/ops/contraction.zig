/// Contraction Operations
/// Ops that contract dimensions (matrix multiply, convolution).
const std = @import("std");
const types = @import("types.zig");
const pr = @import("../pr.zig");
const mlir = @import("../../ffi/mlir/mlir.zig");
const stablehlo = @import("../../ffi/mlir/dialects/stablehlo.zig");

// ============================================================================
// Dot (Matrix Multiply)
// ============================================================================

pub const dot = struct {
    pub const arity = .{ .in = 2, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        const inputs = ctx.inputs();
        const outputs = ctx.outputs();

        if (inputs.len != 2 or outputs.len != 1) return error.InvalidEqnArity;

        const lhs = try ctx.tensorOf(inputs[0]);
        const rhs = try ctx.tensorOf(inputs[1]);
        const out = try ctx.tensorOf(outputs[0]);

        if (lhs.dtype != rhs.dtype or lhs.dtype != out.dtype) return error.DotTypeMismatch;
        if (lhs.shape.rank() != 2 or rhs.shape.rank() != 2 or out.shape.rank() != 2) return error.DotTypeMismatch;
        if (lhs.shape.dims[1] != rhs.shape.dims[0]) return error.DotTypeMismatch;
        if (out.shape.dims[0] != lhs.shape.dims[0] or out.shape.dims[1] != rhs.shape.dims[1]) return error.DotTypeMismatch;
    }

    pub fn inferOutput(ctx: types.InferContext) pr.BuildError!types.Aval {
        if (ctx.inputs.len != 2) return error.InvalidEqnArity;

        const lhs = try ctx.tensorOf(ctx.inputs[0]);
        const rhs = try ctx.tensorOf(ctx.inputs[1]);

        if (lhs.dtype != rhs.dtype) return error.DotTypeMismatch;
        if (lhs.shape.rank() != 2 or rhs.shape.rank() != 2) return error.DotTypeMismatch;
        if (lhs.shape.dims[1] != rhs.shape.dims[0]) return error.DotTypeMismatch;

        const out_dims = try ctx.alloc().dupe(usize, &[_]usize{ lhs.shape.dims[0], rhs.shape.dims[1] });
        return .{ .tensor = .{ .dtype = lhs.dtype, .shape = .{ .dims = out_dims } } };
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);

        if (inputs.len != 2 or outputs.len != 1) return error.InvalidProgram;

        const lhs = ctx.getValue(inputs[0]) orelse return error.InvalidProgram;
        const rhs = ctx.getValue(inputs[1]) orelse return error.InvalidProgram;
        const out_id = outputs[0];
        const out_tensor = try ctx.tensorOf(out_id);
        const out_type = try ctx.tensorToMlirType(out_tensor);

        const op = stablehlo.dot_general(ctx.mlir_ctx, lhs, rhs, out_type, ctx.loc, .{
            .lhs_batching_dimensions = &.{},
            .rhs_batching_dimensions = &.{},
            .lhs_contracting_dimensions = &.{1},
            .rhs_contracting_dimensions = &.{0},
            .precision = .fast,
        });
        ctx.block.appendOperation(op);
        ctx.setValue(out_id, op.result(0));
    }

    pub fn vjpForward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);

        if (inputs.len != 2) return error.UnsupportedEqn;

        const lhs = ctx.getPrimal(inputs[0]) orelse return error.UnsupportedEqn;
        const rhs = ctx.getPrimal(inputs[1]) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.dot(lhs, rhs);
        ctx.setPrimal(outputs[0], out);
    }

    pub fn vjpBackward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);

        if (inputs.len != 2) return error.UnsupportedEqn;

        const out_cot = ctx.getCot(outputs[0]) orelse return;
        const lhs_primal = ctx.getPrimal(inputs[0]) orelse return error.UnsupportedEqn;
        const rhs_primal = ctx.getPrimal(inputs[1]) orelse return error.UnsupportedEqn;

        // For C = A @ B:
        // dA = dC @ B^T
        // dB = A^T @ dC
        const rhs_t = try ctx.builder.transpose(rhs_primal, &.{ 1, 0 });
        const lhs_t = try ctx.builder.transpose(lhs_primal, &.{ 1, 0 });

        const lhs_contrib = try ctx.builder.dot(out_cot, rhs_t);
        const rhs_contrib = try ctx.builder.dot(lhs_t, out_cot);

        try ctx.addCot(inputs[0], lhs_contrib);
        try ctx.addCot(inputs[1], rhs_contrib);
    }
};
