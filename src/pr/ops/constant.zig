/// Constant Operations
/// Ops that produce constant values (no inputs).
const std = @import("std");
const types = @import("types.zig");
const pr = @import("../pr.zig");
const mlir = @import("../../ffi/mlir/mlir.zig");
const stablehlo = @import("../../ffi/mlir/dialects/stablehlo.zig");

// ============================================================================
// Literal
// ============================================================================

pub const literal = struct {
    pub const arity = .{ .in = 0, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        const inputs = ctx.inputs();
        const outputs = ctx.outputs();
        const params = ctx.params();

        if (inputs.len != 0 or outputs.len != 1) return error.InvalidEqnArity;

        const lit = pr.paramLiteral(params) orelse return error.InvalidParams;
        const out = try ctx.tensorOf(outputs[0]);

        if (out.dtype != lit.dtype()) return error.LiteralTypeMismatch;
        if (out.shape.rank() != 0) return error.LiteralTypeMismatch;
    }

    pub fn inferOutput(ctx: types.InferContext) pr.BuildError!types.Aval {
        const lit = pr.paramLiteral(ctx.params) orelse return error.InvalidParams;
        return .{ .tensor = .{ .dtype = lit.dtype(), .shape = .{ .dims = &.{} } } };
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);

        if (inputs.len != 0 or outputs.len != 1) return error.InvalidProgram;

        const out_id = outputs[0];
        const out_tensor = try ctx.tensorOf(out_id);
        if (out_tensor.shape.rank() != 0) return error.InvalidProgram;

        const lit = pr.paramLiteral(params) orelse return error.InvalidProgram;
        const elem_type = types.dtypeToDenseElementsType(out_tensor.dtype);
        const raw_bytes = switch (lit) {
            inline else => |v| std.mem.asBytes(&v),
        };

        const op = stablehlo.constant(ctx.mlir_ctx, &.{}, elem_type, raw_bytes, ctx.loc);
        ctx.block.appendOperation(op);
        ctx.setValue(out_id, op.result(0));
    }

    pub fn vjpForward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);

        const lit = pr.paramLiteral(params) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.literalScalar(lit);
        ctx.setPrimal(outputs[0], out);
    }

    // No vjpBackward needed - constants have zero gradient

    pub fn format(writer: *types.Writer, ctx: types.FormatContext) types.FormatError!void {
        if (pr.paramLiteral(ctx.params())) |lit| {
            switch (lit) {
                // currently exhaustive, but doing this explicitly if we add more later so we cant forget to
                //  add the cases here (compiler should catch non-exhaustive)
                .f32, .f64, .i32, .i64, .u32, .u64 => |v| try writer.print("{d}", .{v}),
            }
        }
    }
};
