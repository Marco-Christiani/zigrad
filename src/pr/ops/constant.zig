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

        const lit = pr.param_literal(params) orelse return error.InvalidParams;
        const out = try ctx.tensor_of(outputs[0]);

        if (out.dtype != lit.dtype()) return error.LiteralTypeMismatch;
        if (out.shape.rank() != 0) return error.LiteralTypeMismatch;
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        const lit = pr.param_literal(ctx.params) orelse return error.InvalidParams;
        return .{ .tensor = .{ .dtype = lit.dtype(), .shape = .{ .dims = &.{} } } };
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);

        if (inputs.len != 0 or outputs.len != 1) return error.InvalidProgram;

        const out_id = outputs[0];
        const out_tensor = try ctx.tensor_of(out_id);
        if (out_tensor.shape.rank() != 0) return error.InvalidProgram;

        const lit = pr.param_literal(params) orelse return error.InvalidProgram;
        const elem_type = types.dtype_to_dense_elements_type(out_tensor.dtype);
        const raw_bytes = switch (lit) {
            inline else => |v| std.mem.asBytes(&v),
        };

        const op = stablehlo.constant(ctx.mlir_ctx, &.{}, elem_type, raw_bytes, ctx.loc);
        ctx.block.append_operation(op);
        ctx.set_value(out_id, op.result(0));
    }

    pub fn vjp_forward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);

        const lit = pr.param_literal(params) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.literal_scalar(lit);
        ctx.set_primal(outputs[0], out);
    }

    // No vjp_backward needed - constants have zero gradient

    pub fn format(writer: *types.Writer, ctx: types.FormatContext) types.FormatError!void {
        if (pr.param_literal(ctx.params())) |lit| {
            switch (lit) {
                // currently exhaustive, but doing this explicitly if we add more later so we cant forget to
                //  add the cases here (compiler should catch non-exhaustive)
                inline .f32, .f64, .i32, .i64, .u32, .u64 => |v| try writer.print("{d}", .{v}),
            }
        }
    }
};
