/// Special Operations
/// Ops with unique semantics (custom_call, etc).
const std = @import("std");
const types = @import("types.zig");
const pr = @import("../pr.zig");
const mlir = @import("../../ffi/mlir/mlir.zig");
const stablehlo = @import("../../ffi/mlir/dialects/stablehlo.zig");

// ============================================================================
// Custom Call
// ============================================================================

pub const custom_call = struct {
    pub const arity = .{ .in = .variadic, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        const inputs = ctx.inputs();
        const outputs = ctx.outputs();
        const params = ctx.params();

        if (outputs.len != 1) return error.InvalidEqnArity;

        _ = pr.paramCallTargetName(params) orelse return error.InvalidParams;
        _ = pr.paramHasSideEffect(params) orelse return error.InvalidParams;
        _ = pr.paramOutAval(params) orelse return error.InvalidParams;

        _ = try ctx.tensorOf(outputs[0]);
        for (inputs) |in_id| _ = try ctx.tensorOf(in_id);
    }

    pub fn inferOutput(ctx: types.InferContext) pr.BuildError!types.Aval {
        const out_aval = pr.paramOutAval(ctx.params) orelse return error.InvalidParams;
        _ = pr.paramCallTargetName(ctx.params) orelse return error.InvalidParams;
        _ = pr.paramHasSideEffect(ctx.params) orelse return error.InvalidParams;
        _ = out_aval.asTensor() orelse return error.CustomCallTypeMismatch;

        for (ctx.inputs) |in_id| _ = try ctx.tensorOf(in_id);
        return out_aval;
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);

        if (outputs.len != 1) return error.InvalidProgram;

        const target = pr.paramCallTargetName(params) orelse return error.InvalidProgram;
        const has_side_effect = pr.paramHasSideEffect(params) orelse return error.InvalidProgram;

        const out_id = outputs[0];
        const out_tensor = try ctx.tensorOf(out_id);
        const out_type = try ctx.tensorToMlirType(out_tensor);

        // Build operand values and layouts
        const operand_values = try ctx.arena.alloc(mlir.Value, inputs.len);
        const operand_layouts = try ctx.arena.alloc([]const usize, inputs.len);

        for (inputs, 0..) |operand_id, i| {
            operand_values[i] = ctx.getValue(operand_id) orelse return error.InvalidProgram;
            const operand_tensor = try ctx.tensorOf(operand_id);
            operand_layouts[i] = try defaultLayout(ctx.arena, operand_tensor.shape.rank());
        }

        const result_layout = try defaultLayout(ctx.arena, out_tensor.shape.rank());

        // Need null-terminated string for call_target_name
        const target_z = try ctx.arena.allocSentinel(u8, target.len, 0);
        @memcpy(target_z, target);

        const op = stablehlo.custom_call(ctx.mlir_ctx, operand_values, .{
            .call_target_name = target_z,
            .has_side_effect = has_side_effect,
            .backend_config = mlir.Attribute.dict(ctx.mlir_ctx, &.{}),
            .operand_layouts = operand_layouts,
            .result_layouts = &.{result_layout},
            .api_version = .typed_ffi,
        }, &.{out_type}, ctx.loc);

        ctx.block.appendOperation(op);
        ctx.setValue(out_id, op.result(0));
    }

    // No vjpForward/vjpBackward - custom_call AD not supported
    // (would require user-provided gradient function)

    pub fn format(writer: *types.Writer, ctx: types.FormatContext) types.FormatError!void {
        const params = ctx.params();
        if (pr.paramCallTargetName(params)) |target| {
            try writer.print("target=\"{s}\"", .{target});
        }
        if (pr.paramHasSideEffect(params)) |se| {
            if (se) try writer.writeAll(", side_effect=true");
        }
    }
};

// ============================================================================
// Helpers
// ============================================================================

fn defaultLayout(arena: std.mem.Allocator, rank: usize) ![]const usize {
    const layout = try arena.alloc(usize, rank);
    for (0..rank) |i| {
        layout[i] = rank - i - 1;
    }
    return layout;
}
