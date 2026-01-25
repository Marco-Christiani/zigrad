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

        _ = pr.param_call_target_name(params) orelse return error.InvalidParams;
        _ = pr.param_has_side_effect(params) orelse return error.InvalidParams;
        _ = pr.param_out_aval(params) orelse return error.InvalidParams;

        _ = try ctx.tensor_of(outputs[0]);
        for (inputs) |in_id| _ = try ctx.tensor_of(in_id);
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        const out_aval = pr.param_out_aval(ctx.params) orelse return error.InvalidParams;
        _ = pr.param_call_target_name(ctx.params) orelse return error.InvalidParams;
        _ = pr.param_has_side_effect(ctx.params) orelse return error.InvalidParams;
        _ = out_aval.as_tensor() orelse return error.CustomCallTypeMismatch;

        for (ctx.inputs) |in_id| _ = try ctx.tensor_of(in_id);
        return out_aval;
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);

        if (outputs.len != 1) return error.InvalidProgram;

        const target = pr.param_call_target_name(params) orelse return error.InvalidProgram;
        const has_side_effect = pr.param_has_side_effect(params) orelse return error.InvalidProgram;

        const out_id = outputs[0];
        const out_tensor = try ctx.tensor_of(out_id);
        const out_type = try ctx.tensor_to_mlir_type(out_tensor);

        // Build operand values and layouts
        const operand_values = try ctx.arena.alloc(mlir.Value, inputs.len);
        const operand_layouts = try ctx.arena.alloc([]const usize, inputs.len);

        for (inputs, 0..) |operand_id, i| {
            operand_values[i] = ctx.get_value(operand_id) orelse return error.InvalidProgram;
            const operand_tensor = try ctx.tensor_of(operand_id);
            operand_layouts[i] = try default_layout(ctx.arena, operand_tensor.shape.rank());
        }

        const result_layout = try default_layout(ctx.arena, out_tensor.shape.rank());

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

        ctx.block.append_operation(op);
        ctx.set_value(out_id, op.result(0));
    }

    // No vjp_forward/vjp_backward - custom_call AD not supported
    // (would require user-provided gradient function)

    pub fn format(writer: *types.Writer, ctx: types.FormatContext) types.FormatError!void {
        const params = ctx.params();
        if (pr.param_call_target_name(params)) |target| {
            try writer.print("target=\"{s}\"", .{target});
        }
        if (pr.param_has_side_effect(params)) |se| {
            if (se) try writer.writeAll(", side_effect=true");
        }
    }
};

// ============================================================================
// Call
// ============================================================================

pub const call = struct {
    pub const arity = .{ .in = .variadic, .out = .variadic };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        const inputs = ctx.inputs();
        const outputs = ctx.outputs();
        const params = ctx.params();

        _ = pr.param_call_callee(params) orelse return error.InvalidParams;

        for (inputs) |in_id| _ = try ctx.tensor_of(in_id);
        for (outputs) |out_id| _ = try ctx.tensor_of(out_id);
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        _ = pr.param_call_callee(ctx.params) orelse return error.InvalidParams;
        return error.InvalidEqnArity;
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);

        const callee = pr.param_call_callee(params) orelse return error.InvalidProgram;

        const operand_values = try ctx.arena.alloc(mlir.Value, inputs.len);
        for (inputs, 0..) |operand_id, i| {
            operand_values[i] = ctx.get_value(operand_id) orelse return error.InvalidProgram;
        }

        const result_types = try ctx.arena.alloc(mlir.Type, outputs.len);
        for (outputs, 0..) |out_id, i| {
            const out_tensor = try ctx.tensor_of(out_id);
            result_types[i] = try ctx.tensor_to_mlir_type(out_tensor);
        }

        const callee_z = try ctx.arena.allocSentinel(u8, callee.len, 0);
        @memcpy(callee_z, callee);

        const op = mlir.Operation.make(ctx.mlir_ctx, "func.call", .{
            .results = result_types,
            .operands = operand_values,
            .attributes = &.{
                .{ "callee", mlir.Attribute.symbol(ctx.mlir_ctx, callee_z) },
            },
            .verify = false,
            .location = ctx.loc,
        });

        ctx.block.append_operation(op);
        for (outputs, 0..) |out_id, i| {
            ctx.set_value(out_id, op.result(i));
        }
    }

    pub fn format(writer: *types.Writer, ctx: types.FormatContext) types.FormatError!void {
        const params = ctx.params();
        if (pr.param_call_callee(params)) |callee| {
            try writer.print("callee=\"{s}\"", .{callee});
        }
    }
};

// ============================================================================
// Helpers
// ============================================================================

fn default_layout(arena: std.mem.Allocator, rank: usize) ![]const usize {
    const layout = try arena.alloc(usize, rank);
    for (0..rank) |i| {
        layout[i] = rank - i - 1;
    }
    return layout;
}
