/// Shape Operations
/// Ops that manipulate tensor shape without changing element values.
const std = @import("std");
const types = @import("types.zig");
const pr = @import("../pr.zig");
const mlir = @import("../../ffi/mlir/mlir.zig");
const stablehlo = @import("../../ffi/mlir/dialects/stablehlo.zig");

// ============================================================================
// Reshape
// ============================================================================

pub const reshape = struct {
    pub const arity = .{ .in = 1, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        const inputs = ctx.inputs();
        const outputs = ctx.outputs();
        const params = ctx.params();

        if (inputs.len != 1 or outputs.len != 1) return error.InvalidEqnArity;

        const out_shape = pr.param_out_shape(params) orelse return error.InvalidParams;
        const operand = try ctx.tensor_of(inputs[0]);
        const out = try ctx.tensor_of(outputs[0]);

        if (!std.mem.eql(usize, out.shape.dims, out_shape)) return error.ReshapeTypeMismatch;
        if (operand.dtype != out.dtype) return error.ReshapeTypeMismatch;
        if (num_elements(operand.shape.dims) != num_elements(out.shape.dims)) return error.ReshapeTypeMismatch;
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        if (ctx.inputs.len != 1) return error.InvalidEqnArity;

        const out_shape = pr.param_out_shape(ctx.params) orelse return error.InvalidParams;
        const operand = try ctx.tensor_of(ctx.inputs[0]);

        if (num_elements(operand.shape.dims) != num_elements(out_shape)) return error.ReshapeTypeMismatch;
        return .{ .tensor = .{ .dtype = operand.dtype, .shape = .{ .dims = out_shape } } };
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);

        if (inputs.len != 1 or outputs.len != 1) return error.InvalidProgram;

        const operand = ctx.get_value(inputs[0]) orelse return error.InvalidProgram;
        const out_id = outputs[0];
        const out_tensor = try ctx.tensor_of(out_id);
        const out_type = try ctx.tensor_to_mlir_type(out_tensor);

        const op = stablehlo.reshape(ctx.mlir_ctx, operand, out_type, ctx.loc);
        ctx.block.append_operation(op);
        ctx.set_value(out_id, op.result(0));
    }

    pub fn vjp_forward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);

        if (inputs.len != 1) return error.UnsupportedEqn;

        const operand = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const out_tensor = ctx.tensor_of(outputs[0]);
        const out = try ctx.builder.reshape(operand, out_tensor.shape.dims);
        ctx.set_primal(outputs[0], out);
    }

    pub fn vjp_backward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);

        if (inputs.len != 1) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(outputs[0]) orelse return;
        const operand_tensor = ctx.tensor_of(inputs[0]);

        // Gradient flows back through inverse reshape
        const contrib = try ctx.builder.reshape(out_cot, operand_tensor.shape.dims);
        try ctx.add_cot(inputs[0], contrib);
    }

    pub fn format(writer: *types.Writer, ctx: types.FormatContext) types.FormatError!void {
        const src = ctx.input_tensor(0) orelse return;
        try format_shape(writer, src.shape.dims);
        try writer.writeAll(" -> ");
        if (pr.param_out_shape(ctx.params())) |out_shape| {
            try format_shape(writer, out_shape);
        }
    }
};

// ============================================================================
// Transpose
// ============================================================================

pub const transpose = struct {
    pub const arity = .{ .in = 1, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        const inputs = ctx.inputs();
        const outputs = ctx.outputs();
        const params = ctx.params();

        if (inputs.len != 1 or outputs.len != 1) return error.InvalidEqnArity;

        const perm = pr.param_permutation(params) orelse return error.InvalidParams;
        const operand = try ctx.tensor_of(inputs[0]);
        const out = try ctx.tensor_of(outputs[0]);

        if (operand.dtype != out.dtype) return error.TransposeTypeMismatch;
        if (!is_permutation(perm, operand.shape.rank())) return error.TransposeTypeMismatch;
        if (out.shape.rank() != operand.shape.rank()) return error.TransposeTypeMismatch;

        for (perm, 0..) |p, out_axis| {
            const in_axis: usize = @intCast(p);
            if (out.shape.dims[out_axis] != operand.shape.dims[in_axis]) return error.TransposeTypeMismatch;
        }
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        if (ctx.inputs.len != 1) return error.InvalidEqnArity;

        const perm = pr.param_permutation(ctx.params) orelse return error.InvalidParams;
        const operand = try ctx.tensor_of(ctx.inputs[0]);

        if (!is_permutation(perm, operand.shape.rank())) return error.TransposeTypeMismatch;

        const out_dims = try ctx.alloc().alloc(usize, operand.shape.rank());
        for (perm, 0..) |p, i| out_dims[i] = operand.shape.dims[@intCast(p)];
        return .{ .tensor = .{ .dtype = operand.dtype, .shape = .{ .dims = out_dims } } };
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);

        if (inputs.len != 1 or outputs.len != 1) return error.InvalidProgram;

        const perm = pr.param_permutation(params) orelse return error.InvalidProgram;
        const operand = ctx.get_value(inputs[0]) orelse return error.InvalidProgram;
        const out_id = outputs[0];
        const out_tensor = try ctx.tensor_of(out_id);
        const out_type = try ctx.tensor_to_mlir_type(out_tensor);

        const op = stablehlo.transpose(ctx.mlir_ctx, operand, out_type, ctx.loc, .{ .permutation = perm });
        ctx.block.append_operation(op);
        ctx.set_value(out_id, op.result(0));
    }

    pub fn vjp_forward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);

        if (inputs.len != 1) return error.UnsupportedEqn;

        const operand = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const perm = pr.param_permutation(params) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.transpose(operand, perm);
        ctx.set_primal(outputs[0], out);
    }

    pub fn vjp_backward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);

        if (inputs.len != 1) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(outputs[0]) orelse return;
        const perm = pr.param_permutation(params) orelse return error.UnsupportedEqn;

        // Inverse permutation
        const inv = try ctx.allocator.alloc(i64, perm.len);
        defer ctx.allocator.free(inv);
        for (perm, 0..) |p, i| inv[@intCast(p)] = @intCast(i);

        const contrib = try ctx.builder.transpose(out_cot, inv);
        try ctx.add_cot(inputs[0], contrib);
    }

    pub fn format(writer: *types.Writer, ctx: types.FormatContext) types.FormatError!void {
        if (pr.param_permutation(ctx.params())) |perm| {
            try writer.writeAll("perm=[");
            for (perm, 0..) |p, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("{d}", .{p});
            }
            try writer.writeByte(']');
        }
    }
};

// ============================================================================
// Broadcast In Dim
// ============================================================================

pub const broadcast_in_dim = struct {
    pub const arity = .{ .in = 1, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        const inputs = ctx.inputs();
        const outputs = ctx.outputs();
        const params = ctx.params();

        if (inputs.len != 1 or outputs.len != 1) return error.InvalidEqnArity;

        const out_shape = pr.param_out_shape(params) orelse return error.InvalidParams;
        const bd = pr.param_broadcast_dims(params) orelse return error.InvalidParams;
        const operand = try ctx.tensor_of(inputs[0]);
        const out = try ctx.tensor_of(outputs[0]);

        if (!std.mem.eql(usize, out.shape.dims, out_shape)) return error.BroadcastInDimTypeMismatch;
        try validate_broadcast_in_dim_op(operand, out, bd);
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        if (ctx.inputs.len != 1) return error.InvalidEqnArity;

        const out_shape = pr.param_out_shape(ctx.params) orelse return error.InvalidParams;
        const bd = pr.param_broadcast_dims(ctx.params) orelse return error.InvalidParams;
        const operand = try ctx.tensor_of(ctx.inputs[0]);
        const out_tensor = types.Tensor{ .dtype = operand.dtype, .shape = .{ .dims = out_shape } };

        try validate_broadcast_in_dim_op(operand, out_tensor, bd);
        return .{ .tensor = out_tensor };
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);

        if (inputs.len != 1 or outputs.len != 1) return error.InvalidProgram;

        const bd = pr.param_broadcast_dims(params) orelse return error.InvalidProgram;
        const operand = ctx.get_value(inputs[0]) orelse return error.InvalidProgram;
        const out_id = outputs[0];
        const out_tensor = try ctx.tensor_of(out_id);
        const out_type = try ctx.tensor_to_mlir_type(out_tensor);

        const op = stablehlo.broadcast_in_dim(ctx.mlir_ctx, operand, bd, out_type, ctx.loc);
        ctx.block.append_operation(op);
        ctx.set_value(out_id, op.result(0));
    }

    // No vjp_forward/vjp_backward - broadcast_in_dim AD not yet supported
    // (requires reduce_sum to sum over broadcasted dimensions)

    pub fn format(writer: *types.Writer, ctx: types.FormatContext) types.FormatError!void {
        const src = ctx.input_tensor(0) orelse return;
        try format_shape(writer, src.shape.dims);
        try writer.writeAll(" -> ");
        if (pr.param_out_shape(ctx.params())) |out_shape| {
            try format_shape(writer, out_shape);
        }
        if (pr.param_broadcast_dims(ctx.params())) |bd| {
            try writer.writeAll(", dims=[");
            for (bd, 0..) |d, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("{d}", .{d});
            }
            try writer.writeByte(']');
        }
    }
};

// ============================================================================
// Helpers
// ============================================================================

fn num_elements(dims: []const usize) usize {
    var n: usize = 1;
    for (dims) |d| n *= d;
    return n;
}

fn is_permutation(perm: []const i64, rank: usize) bool {
    if (perm.len != rank) return false;
    if (rank == 0) return true;

    const max_rank: usize = 64;
    if (rank > max_rank) return false;
    var seen = [_]bool{false} ** max_rank;

    for (perm) |p| {
        if (p < 0) return false;
        const idx: usize = @intCast(p);
        if (idx >= rank) return false;
        if (seen[idx]) return false;
        seen[idx] = true;
    }
    return true;
}

fn format_shape(writer: *types.Writer, dims: []const usize) types.FormatError!void {
    try writer.writeByte('[');
    for (dims, 0..) |d, i| {
        if (i > 0) try writer.writeAll(", ");
        try writer.print("{d}", .{d});
    }
    try writer.writeByte(']');
}

fn validate_broadcast_in_dim_op(operand: types.Tensor, out: types.Tensor, broadcast_dimensions: []const i64) pr.ValidationError!void {
    if (operand.dtype != out.dtype) return error.BroadcastInDimTypeMismatch;
    if (broadcast_dimensions.len != operand.shape.rank()) return error.BroadcastInDimTypeMismatch;
    if (out.shape.rank() < operand.shape.rank()) return error.BroadcastInDimTypeMismatch;

    const max_rank: usize = 64;
    if (out.shape.rank() > max_rank) return error.BroadcastInDimTypeMismatch;
    var seen = [_]bool{false} ** max_rank;

    for (broadcast_dimensions, 0..) |d, i| {
        if (d < 0) return error.BroadcastInDimTypeMismatch;
        const out_dim_index: usize = @intCast(d);
        if (out_dim_index >= out.shape.rank()) return error.BroadcastInDimTypeMismatch;
        if (seen[out_dim_index]) return error.BroadcastInDimTypeMismatch;
        seen[out_dim_index] = true;

        const in_dim = operand.shape.dims[i];
        const out_dim = out.shape.dims[out_dim_index];
        if (in_dim != 1 and in_dim != out_dim) return error.BroadcastInDimTypeMismatch;
    }
}
