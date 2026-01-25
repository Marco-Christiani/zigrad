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
// Reduce Sum
// ============================================================================

pub const reduce_sum = struct {
    pub const arity = .{ .in = 1, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        const inputs = ctx.inputs();
        const outputs = ctx.outputs();
        const params = ctx.params();

        if (inputs.len != 1 or outputs.len != 1) return error.InvalidEqnArity;

        const axes = pr.param_reduce_axes(params) orelse return error.InvalidParams;
        const operand = try ctx.tensor_of(inputs[0]);
        const out = try ctx.tensor_of(outputs[0]);

        if (!reduce_sum_matches(operand.shape.dims, out.shape.dims, axes)) {
            return error.ReduceSumTypeMismatch;
        }
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        if (ctx.inputs.len != 1) return error.InvalidEqnArity;
        const axes = pr.param_reduce_axes(ctx.params) orelse return error.InvalidParams;
        const operand = try ctx.tensor_of(ctx.inputs[0]);
        const out_dims = try reduce_sum_output_dims(ctx.alloc(), operand.shape.dims, axes);
        return .{ .tensor = .{ .dtype = operand.dtype, .shape = .{ .dims = out_dims } } };
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);

        if (inputs.len != 1 or outputs.len != 1) return error.InvalidProgram;

        const axes = pr.param_reduce_axes(params) orelse return error.InvalidProgram;
        const operand = ctx.get_value(inputs[0]) orelse return error.InvalidProgram;
        const out_id = outputs[0];
        const out_tensor = try ctx.tensor_of(out_id);

        const elem_type = types.dtype_to_dense_elements_type(out_tensor.dtype);
        const zero_bytes = scalar_zero_bytes(out_tensor.dtype);
        const zero_op = stablehlo.constant(ctx.mlir_ctx, &.{}, elem_type, zero_bytes, ctx.loc);
        ctx.block.append_operation(zero_op);

        const op = stablehlo.reduce(
            ctx.mlir_ctx,
            &.{operand},
            &.{zero_op.result(0)},
            axes,
            {},
            reduce_add_block,
            ctx.loc,
        );
        ctx.block.append_operation(op);
        ctx.set_value(out_id, op.result(0));
    }

    pub fn vjp_forward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);
        if (inputs.len != 1) return error.UnsupportedEqn;

        const axes = pr.param_reduce_axes(params) orelse return error.UnsupportedEqn;
        const operand = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.reduce_sum(operand, axes);
        ctx.set_primal(outputs[0], out);
    }

    pub fn vjp_backward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);
        if (inputs.len != 1) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(outputs[0]) orelse return;
        const axes = pr.param_reduce_axes(params) orelse return error.UnsupportedEqn;
        const in_tensor = ctx.tensor_of(inputs[0]);

        const bd = try reduce_sum_broadcast_dims(ctx.allocator, in_tensor.shape.dims.len, axes);
        defer ctx.allocator.free(bd);
        const contrib = try ctx.builder.broadcast_in_dim(out_cot, in_tensor.shape.dims, bd);
        try ctx.add_cot(inputs[0], contrib);
    }

    pub fn format(writer: *types.Writer, ctx: types.FormatContext) types.FormatError!void {
        if (pr.param_reduce_axes(ctx.params())) |axes| {
            try writer.writeAll("axes=[");
            for (axes, 0..) |d, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("{d}", .{d});
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

    pub fn vjp_forward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);
        if (inputs.len != 1) return error.UnsupportedEqn;

        const out_shape = pr.param_out_shape(params) orelse return error.UnsupportedEqn;
        const bd = pr.param_broadcast_dims(params) orelse return error.UnsupportedEqn;
        const operand = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.broadcast_in_dim(operand, out_shape, bd);
        ctx.set_primal(outputs[0], out);
    }

    pub fn vjp_backward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);
        if (inputs.len != 1) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(outputs[0]) orelse return;
        const bd = pr.param_broadcast_dims(params) orelse return error.UnsupportedEqn;
        const in_tensor = ctx.tensor_of(inputs[0]);
        const out_tensor = ctx.tensor_of(outputs[0]);

        const reduce_axes = try broadcast_reduce_axes(ctx.allocator, in_tensor, out_tensor, bd);
        defer ctx.allocator.free(reduce_axes);

        var contrib = out_cot;
        if (reduce_axes.len > 0) {
            contrib = try ctx.builder.reduce_sum(contrib, reduce_axes);
        }

        const reduced_dims = try reduce_sum_output_dims(ctx.allocator, out_tensor.shape.dims, reduce_axes);
        defer ctx.allocator.free(reduced_dims);
        if (!std.mem.eql(usize, reduced_dims, in_tensor.shape.dims)) {
            contrib = try ctx.builder.reshape(contrib, in_tensor.shape.dims);
        }

        try ctx.add_cot(inputs[0], contrib);
    }

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

fn reduce_sum_output_dims(allocator: std.mem.Allocator, in_dims: []const usize, axes: []const i64) pr.BuildError![]const usize {
    const rank = in_dims.len;
    const max_rank: usize = 64;
    if (rank > max_rank) return error.ReduceSumTypeMismatch;

    var reduce = [_]bool{false} ** max_rank;
    for (axes) |axis| {
        if (axis < 0) return error.ReduceSumTypeMismatch;
        const idx: usize = @intCast(axis);
        if (idx >= rank) return error.ReduceSumTypeMismatch;
        if (reduce[idx]) return error.ReduceSumTypeMismatch;
        reduce[idx] = true;
    }

    var out_count: usize = 0;
    for (0..rank) |i| {
        if (!reduce[i]) out_count += 1;
    }
    const out_dims = try allocator.alloc(usize, out_count);
    var out_i: usize = 0;
    for (0..rank) |i| {
        if (reduce[i]) continue;
        out_dims[out_i] = in_dims[i];
        out_i += 1;
    }
    return out_dims;
}

fn reduce_sum_matches(in_dims: []const usize, out_dims: []const usize, axes: []const i64) bool {
    const rank = in_dims.len;
    const max_rank: usize = 64;
    if (rank > max_rank) return false;

    var reduce = [_]bool{false} ** max_rank;
    for (axes) |axis| {
        if (axis < 0) return false;
        const idx: usize = @intCast(axis);
        if (idx >= rank) return false;
        if (reduce[idx]) return false;
        reduce[idx] = true;
    }

    var out_i: usize = 0;
    for (0..rank) |i| {
        if (reduce[i]) continue;
        if (out_i >= out_dims.len) return false;
        if (in_dims[i] != out_dims[out_i]) return false;
        out_i += 1;
    }

    return out_i == out_dims.len;
}

fn reduce_sum_broadcast_dims(allocator: std.mem.Allocator, rank: usize, axes: []const i64) pr.BuildError![]const i64 {
    const max_rank: usize = 64;
    if (rank > max_rank) return error.ReduceSumTypeMismatch;

    var reduce = [_]bool{false} ** max_rank;
    for (axes) |axis| {
        const idx: usize = @intCast(axis);
        reduce[idx] = true;
    }

    const keep = rank - axes.len;
    const bd = try allocator.alloc(i64, keep);
    var out_i: usize = 0;
    for (0..rank) |i| {
        if (reduce[i]) continue;
        bd[out_i] = @intCast(i);
        out_i += 1;
    }
    return bd;
}

fn broadcast_reduce_axes(
    allocator: std.mem.Allocator,
    in_tensor: types.Tensor,
    out_tensor: types.Tensor,
    bd: []const i64,
) pr.BuildError![]const i64 {
    const max_rank: usize = 64;
    if (out_tensor.shape.rank() > max_rank) return error.ReduceSumTypeMismatch;

    var mapped = [_]bool{false} ** max_rank;
    var reduce = [_]bool{false} ** max_rank;

    for (bd, 0..) |d, i| {
        const out_idx: usize = @intCast(d);
        mapped[out_idx] = true;
        const in_dim = in_tensor.shape.dims[i];
        const out_dim = out_tensor.shape.dims[out_idx];
        if (in_dim == 1 and out_dim > 1) {
            reduce[out_idx] = true;
        }
    }

    for (0..out_tensor.shape.rank()) |i| {
        if (!mapped[i]) {
            reduce[i] = true;
        }
    }

    var count: usize = 0;
    for (0..out_tensor.shape.rank()) |i| {
        if (reduce[i]) count += 1;
    }

    const axes = try allocator.alloc(i64, count);
    var idx: usize = 0;
    for (0..out_tensor.shape.rank()) |i| {
        if (!reduce[i]) continue;
        axes[idx] = @intCast(i);
        idx += 1;
    }
    return axes;
}

fn scalar_zero_bytes(dtype: pr.DType) []const u8 {
    return switch (dtype) {
        .f32 => std.mem.asBytes(&@as(f32, 0.0)),
        .f64 => std.mem.asBytes(&@as(f64, 0.0)),
        .i32 => std.mem.asBytes(&@as(i32, 0)),
        .i64 => std.mem.asBytes(&@as(i64, 0)),
        .u32 => std.mem.asBytes(&@as(u32, 0)),
        .u64 => std.mem.asBytes(&@as(u64, 0)),
    };
}

fn reduce_add_block(_: anytype, ctx: mlir.Context, inputs: []const mlir.Value, accs: []const mlir.Value) mlir.Operation {
    return stablehlo.add(ctx, inputs[0], accs[0], mlir.Location.unknown(ctx));
}
