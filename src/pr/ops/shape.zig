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

        const out_shape = pr.paramOutShape(params) orelse return error.InvalidParams;
        const operand = try ctx.tensorOf(inputs[0]);
        const out = try ctx.tensorOf(outputs[0]);

        if (!std.mem.eql(usize, out.shape.dims, out_shape)) return error.ReshapeTypeMismatch;
        if (operand.dtype != out.dtype) return error.ReshapeTypeMismatch;
        if (numElements(operand.shape.dims) != numElements(out.shape.dims)) return error.ReshapeTypeMismatch;
    }

    pub fn inferOutput(ctx: types.InferContext) pr.BuildError!types.Aval {
        if (ctx.inputs.len != 1) return error.InvalidEqnArity;

        const out_shape = pr.paramOutShape(ctx.params) orelse return error.InvalidParams;
        const operand = try ctx.tensorOf(ctx.inputs[0]);

        if (numElements(operand.shape.dims) != numElements(out_shape)) return error.ReshapeTypeMismatch;
        return .{ .tensor = .{ .dtype = operand.dtype, .shape = .{ .dims = out_shape } } };
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);

        if (inputs.len != 1 or outputs.len != 1) return error.InvalidProgram;

        const operand = ctx.getValue(inputs[0]) orelse return error.InvalidProgram;
        const out_id = outputs[0];
        const out_tensor = try ctx.tensorOf(out_id);
        const out_type = try ctx.tensorToMlirType(out_tensor);

        const op = stablehlo.reshape(ctx.mlir_ctx, operand, out_type, ctx.loc);
        ctx.block.appendOperation(op);
        ctx.setValue(out_id, op.result(0));
    }

    pub fn vjpForward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);

        if (inputs.len != 1) return error.UnsupportedEqn;

        const operand = ctx.getPrimal(inputs[0]) orelse return error.UnsupportedEqn;
        const out_tensor = ctx.tensorOf(outputs[0]);
        const out = try ctx.builder.reshape(operand, out_tensor.shape.dims);
        ctx.setPrimal(outputs[0], out);
    }

    pub fn vjpBackward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);

        if (inputs.len != 1) return error.UnsupportedEqn;

        const out_cot = ctx.getCot(outputs[0]) orelse return;
        const operand_tensor = ctx.tensorOf(inputs[0]);

        // Gradient flows back through inverse reshape
        const contrib = try ctx.builder.reshape(out_cot, operand_tensor.shape.dims);
        try ctx.addCot(inputs[0], contrib);
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

        const perm = pr.paramPermutation(params) orelse return error.InvalidParams;
        const operand = try ctx.tensorOf(inputs[0]);
        const out = try ctx.tensorOf(outputs[0]);

        if (operand.dtype != out.dtype) return error.TransposeTypeMismatch;
        if (!isPermutation(perm, operand.shape.rank())) return error.TransposeTypeMismatch;
        if (out.shape.rank() != operand.shape.rank()) return error.TransposeTypeMismatch;

        for (perm, 0..) |p, out_axis| {
            const in_axis: usize = @intCast(p);
            if (out.shape.dims[out_axis] != operand.shape.dims[in_axis]) return error.TransposeTypeMismatch;
        }
    }

    pub fn inferOutput(ctx: types.InferContext) pr.BuildError!types.Aval {
        if (ctx.inputs.len != 1) return error.InvalidEqnArity;

        const perm = pr.paramPermutation(ctx.params) orelse return error.InvalidParams;
        const operand = try ctx.tensorOf(ctx.inputs[0]);

        if (!isPermutation(perm, operand.shape.rank())) return error.TransposeTypeMismatch;

        const out_dims = try ctx.alloc().alloc(usize, operand.shape.rank());
        for (perm, 0..) |p, i| out_dims[i] = operand.shape.dims[@intCast(p)];
        return .{ .tensor = .{ .dtype = operand.dtype, .shape = .{ .dims = out_dims } } };
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);

        if (inputs.len != 1 or outputs.len != 1) return error.InvalidProgram;

        const perm = pr.paramPermutation(params) orelse return error.InvalidProgram;
        const operand = ctx.getValue(inputs[0]) orelse return error.InvalidProgram;
        const out_id = outputs[0];
        const out_tensor = try ctx.tensorOf(out_id);
        const out_type = try ctx.tensorToMlirType(out_tensor);

        const op = stablehlo.transpose(ctx.mlir_ctx, operand, out_type, ctx.loc, .{ .permutation = perm });
        ctx.block.appendOperation(op);
        ctx.setValue(out_id, op.result(0));
    }

    pub fn vjpForward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);

        if (inputs.len != 1) return error.UnsupportedEqn;

        const operand = ctx.getPrimal(inputs[0]) orelse return error.UnsupportedEqn;
        const perm = pr.paramPermutation(params) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.transpose(operand, perm);
        ctx.setPrimal(outputs[0], out);
    }

    pub fn vjpBackward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);

        if (inputs.len != 1) return error.UnsupportedEqn;

        const out_cot = ctx.getCot(outputs[0]) orelse return;
        const perm = pr.paramPermutation(params) orelse return error.UnsupportedEqn;

        // Inverse permutation
        const inv = try ctx.allocator.alloc(i64, perm.len);
        defer ctx.allocator.free(inv);
        for (perm, 0..) |p, i| inv[@intCast(p)] = @intCast(i);

        const contrib = try ctx.builder.transpose(out_cot, inv);
        try ctx.addCot(inputs[0], contrib);
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

        const out_shape = pr.paramOutShape(params) orelse return error.InvalidParams;
        const bd = pr.paramBroadcastDims(params) orelse return error.InvalidParams;
        const operand = try ctx.tensorOf(inputs[0]);
        const out = try ctx.tensorOf(outputs[0]);

        if (!std.mem.eql(usize, out.shape.dims, out_shape)) return error.BroadcastInDimTypeMismatch;
        try validateBroadcastInDimOp(operand, out, bd);
    }

    pub fn inferOutput(ctx: types.InferContext) pr.BuildError!types.Aval {
        if (ctx.inputs.len != 1) return error.InvalidEqnArity;

        const out_shape = pr.paramOutShape(ctx.params) orelse return error.InvalidParams;
        const bd = pr.paramBroadcastDims(ctx.params) orelse return error.InvalidParams;
        const operand = try ctx.tensorOf(ctx.inputs[0]);
        const out_tensor = types.Tensor{ .dtype = operand.dtype, .shape = .{ .dims = out_shape } };

        try validateBroadcastInDimOp(operand, out_tensor, bd);
        return .{ .tensor = out_tensor };
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);

        if (inputs.len != 1 or outputs.len != 1) return error.InvalidProgram;

        const bd = pr.paramBroadcastDims(params) orelse return error.InvalidProgram;
        const operand = ctx.getValue(inputs[0]) orelse return error.InvalidProgram;
        const out_id = outputs[0];
        const out_tensor = try ctx.tensorOf(out_id);
        const out_type = try ctx.tensorToMlirType(out_tensor);

        const op = stablehlo.broadcast_in_dim(ctx.mlir_ctx, operand, bd, out_type, ctx.loc);
        ctx.block.appendOperation(op);
        ctx.setValue(out_id, op.result(0));
    }

    // No vjpForward/vjpBackward - broadcast_in_dim AD not yet supported
    // (requires reduce_sum to sum over broadcasted dimensions)
};

// ============================================================================
// Helpers
// ============================================================================

fn numElements(dims: []const usize) usize {
    var n: usize = 1;
    for (dims) |d| n *= d;
    return n;
}

fn isPermutation(perm: []const i64, rank: usize) bool {
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

fn validateBroadcastInDimOp(operand: types.Tensor, out: types.Tensor, broadcast_dimensions: []const i64) pr.ValidationError!void {
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
