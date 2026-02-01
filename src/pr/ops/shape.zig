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
// Slice
// ============================================================================

pub const slice = struct {
    pub const arity = .{ .in = 1, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        const inputs = ctx.inputs();
        const outputs = ctx.outputs();
        const params = ctx.params();

        if (inputs.len != 1 or outputs.len != 1) return error.InvalidEqnArity;
        const sparams = pr.param_slice(params) orelse return error.InvalidParams;
        const operand = try ctx.tensor_of(inputs[0]);
        const out = try ctx.tensor_of(outputs[0]);
        if (!slice_matches_local(operand.shape.dims, out.shape.dims, sparams)) return error.SliceTypeMismatch;
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        if (ctx.inputs.len != 1) return error.InvalidEqnArity;
        const sparams = pr.param_slice(ctx.params) orelse return error.InvalidParams;
        const operand = try ctx.tensor_of(ctx.inputs[0]);
        const out_dims = try slice_output_dims_local(ctx.alloc(), operand.shape.dims, sparams);
        return .{ .tensor = .{ .dtype = operand.dtype, .shape = .{ .dims = out_dims } } };
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);

        if (inputs.len != 1 or outputs.len != 1) return error.InvalidProgram;
        const sparams = pr.param_slice(params) orelse return error.InvalidProgram;

        const operand = ctx.get_value(inputs[0]) orelse return error.InvalidProgram;
        const out_tensor = try ctx.tensor_of(outputs[0]);
        const out_type = try ctx.tensor_to_mlir_type(out_tensor);
        const op = stablehlo.slice(
            ctx.mlir_ctx,
            operand,
            sparams.start_indices,
            sparams.limit_indices,
            sparams.strides,
            out_type,
            ctx.loc,
        );
        ctx.block.append_operation(op);
        ctx.set_value(outputs[0], op.result(0));
    }

    pub fn vjp_forward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);
        if (inputs.len != 1) return error.UnsupportedEqn;
        const sparams = pr.param_slice(params) orelse return error.UnsupportedEqn;

        const operand = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.slice(operand, sparams);
        ctx.set_primal(outputs[0], out);
    }

    pub fn vjp_backward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);
        if (inputs.len != 1) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(outputs[0]) orelse return;
        const sparams = pr.param_slice(params) orelse return error.UnsupportedEqn;
        const in_tensor = ctx.tensor_of(inputs[0]);

        for (sparams.strides) |s| {
            if (s != 1) return error.UnsupportedEqn;
        }

        var cur = out_cot;
        var cur_dims = ctx.tensor_of(outputs[0]).shape.dims;

        var axis: usize = 0;
        while (axis < in_tensor.shape.rank()) : (axis += 1) {
            const start = sparams.start_indices[axis];
            const limit = sparams.limit_indices[axis];
            if (start < 0 or limit < 0) return error.UnsupportedEqn;
            const start_u: usize = @intCast(start);
            const limit_u: usize = @intCast(limit);
            const pre = start_u;
            const post = in_tensor.shape.dims[axis] - limit_u;

            if (pre > 0) {
                const pre_dims = try ctx.allocator.dupe(usize, cur_dims);
                defer ctx.allocator.free(pre_dims);
                pre_dims[axis] = pre;
                const pre_tensor = try zeros_like(ctx.builder, in_tensor.dtype, pre_dims);
                const cat = try ctx.builder.concatenate(&.{ pre_tensor, cur }, @intCast(axis));
                cur = cat;
                cur_dims = ctx.builder.avals.items[@intCast(cur)].as_tensor().?.shape.dims;
            }

            if (post > 0) {
                const post_dims = try ctx.allocator.dupe(usize, cur_dims);
                defer ctx.allocator.free(post_dims);
                post_dims[axis] = post;
                const post_tensor = try zeros_like(ctx.builder, in_tensor.dtype, post_dims);
                const cat = try ctx.builder.concatenate(&.{ cur, post_tensor }, @intCast(axis));
                cur = cat;
                cur_dims = ctx.builder.avals.items[@intCast(cur)].as_tensor().?.shape.dims;
            }
        }

        try ctx.add_cot(inputs[0], cur);
    }
};

// ============================================================================
// Concatenate
// ============================================================================

pub const concatenate = struct {
    pub const arity = .{ .in = .any, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        const inputs = ctx.inputs();
        const outputs = ctx.outputs();
        const params = ctx.params();

        if (inputs.len == 0 or outputs.len != 1) return error.InvalidEqnArity;
        const axis = pr.param_concat_axis(params) orelse return error.InvalidParams;

        const first = try ctx.tensor_of(inputs[0]);
        const out = try ctx.tensor_of(outputs[0]);
        if (!concat_matches_local(ctx.func, inputs, out, axis)) return error.ConcatTypeMismatch;
        if (first.dtype != out.dtype) return error.ConcatTypeMismatch;
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        if (ctx.inputs.len == 0) return error.InvalidEqnArity;
        const axis = pr.param_concat_axis(ctx.params) orelse return error.InvalidParams;
        const first = try ctx.tensor_of(ctx.inputs[0]);
        const out_dims = try concat_output_dims_local(ctx, ctx.inputs, axis);
        return .{ .tensor = .{ .dtype = first.dtype, .shape = .{ .dims = out_dims } } };
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);

        if (inputs.len == 0 or outputs.len != 1) return error.InvalidProgram;
        const axis = pr.param_concat_axis(params) orelse return error.InvalidProgram;

        var values = try ctx.arena.alloc(mlir.Value, inputs.len);
        for (inputs, 0..) |id, i| {
            values[i] = ctx.get_value(id) orelse return error.InvalidProgram;
        }

        const op = stablehlo.concatenate(ctx.mlir_ctx, values, axis, ctx.loc);
        ctx.block.append_operation(op);
        ctx.set_value(outputs[0], op.result(0));
    }

    pub fn vjp_forward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);
        if (inputs.len == 0) return error.UnsupportedEqn;
        const axis = pr.param_concat_axis(params) orelse return error.UnsupportedEqn;

        var primals = try ctx.allocator.alloc(pr.VarId, inputs.len);
        defer ctx.allocator.free(primals);
        for (inputs, 0..) |id, i| {
            primals[i] = ctx.get_primal(id) orelse return error.UnsupportedEqn;
        }
        const out = try ctx.builder.concatenate(primals, axis);
        ctx.set_primal(outputs[0], out);
    }

    pub fn vjp_backward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);
        if (inputs.len == 0) return error.UnsupportedEqn;
        const axis = pr.param_concat_axis(params) orelse return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(outputs[0]) orelse return;
        const axis_u: usize = @intCast(axis);

        var offset: usize = 0;
        for (inputs) |id| {
            const t = ctx.tensor_of(id);
            const len = t.shape.dims[axis_u];
            const start = @as(i64, @intCast(offset));
            const limit = @as(i64, @intCast(offset + len));

            var start_indices = try ctx.allocator.alloc(i64, t.shape.rank());
            var limit_indices = try ctx.allocator.alloc(i64, t.shape.rank());
            var strides = try ctx.allocator.alloc(i64, t.shape.rank());
            defer ctx.allocator.free(start_indices);
            defer ctx.allocator.free(limit_indices);
            defer ctx.allocator.free(strides);

            for (t.shape.dims, 0..) |d, i| {
                start_indices[i] = 0;
                limit_indices[i] = @intCast(d);
                strides[i] = 1;
            }
            start_indices[axis_u] = start;
            limit_indices[axis_u] = limit;

            const slice_out = try ctx.builder.slice(out_cot, .{
                .start_indices = start_indices,
                .limit_indices = limit_indices,
                .strides = strides,
            });
            try ctx.add_cot(id, slice_out);
            offset += len;
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
// Reduce Max
// ============================================================================

pub const reduce_max = struct {
    pub const arity = .{ .in = 1, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        const inputs = ctx.inputs();
        const outputs = ctx.outputs();
        const params = ctx.params();

        if (inputs.len != 1 or outputs.len != 1) return error.InvalidEqnArity;
        const axes = pr.param_reduce_axes(params) orelse return error.InvalidParams;
        const operand = try ctx.tensor_of(inputs[0]);
        const out = try ctx.tensor_of(outputs[0]);
        if (!reduce_sum_matches(operand.shape.dims, out.shape.dims, axes)) return error.ReduceMaxTypeMismatch;
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
        const min_bytes = scalar_min_bytes(out_tensor.dtype);
        const min_op = stablehlo.constant(ctx.mlir_ctx, &.{}, elem_type, min_bytes, ctx.loc);
        ctx.block.append_operation(min_op);

        const op = stablehlo.reduce(
            ctx.mlir_ctx,
            &.{operand},
            &.{min_op.result(0)},
            axes,
            {},
            reduce_max_block,
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
        const out = try ctx.builder.reduce_max(operand, axes);
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

        const out_primal = ctx.get_primal(outputs[0]) orelse return error.UnsupportedEqn;
        const max_b = try ctx.builder.broadcast_in_dim(out_primal, in_tensor.shape.dims, bd);
        const out_cot_b = try ctx.builder.broadcast_in_dim(out_cot, in_tensor.shape.dims, bd);

        const cmp_type = compare_type_for_dtype(in_tensor.dtype);
        const operand = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const mask = try ctx.builder.compare(operand, max_b, .{
            .direction = .EQ,
            .compare_type = cmp_type,
        });
        const mask_f = try ctx.builder.convert(mask, in_tensor.dtype);
        const contrib = try ctx.builder.multiply(out_cot_b, mask_f);
        try ctx.add_cot(inputs[0], contrib);
    }
};

// ============================================================================
// Gather
// ============================================================================

pub const gather = struct {
    pub const arity = .{ .in = 2, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        const inputs = ctx.inputs();
        const outputs = ctx.outputs();
        const params = ctx.params();

        if (inputs.len != 2 or outputs.len != 1) return error.InvalidEqnArity;
        const gparams = pr.param_gather(params) orelse return error.InvalidParams;
        const operand = try ctx.tensor_of(inputs[0]);
        const indices = try ctx.tensor_of(inputs[1]);
        const out = try ctx.tensor_of(outputs[0]);

        if (out.dtype != operand.dtype) return error.GatherTypeMismatch;
        if (indices.dtype != .i32 and indices.dtype != .i64) return error.GatherTypeMismatch;
        const expected = gather_output_dims_local(operand.shape.dims, indices.shape.dims, gparams) orelse return error.GatherTypeMismatch;
        if (!std.mem.eql(usize, out.shape.dims, expected)) return error.GatherTypeMismatch;
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        if (ctx.inputs.len != 2) return error.InvalidEqnArity;
        const gparams = pr.param_gather(ctx.params) orelse return error.InvalidParams;
        const operand = try ctx.tensor_of(ctx.inputs[0]);
        const indices = try ctx.tensor_of(ctx.inputs[1]);
        if (indices.dtype != .i32 and indices.dtype != .i64) return error.GatherTypeMismatch;
        const out_dims = try pr.gather_output_dims(ctx.alloc(), operand.shape.dims, indices.shape.dims, gparams);
        return .{ .tensor = .{ .dtype = operand.dtype, .shape = .{ .dims = out_dims } } };
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);

        if (inputs.len != 2 or outputs.len != 1) return error.InvalidProgram;
        const gparams = pr.param_gather(params) orelse return error.InvalidProgram;

        const operand = ctx.get_value(inputs[0]) orelse return error.InvalidProgram;
        const indices = ctx.get_value(inputs[1]) orelse return error.InvalidProgram;
        const op = stablehlo.gather(ctx.mlir_ctx, operand, indices, gparams.slice_sizes, ctx.loc, .{
            .offset_dims = gparams.offset_dims,
            .collapsed_slice_dims = gparams.collapsed_slice_dims,
            .operand_batching_dims = &.{},
            .start_indices_batching_dims = &.{},
            .start_index_map = gparams.start_index_map,
            .index_vector_dim = gparams.index_vector_dim,
        });
        ctx.block.append_operation(op);
        ctx.set_value(outputs[0], op.result(0));
    }

    pub fn vjp_forward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);
        if (inputs.len != 2) return error.UnsupportedEqn;
        const gparams = pr.param_gather(params) orelse return error.UnsupportedEqn;

        const operand = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const indices = ctx.get_primal(inputs[1]) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.gather(operand, indices, gparams);
        ctx.set_primal(outputs[0], out);
    }

    pub fn vjp_backward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);
        if (inputs.len != 2) return error.UnsupportedEqn;
        const gparams = pr.param_gather(params) orelse return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(outputs[0]) orelse return;
        const indices = ctx.get_primal(inputs[1]) orelse return error.UnsupportedEqn;
        const operand_tensor = ctx.tensor_of(inputs[0]);

        const zero = try ctx.builder.literal_scalar(types.scalar_literal(operand_tensor.dtype, 0.0));
        const zero_full = try ctx.builder.broadcast_in_dim(zero, operand_tensor.shape.dims, &.{});

        const sparams = scatter_params_for_gather(gparams) catch return error.UnsupportedEqn;
        const contrib = try ctx.builder.scatter(zero_full, indices, out_cot, sparams);
        try ctx.add_cot(inputs[0], contrib);
    }
};

// ============================================================================
// Scatter Add
// ============================================================================

pub const scatter = struct {
    pub const arity = .{ .in = 3, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        const inputs = ctx.inputs();
        const outputs = ctx.outputs();
        const params = ctx.params();

        if (inputs.len != 3 or outputs.len != 1) return error.InvalidEqnArity;
        _ = pr.param_scatter(params) orelse return error.InvalidParams;
        const operand = try ctx.tensor_of(inputs[0]);
        const indices = try ctx.tensor_of(inputs[1]);
        const updates = try ctx.tensor_of(inputs[2]);
        const out = try ctx.tensor_of(outputs[0]);

        if (!types.same_tensor_type(operand, out)) return error.ScatterAddTypeMismatch;
        if (indices.dtype != .i32 and indices.dtype != .i64) return error.ScatterTypeMismatch;
        if (updates.dtype != operand.dtype) return error.ScatterTypeMismatch;
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        if (ctx.inputs.len != 3) return error.InvalidEqnArity;
        _ = pr.param_scatter(ctx.params) orelse return error.InvalidParams;
        const operand = try ctx.tensor_of(ctx.inputs[0]);
        const indices = try ctx.tensor_of(ctx.inputs[1]);
        if (indices.dtype != .i32 and indices.dtype != .i64) return error.ScatterTypeMismatch;
        const updates = try ctx.tensor_of(ctx.inputs[2]);
        if (updates.dtype != operand.dtype) return error.ScatterTypeMismatch;
        return .{ .tensor = operand };
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);

        if (inputs.len != 3 or outputs.len != 1) return error.InvalidProgram;
        const sparams = pr.param_scatter(params) orelse return error.InvalidProgram;

        const operand = ctx.get_value(inputs[0]) orelse return error.InvalidProgram;
        const indices = ctx.get_value(inputs[1]) orelse return error.InvalidProgram;
        const updates = ctx.get_value(inputs[2]) orelse return error.InvalidProgram;

        const update_block = make_update_block(ctx.mlir_ctx, operand.get_type(), ctx.loc, sparams.reduction);
        const op = stablehlo.scatter(
            ctx.mlir_ctx,
            &.{operand},
            &.{indices},
            &.{updates},
            update_block,
            .{
                .update_window_dims = sparams.update_window_dims,
                .inserted_window_dims = sparams.inserted_window_dims,
                .input_batching_dims = &.{},
                .scatter_indices_batching_dims = &.{},
                .scatter_dims_to_operand_dims = sparams.scatter_dims_to_operand_dims,
                .index_vector_dim = sparams.index_vector_dim,
            },
            ctx.loc,
        );
        ctx.block.append_operation(op);
        ctx.set_value(outputs[0], op.result(0));
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
        .bf16 => std.mem.asBytes(&@as(u16, 0)),
        .f32 => std.mem.asBytes(&@as(f32, 0.0)),
        .f64 => std.mem.asBytes(&@as(f64, 0.0)),
        .i32 => std.mem.asBytes(&@as(i32, 0)),
        .i64 => std.mem.asBytes(&@as(i64, 0)),
        .u32 => std.mem.asBytes(&@as(u32, 0)),
        .u64 => std.mem.asBytes(&@as(u64, 0)),
        .bool => std.mem.asBytes(&@as(bool, false)),
    };
}

fn f32_to_bf16_bits(val: f32) u16 {
    const bits: u32 = @bitCast(val);
    return @intCast(bits >> 16);
}

fn scalar_min_bytes(dtype: pr.DType) []const u8 {
    return switch (dtype) {
        .bf16 => std.mem.asBytes(&f32_to_bf16_bits(-std.math.inf(f32))),
        .f32 => std.mem.asBytes(&@as(f32, -std.math.inf(f32))),
        .f64 => std.mem.asBytes(&@as(f64, -std.math.inf(f64))),
        .i32 => std.mem.asBytes(&std.math.minInt(i32)),
        .i64 => std.mem.asBytes(&std.math.minInt(i64)),
        .u32 => std.mem.asBytes(&@as(u32, 0)),
        .u64 => std.mem.asBytes(&@as(u64, 0)),
        .bool => std.mem.asBytes(&@as(bool, false)),
    };
}

fn reduce_add_block(_: anytype, ctx: mlir.Context, inputs: []const mlir.Value, accs: []const mlir.Value) mlir.Operation {
    return stablehlo.add(ctx, inputs[0], accs[0], mlir.Location.unknown(ctx));
}

fn reduce_max_block(_: anytype, ctx: mlir.Context, inputs: []const mlir.Value, accs: []const mlir.Value) mlir.Operation {
    return stablehlo.maximum(ctx, inputs[0], accs[0], mlir.Location.unknown(ctx));
}

fn make_update_block(ctx: mlir.Context, operand_type: mlir.Type, loc: mlir.Location, reduction: pr.ScatterReduction) mlir.Block {
    const elem_type = if (operand_type.as(mlir.RankedTensorType)) |shaped| shaped.get_element_type() else operand_type;
    const arg_type: mlir.Type = .tensor(&.{}, elem_type);
    var block = mlir.Block.init(&.{ arg_type, arg_type }, &.{ loc, loc }) catch unreachable;
    const op = switch (reduction) {
        .add => stablehlo.add(ctx, block.argument(0), block.argument(1), loc),
        .max => stablehlo.maximum(ctx, block.argument(0), block.argument(1), loc),
        .min => stablehlo.minimum(ctx, block.argument(0), block.argument(1), loc),
        .mul => stablehlo.multiply(ctx, block.argument(0), block.argument(1), loc),
    };
    block.append_operation(op);
    const ret = stablehlo.return_(ctx, op.result(0), loc);
    block.append_operation(ret);
    return block;
}

fn gather_output_dims_local(
    operand_dims: []const usize,
    indices_dims: []const usize,
    params: pr.GatherParams,
) ?[]const usize {
    const max_rank: usize = 64;
    if (operand_dims.len > max_rank) return null;
    if (params.slice_sizes.len != operand_dims.len) return null;
    if (params.index_vector_dim < 0) return null;
    const index_vector_dim: usize = @intCast(params.index_vector_dim);
    if (index_vector_dim > indices_dims.len) return null;

    const index_vector_len: usize = if (index_vector_dim == indices_dims.len)
        1
    else
        indices_dims[index_vector_dim];
    if (params.start_index_map.len != index_vector_len) return null;

    var collapsed = [_]bool{false} ** max_rank;
    for (params.collapsed_slice_dims) |axis| {
        if (axis < 0) return null;
        const idx: usize = @intCast(axis);
        if (idx >= operand_dims.len) return null;
        if (collapsed[idx]) return null;
        collapsed[idx] = true;
        if (params.slice_sizes[idx] != 1) return null;
    }

    const out_rank = indices_dims.len - (if (index_vector_dim == indices_dims.len) 0 else 1) +
        (operand_dims.len - params.collapsed_slice_dims.len);
    if (out_rank > max_rank) return null;

    var out_dims_buf: [max_rank]usize = undefined;
    var out_i: usize = 0;
    for (indices_dims, 0..) |d, i| {
        if (i == index_vector_dim) continue;
        out_dims_buf[out_i] = d;
        out_i += 1;
    }
    for (0..operand_dims.len) |i| {
        if (collapsed[i]) continue;
        out_dims_buf[out_i] = @intCast(params.slice_sizes[i]);
        out_i += 1;
    }

    return out_dims_buf[0..out_i];
}

fn slice_matches_local(in_dims: []const usize, out_dims: []const usize, params: pr.SliceParams) bool {
    if (params.start_indices.len != in_dims.len) return false;
    if (params.limit_indices.len != in_dims.len) return false;
    if (params.strides.len != in_dims.len) return false;
    if (out_dims.len != in_dims.len) return false;

    for (in_dims, 0..) |dim, i| {
        const start = params.start_indices[i];
        const limit = params.limit_indices[i];
        const stride = params.strides[i];
        if (start < 0 or limit < 0 or stride <= 0) return false;
        const start_u: usize = @intCast(start);
        const limit_u: usize = @intCast(limit);
        const stride_u: usize = @intCast(stride);
        if (limit_u > dim or start_u >= limit_u) return false;
        const span = limit_u - start_u;
        const out = (span + stride_u - 1) / stride_u;
        if (out_dims[i] != out) return false;
    }
    return true;
}

fn slice_output_dims_local(allocator: std.mem.Allocator, in_dims: []const usize, params: pr.SliceParams) pr.BuildError![]const usize {
    if (params.start_indices.len != in_dims.len) return error.SliceTypeMismatch;
    if (params.limit_indices.len != in_dims.len) return error.SliceTypeMismatch;
    if (params.strides.len != in_dims.len) return error.SliceTypeMismatch;

    const out_dims = try allocator.alloc(usize, in_dims.len);
    for (in_dims, 0..) |dim, i| {
        const start = params.start_indices[i];
        const limit = params.limit_indices[i];
        const stride = params.strides[i];
        if (start < 0 or limit < 0 or stride <= 0) return error.SliceTypeMismatch;
        const start_u: usize = @intCast(start);
        const limit_u: usize = @intCast(limit);
        const stride_u: usize = @intCast(stride);
        if (limit_u > dim or start_u >= limit_u) return error.SliceTypeMismatch;
        const span = limit_u - start_u;
        out_dims[i] = (span + stride_u - 1) / stride_u;
    }
    return out_dims;
}

fn concat_matches_local(func: pr.Function, inputs: []const pr.VarId, out: pr.Tensor, axis: i64) bool {
    if (axis < 0) return false;
    const axis_u: usize = @intCast(axis);
    if (out.shape.rank() == 0 or axis_u >= out.shape.rank()) return false;

    var out_sum: usize = 0;
    for (inputs) |id| {
        const t = func.avals[@intCast(id)].as_tensor() orelse return false;
        if (t.dtype != out.dtype) return false;
        if (t.shape.rank() != out.shape.rank()) return false;
        for (t.shape.dims, 0..) |d, i| {
            if (i == axis_u) continue;
            if (d != out.shape.dims[i]) return false;
        }
        out_sum += t.shape.dims[axis_u];
    }
    return out_sum == out.shape.dims[axis_u];
}

fn concat_output_dims_local(ctx: types.InferContext, inputs: []const pr.VarId, axis: i64) pr.BuildError![]const usize {
    if (axis < 0) return error.ConcatTypeMismatch;
    const axis_u: usize = @intCast(axis);
    const first = try ctx.tensor_of(inputs[0]);
    if (first.shape.rank() == 0 or axis_u >= first.shape.rank()) return error.ConcatTypeMismatch;

    var out_dims = try ctx.alloc().dupe(usize, first.shape.dims);
    var total = out_dims[axis_u];
    for (inputs[1..]) |id| {
        const t = try ctx.tensor_of(id);
        if (t.dtype != first.dtype) return error.ConcatTypeMismatch;
        if (t.shape.rank() != first.shape.rank()) return error.ConcatTypeMismatch;
        for (t.shape.dims, 0..) |d, i| {
            if (i == axis_u) continue;
            if (d != first.shape.dims[i]) return error.ConcatTypeMismatch;
        }
        total += t.shape.dims[axis_u];
    }
    out_dims[axis_u] = total;
    return out_dims;
}

fn zeros_like(bld: *pr.FunctionBuilder, dtype: pr.DType, dims: []const usize) pr.BuildError!pr.VarId {
    const lit = try bld.literal_scalar(types.scalar_literal(dtype, 0.0));
    if (dims.len == 0) return lit;
    return bld.broadcast_in_dim(lit, dims, &.{});
}

fn compare_type_for_dtype(dt: pr.DType) pr.CompareType {
    return switch (dt) {
        .bf16, .f32, .f64 => .FLOAT,
        .i32, .i64 => .SIGNED,
        .u32, .u64, .bool => .UNSIGNED,
    };
}

fn scatter_params_for_gather(params: pr.GatherParams) pr.BuildError!pr.ScatterParams {
    if (params.slice_sizes.len == 0) return error.GatherTypeMismatch;
    if (params.index_vector_dim < 0) return error.GatherTypeMismatch;

    if (params.start_index_map.len == 1) {
        return .{
            .update_window_dims = &.{1},
            .inserted_window_dims = &.{0},
            .scatter_dims_to_operand_dims = &.{0},
            .index_vector_dim = params.index_vector_dim,
            .reduction = .add,
        };
    }

    if (params.start_index_map.len == 2 and params.slice_sizes.len == 2 and params.collapsed_slice_dims.len == 2 and params.offset_dims.len == 0) {
        return .{
            .update_window_dims = &.{},
            .inserted_window_dims = &.{ 0, 1 },
            .scatter_dims_to_operand_dims = &.{ 0, 1 },
            .index_vector_dim = params.index_vector_dim,
            .reduction = .add,
        };
    }

    return error.GatherTypeMismatch;
}
