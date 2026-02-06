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

        const lhs = try ctx.tensor_of(inputs[0]);
        const rhs = try ctx.tensor_of(inputs[1]);
        const out = try ctx.tensor_of(outputs[0]);

        if (lhs.dtype != rhs.dtype or lhs.dtype != out.dtype) return error.DotTypeMismatch;
        if (lhs.shape.rank() != 2 or rhs.shape.rank() != 2 or out.shape.rank() != 2) return error.DotTypeMismatch;
        if (lhs.shape.dims[1] != rhs.shape.dims[0]) return error.DotTypeMismatch;
        if (out.shape.dims[0] != lhs.shape.dims[0] or out.shape.dims[1] != rhs.shape.dims[1]) return error.DotTypeMismatch;
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        if (ctx.inputs.len != 2) return error.InvalidEqnArity;

        const lhs = try ctx.tensor_of(ctx.inputs[0]);
        const rhs = try ctx.tensor_of(ctx.inputs[1]);

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

        const lhs = ctx.get_value(inputs[0]) orelse return error.InvalidProgram;
        const rhs = ctx.get_value(inputs[1]) orelse return error.InvalidProgram;
        const out_id = outputs[0];
        const out_tensor = try ctx.tensor_of(out_id);
        const out_type = try ctx.tensor_to_mlir_type(out_tensor);

        const op = stablehlo.dot_general(ctx.mlir_ctx, lhs, rhs, out_type, ctx.loc, .{
            .lhs_batching_dimensions = &.{},
            .rhs_batching_dimensions = &.{},
            .lhs_contracting_dimensions = &.{1},
            .rhs_contracting_dimensions = &.{0},
            .precision = .fast,
        });
        ctx.block.append_operation(op);
        ctx.set_value(out_id, op.result(0));
    }

    pub fn vjp_forward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);

        if (inputs.len != 2) return error.UnsupportedEqn;

        const lhs = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const rhs = ctx.get_primal(inputs[1]) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.dot(lhs, rhs);
        ctx.set_primal(outputs[0], out);
    }

    pub fn vjp_backward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);

        if (inputs.len != 2) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(outputs[0]) orelse return;
        const lhs_primal = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const rhs_primal = ctx.get_primal(inputs[1]) orelse return error.UnsupportedEqn;

        // For C = A @ B:
        // dA = dC @ B^T
        // dB = A^T @ dC
        const rhs_t = try ctx.builder.transpose(rhs_primal, &.{ 1, 0 });
        const lhs_t = try ctx.builder.transpose(lhs_primal, &.{ 1, 0 });

        const lhs_contrib = try ctx.builder.dot(out_cot, rhs_t);
        const rhs_contrib = try ctx.builder.dot(lhs_t, out_cot);

        try ctx.add_cot(inputs[0], lhs_contrib);
        try ctx.add_cot(inputs[1], rhs_contrib);
    }

    pub fn format(writer: *types.Writer, ctx: types.FormatContext) types.FormatError!void {
        const lhs = ctx.input_tensor(0) orelse return;
        const contract_dim = lhs.shape.rank() - 1;
        try writer.print("contracting=([{d}], [0]), K={d}", .{
            contract_dim,
            lhs.shape.dims[contract_dim],
        });
    }
};

// ============================================================================
// Dot General
// ============================================================================

pub const dot_general = struct {
    pub const arity = .{ .in = 2, .out = 1 };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        const inputs = ctx.inputs();
        const outputs = ctx.outputs();
        const params = ctx.params();

        if (inputs.len != 2 or outputs.len != 1) return error.InvalidEqnArity;
        const dg_params = pr.param_dot_general(params) orelse return error.InvalidParams;

        const lhs = try ctx.tensor_of(inputs[0]);
        const rhs = try ctx.tensor_of(inputs[1]);
        const out = try ctx.tensor_of(outputs[0]);
        if (lhs.dtype != rhs.dtype or lhs.dtype != out.dtype) return error.DotGeneralTypeMismatch;

        if (!pr.dot_general_matches(lhs, rhs, out.shape.dims, dg_params)) return error.DotGeneralTypeMismatch;
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        if (ctx.inputs.len != 2) return error.InvalidEqnArity;
        const dg_params = pr.param_dot_general(ctx.params) orelse return error.InvalidParams;
        const lhs = try ctx.tensor_of(ctx.inputs[0]);
        const rhs = try ctx.tensor_of(ctx.inputs[1]);
        if (lhs.dtype != rhs.dtype) return error.DotGeneralTypeMismatch;
        const out_dims = try pr.dot_general_output_dims(ctx.alloc(), lhs, rhs, dg_params);
        return .{ .tensor = .{ .dtype = lhs.dtype, .shape = .{ .dims = out_dims } } };
    }

    pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);

        if (inputs.len != 2 or outputs.len != 1) return error.InvalidProgram;
        const dg_params = pr.param_dot_general(params) orelse return error.InvalidProgram;

        const lhs = ctx.get_value(inputs[0]) orelse return error.InvalidProgram;
        const rhs = ctx.get_value(inputs[1]) orelse return error.InvalidProgram;
        const out_tensor = try ctx.tensor_of(outputs[0]);
        const out_type = try ctx.tensor_to_mlir_type(out_tensor);

        const op = stablehlo.dot_general(ctx.mlir_ctx, lhs, rhs, out_type, ctx.loc, .{
            .lhs_batching_dimensions = dg_params.lhs_batch_dims,
            .rhs_batching_dimensions = dg_params.rhs_batch_dims,
            .lhs_contracting_dimensions = dg_params.lhs_contracting_dims,
            .rhs_contracting_dimensions = dg_params.rhs_contracting_dims,
            .precision = .fast,
        });
        ctx.block.append_operation(op);
        ctx.set_value(outputs[0], op.result(0));
    }

    pub fn vjp_forward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);
        if (inputs.len != 2) return error.UnsupportedEqn;
        const dg_params = pr.param_dot_general(params) orelse return error.UnsupportedEqn;

        const lhs = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const rhs = ctx.get_primal(inputs[1]) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.dot_general(lhs, rhs, dg_params);
        ctx.set_primal(outputs[0], out);
    }

    pub fn vjp_backward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const inputs = ctx.inputs(eqn);
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);
        if (inputs.len != 2) return error.UnsupportedEqn;
        const dg_params = pr.param_dot_general(params) orelse return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(outputs[0]) orelse return;
        const lhs_primal = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const rhs_primal = ctx.get_primal(inputs[1]) orelse return error.UnsupportedEqn;

        const lhs_contrib, const rhs_contrib = blk: {
            // Case A: general batched matmul (covers attention), allowing batch dims
            // anywhere and in any order, as long as each operand has:
            //   - batch dims (paired positionally between lhs_batch_dims/rhs_batch_dims)
            //   - exactly one contracting dim (K)
            //   - exactly one remaining non-batch non-contract dim (M for lhs, N for rhs)
            // Forward shape pattern (up to permutation):
            //   lhs: [B..., M, K]
            //   rhs: [B..., K, N]
            //   out: [B..., M, N]
            // Output batch dims order follows lhs_batch_dims order.
            if (try try_batched_matmul_vjp(ctx, inputs[0], inputs[1], out_cot, lhs_primal, rhs_primal, dg_params)) |pair| {
                break :blk .{ pair.lhs, pair.rhs };
            }

            // Case B/C: (B,M,K) x (K,N) or (B,M,K) x (N,K) -> (B,M,N).
            if (dg_params.lhs_batch_dims.len == 0 and dg_params.rhs_batch_dims.len == 0 and
                dg_params.lhs_contracting_dims.len == 1 and dg_params.lhs_contracting_dims[0] == 2 and
                dg_params.rhs_contracting_dims.len == 1 and
                (dg_params.rhs_contracting_dims[0] == 0 or dg_params.rhs_contracting_dims[0] == 1))
            {
                const lhs_t = ctx.tensor_of(inputs[0]);
                const rhs_t = ctx.tensor_of(inputs[1]);
                const out_t = ctx.tensor_of(outputs[0]);

                if (lhs_t.shape.rank() != 3 or rhs_t.shape.rank() != 2 or out_t.shape.rank() != 3) {
                    return error.UnsupportedEqn;
                }

                const b = out_t.shape.dims[0];
                const m = out_t.shape.dims[1];
                const k = lhs_t.shape.dims[2];
                const rhs_contract_dim = dg_params.rhs_contracting_dims[0];
                const n = if (rhs_contract_dim == 0) rhs_t.shape.dims[1] else rhs_t.shape.dims[0];
                const bm = b * m;

                const out2 = try ctx.builder.reshape(out_cot, &.{ bm, n });

                const lhs_c = if (rhs_contract_dim == 0) lhs_blk: {
                    const rhs_t2 = try ctx.builder.transpose(rhs_primal, &.{ 1, 0 }); // (N,K)
                    const lhs_flat = try ctx.builder.dot(out2, rhs_t2); // (B*M,K)
                    break :lhs_blk try ctx.builder.reshape(lhs_flat, &.{ b, m, k });
                } else lhs_blk: {
                    const lhs_flat = try ctx.builder.dot(out2, rhs_primal); // (B*M,K)
                    break :lhs_blk try ctx.builder.reshape(lhs_flat, &.{ b, m, k });
                };

                const lhs2 = try ctx.builder.reshape(lhs_primal, &.{ bm, k });
                const lhs2_t = try ctx.builder.transpose(lhs2, &.{ 1, 0 }); // (K,B*M)
                const rhs_c = if (rhs_contract_dim == 0) rhs_blk: {
                    break :rhs_blk try ctx.builder.dot(lhs2_t, out2); // (K,N)
                } else rhs_blk: {
                    const rhs_k_n = try ctx.builder.dot(lhs2_t, out2); // (K,N)
                    break :rhs_blk try ctx.builder.transpose(rhs_k_n, &.{ 1, 0 }); // (N,K)
                };

                break :blk .{ lhs_c, rhs_c };
            }

            return error.UnsupportedEqn;
        };

        try ctx.add_cot(inputs[0], lhs_contrib);
        try ctx.add_cot(inputs[1], rhs_contrib);
    }

    pub fn format(writer: *types.Writer, ctx: types.FormatContext) types.FormatError!void {
        const dg_params = pr.param_dot_general(ctx.params()) orelse return;
        try writer.writeAll("batch=(lhs=");
        try write_dims(writer, dg_params.lhs_batch_dims);
        try writer.writeAll(", rhs=");
        try write_dims(writer, dg_params.rhs_batch_dims);
        try writer.writeAll(") contracting=(lhs=");
        try write_dims(writer, dg_params.lhs_contracting_dims);
        try writer.writeAll(", rhs=");
        try write_dims(writer, dg_params.rhs_contracting_dims);
        try writer.writeAll(")");
    }
};

const BatchedMatmulVjpPair = struct { lhs: pr.VarId, rhs: pr.VarId };

fn try_batched_matmul_vjp(
    ctx: types.AdContext,
    lhs_id: pr.VarId,
    rhs_id: pr.VarId,
    out_cot: pr.VarId,
    lhs_primal: pr.VarId,
    rhs_primal: pr.VarId,
    params: pr.DotGeneralParams,
) types.AdError!?BatchedMatmulVjpPair {
    const batch_len: usize = params.lhs_batch_dims.len;
    if (batch_len == 0) return null;
    if (batch_len != params.rhs_batch_dims.len) return null;
    if (params.lhs_contracting_dims.len != 1) return null;
    if (params.rhs_contracting_dims.len != 1) return null;

    const lhs_t = ctx.tensor_of(lhs_id);
    const rhs_t = ctx.tensor_of(rhs_id);
    const lhs_rank = lhs_t.shape.rank();
    const rhs_rank = rhs_t.shape.rank();
    if (lhs_rank < 2 or rhs_rank < 2) return null;

    const lhs_k_dim: i64 = params.lhs_contracting_dims[0];
    const rhs_k_dim: i64 = params.rhs_contracting_dims[0];
    const lhs_m_dim = find_single_other_dim(lhs_rank, params.lhs_batch_dims, lhs_k_dim) orelse return null;
    const rhs_n_dim = find_single_other_dim(rhs_rank, params.rhs_batch_dims, rhs_k_dim) orelse return null;

    // out dims are: [batch_len batch dims] + [lhs M] + [rhs N]
    const out_m_dim: i64 = @intCast(batch_len);
    const out_n_dim: i64 = @intCast(batch_len + 1);

    // Batch dims in out_cot are always prefix [0..batch_len).
    var out_batch_buf: [8]i64 = undefined;
    if (batch_len > out_batch_buf.len) return error.UnsupportedEqn;
    for (0..batch_len) |i| out_batch_buf[i] = @intCast(i);
    const out_batch = out_batch_buf[0..batch_len];

    // d_lhs = dot_general(d_out, rhs, contracting over N)
    var out_contract_n: [1]i64 = .{out_n_dim};
    var rhs_contract_n: [1]i64 = .{rhs_n_dim};
    const d_lhs_canon = try ctx.builder.dot_general(out_cot, rhs_primal, .{
        .lhs_batch_dims = out_batch,
        .rhs_batch_dims = params.rhs_batch_dims,
        .lhs_contracting_dims = out_contract_n[0..],
        .rhs_contracting_dims = rhs_contract_n[0..],
    });

    const d_lhs = try transpose_to_match(ctx.builder, d_lhs_canon, lhs_rank, params.lhs_batch_dims, lhs_m_dim, lhs_k_dim, batch_len);

    // d_rhs = dot_general(lhs, d_out, contracting over M)
    var lhs_contract_m: [1]i64 = .{lhs_m_dim};
    var out_contract_m: [1]i64 = .{out_m_dim};
    const d_rhs_canon = try ctx.builder.dot_general(lhs_primal, out_cot, .{
        .lhs_batch_dims = params.lhs_batch_dims,
        .rhs_batch_dims = out_batch,
        .lhs_contracting_dims = lhs_contract_m[0..],
        .rhs_contracting_dims = out_contract_m[0..],
    });

    const d_rhs = try transpose_to_match(ctx.builder, d_rhs_canon, rhs_rank, params.rhs_batch_dims, rhs_k_dim, rhs_n_dim, batch_len);

    return .{ .lhs = d_lhs, .rhs = d_rhs };
}

fn find_single_other_dim(rank: usize, batch_dims: []const i64, contracting_dim: i64) ?i64 {
    var found: ?i64 = null;
    var d: usize = 0;
    while (d < rank) : (d += 1) {
        const dim_i64: i64 = @intCast(d);
        if (dim_i64 == contracting_dim) continue;
        if (index_of_i64(batch_dims, dim_i64) != null) continue;
        if (found != null) return null;
        found = dim_i64;
    }
    return found;
}

fn index_of_i64(list: []const i64, needle: i64) ?usize {
    for (list, 0..) |v, i| {
        if (v == needle) return i;
    }
    return null;
}

fn transpose_to_match(
    b: *pr.FunctionBuilder,
    canon: pr.VarId,
    rank: usize,
    batch_dims: []const i64,
    a_dim: i64,
    b_dim: i64,
    batch_len: usize,
) types.AdError!pr.VarId {
    // Canonical layout is: [batch dims] + [a_dim] + [b_dim].
    // Produce a transpose permutation that yields dims in original operand order [0..rank).
    var perm_buf: [8]i64 = undefined;
    if (rank > perm_buf.len) return error.UnsupportedEqn;

    var is_identity = true;
    var d: usize = 0;
    while (d < rank) : (d += 1) {
        const dim_i64: i64 = @intCast(d);
        const src: i64 = if (index_of_i64(batch_dims, dim_i64)) |bi| blk: {
            break :blk @intCast(bi);
        } else if (dim_i64 == a_dim) @intCast(batch_len) else if (dim_i64 == b_dim) @intCast(batch_len + 1) else return error.UnsupportedEqn;
        perm_buf[d] = src;
        if (src != dim_i64) is_identity = false;
    }

    if (is_identity) return canon;
    return try b.transpose(canon, perm_buf[0..rank]);
}

fn write_dims(writer: *types.Writer, dims: []const i64) types.FormatError!void {
    try writer.writeAll("[");
    for (dims, 0..) |d, i| {
        if (i != 0) try writer.writeAll(", ");
        try writer.print("{d}", .{d});
    }
    try writer.writeAll("]");
}
