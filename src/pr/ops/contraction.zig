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

        if (!dot_general_vjp_supported(dg_params)) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(outputs[0]) orelse return;
        const lhs_primal = ctx.get_primal(inputs[0]) orelse return error.UnsupportedEqn;
        const rhs_primal = ctx.get_primal(inputs[1]) orelse return error.UnsupportedEqn;

        const lhs_contrib, const rhs_contrib = blk: {
            // Case A: batched attention matmul.
            if (dg_params.lhs_batch_dims.len == 1 and dg_params.lhs_batch_dims[0] == 0 and
                dg_params.rhs_batch_dims.len == 1 and dg_params.rhs_batch_dims[0] == 0 and
                dg_params.lhs_contracting_dims.len == 1 and dg_params.lhs_contracting_dims[0] == 2 and
                dg_params.rhs_contracting_dims.len == 1 and dg_params.rhs_contracting_dims[0] == 1)
            {
                const lhs_c = try ctx.builder.dot_general(out_cot, rhs_primal, .{
                    .lhs_batch_dims = &.{0},
                    .rhs_batch_dims = &.{0},
                    .lhs_contracting_dims = &.{2},
                    .rhs_contracting_dims = &.{2},
                });
                const rhs_c = try ctx.builder.dot_general(lhs_primal, out_cot, .{
                    .lhs_batch_dims = &.{0},
                    .rhs_batch_dims = &.{0},
                    .lhs_contracting_dims = &.{1},
                    .rhs_contracting_dims = &.{1},
                });
                break :blk .{ lhs_c, rhs_c };
            }

            // Case B: (B,M,K) x (K,N) -> (B,M,N) with no explicit batching dims.
            // dL/dlhs = (dL/dout) x rhs^T  (contract N)
            // dL/drhs = lhs^T x (dL/dout)  (contract B and M)
            if (dg_params.lhs_batch_dims.len == 0 and dg_params.rhs_batch_dims.len == 0 and
                dg_params.lhs_contracting_dims.len == 1 and dg_params.lhs_contracting_dims[0] == 2 and
                dg_params.rhs_contracting_dims.len == 1 and dg_params.rhs_contracting_dims[0] == 0)
            {
                const lhs_t = ctx.tensor_of(inputs[0]);
                const rhs_t = ctx.tensor_of(inputs[1]);
                const out_t = ctx.tensor_of(outputs[0]);

                if (lhs_t.shape.rank() != 3 or rhs_t.shape.rank() != 2 or out_t.shape.rank() != 3) {
                    return error.UnsupportedEqn;
                }

                const b = out_t.shape.dims[0];
                const m = out_t.shape.dims[1];
                const n = out_t.shape.dims[2];
                const k = rhs_t.shape.dims[0];
                const bm = b * m;

                // Flatten (B,M,*) -> (B*M,*) and use 2D dot/transpose.
                const out2 = try ctx.builder.reshape(out_cot, &.{ bm, n });

                const rhs_t2 = try ctx.builder.transpose(rhs_primal, &.{ 1, 0 }); // (N,K)
                const lhs_flat = try ctx.builder.dot(out2, rhs_t2); // (B*M,K)
                const lhs_c = try ctx.builder.reshape(lhs_flat, &.{ b, m, k });

                const lhs2 = try ctx.builder.reshape(lhs_primal, &.{ bm, k });
                const lhs2_t = try ctx.builder.transpose(lhs2, &.{ 1, 0 }); // (K,B*M)
                const rhs_c = try ctx.builder.dot(lhs2_t, out2); // (K,N)

                break :blk .{ lhs_c, rhs_c };
            }

            return error.UnsupportedEqn;
        };

        try ctx.add_cot(inputs[0], lhs_contrib);
        try ctx.add_cot(inputs[1], rhs_contrib);
    }
};

fn dot_general_vjp_supported(params: pr.DotGeneralParams) bool {
    // Case A: batched matmul used by attention.
    if (params.lhs_batch_dims.len == 1 and
        params.rhs_batch_dims.len == 1 and
        params.lhs_batch_dims[0] == 0 and
        params.rhs_batch_dims[0] == 0 and
        params.lhs_contracting_dims.len == 1 and
        params.rhs_contracting_dims.len == 1 and
        params.lhs_contracting_dims[0] == 2 and
        params.rhs_contracting_dims[0] == 1)
    {
        return true;
    }

    // Case B: explicit batch dim on LHS with 2D RHS (B,M,K) x (K,N) -> (B,M,N).
    // (lhs_batch_dims/rhs_batch_dims empty, contracting dims are (2) and (0)).
    return params.lhs_batch_dims.len == 0 and
        params.rhs_batch_dims.len == 0 and
        params.lhs_contracting_dims.len == 1 and
        params.rhs_contracting_dims.len == 1 and
        params.lhs_contracting_dims[0] == 2 and
        params.rhs_contracting_dims[0] == 0;
}
