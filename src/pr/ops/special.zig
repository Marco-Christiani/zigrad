/// Special Operations
/// Ops with unique semantics (custom_call, call).
const types = @import("types.zig");
const pr = @import("../pr.zig");

// ============================================================================
// Custom Call
// ============================================================================

pub const custom_call = struct {
    pub const arity = .{ .in = .variadic, .out = .variadic };

    pub fn validate(ctx: types.ValidateContext) pr.ValidationError!void {
        const inputs = ctx.inputs();
        const outputs = ctx.outputs();
        const params = ctx.params();

        _ = pr.param_call_target_name(params) orelse return error.InvalidParams;
        _ = pr.param_has_side_effect(params) orelse return error.InvalidParams;

        const single_out = pr.param_out_aval(params);
        const multi_outs = pr.param_out_avals(params);
        if ((single_out == null) == (multi_outs == null)) return error.InvalidParams;

        if (single_out) |out_aval| {
            if (outputs.len != 1) return error.InvalidEqnArity;
            _ = out_aval.as_tensor() orelse return error.CustomCallTypeMismatch;
        } else {
            const out_avals = multi_outs.?;
            if (outputs.len == 0 or out_avals.len != outputs.len) return error.InvalidEqnArity;
            for (out_avals) |out_aval| {
                _ = out_aval.as_tensor() orelse return error.CustomCallTypeMismatch;
            }
            for (outputs) |out_id| _ = try ctx.tensor_of(out_id);
        }

        if (single_out != null) _ = try ctx.tensor_of(outputs[0]);
        for (inputs) |in_id| _ = try ctx.tensor_of(in_id);
    }

    pub fn infer_output(ctx: types.InferContext) pr.BuildError!types.Aval {
        _ = pr.param_call_target_name(ctx.params) orelse return error.InvalidParams;
        _ = pr.param_has_side_effect(ctx.params) orelse return error.InvalidParams;

        const single_out = pr.param_out_aval(ctx.params);
        const multi_outs = pr.param_out_avals(ctx.params);
        if ((single_out == null) == (multi_outs == null)) return error.InvalidParams;

        for (ctx.inputs) |in_id| _ = try ctx.tensor_of(in_id);

        if (single_out) |out_aval| {
            _ = out_aval.as_tensor() orelse return error.CustomCallTypeMismatch;
            return out_aval;
        }

        const out_avals = multi_outs.?;
        if (out_avals.len != 1) return error.InvalidEqnArity;
        const out_aval = out_avals[0];
        _ = out_aval.as_tensor() orelse return error.CustomCallTypeMismatch;
        return out_aval;
    }

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

    pub fn format(writer: *types.Writer, ctx: types.FormatContext) types.FormatError!void {
        const params = ctx.params();
        if (pr.param_call_callee(params)) |callee| {
            try writer.print("callee=\"{s}\"", .{callee});
        }
    }
};
