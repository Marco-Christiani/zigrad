/// Constant Operations
/// Ops that produce constant values (no inputs).
const types = @import("types.zig");
const pr = @import("../pr.zig");

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

    pub fn vjp_forward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const outputs = ctx.outputs(eqn);
        const params = ctx.params(eqn);

        const lit = pr.param_literal(params) orelse return error.UnsupportedEqn;
        const out = try ctx.builder.literal_scalar(lit);
        ctx.set_primal(outputs[0], out);
    }

    // No vjp_backward needed - constants have zero gradient

    /// JVP: constants have zero tangent.
    pub fn jvp(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
        const outputs = ctx.outputs(eqn);
        const out_tensor = ctx.builder_tensor_of(ctx.get_primal(outputs[0]) orelse return error.UnsupportedEqn);
        const z = try ctx.builder.literal_scalar(types.scalar_literal(out_tensor.dtype, 0.0));
        ctx.set_tangent(outputs[0], z);
    }

    pub fn format(writer: *types.Writer, ctx: types.FormatContext) types.FormatError!void {
        if (pr.param_literal(ctx.params())) |lit| {
            switch (lit) {
                .bf16 => |v| try writer.print("{d}", .{bf16_to_f32(v)}),
                inline .f32, .f64, .i32, .i64, .u32, .u64 => |v| try writer.print("{d}", .{v}),
                .bool => |v| try writer.print("{s}", .{if (v) "true" else "false"}),
            }
        }
    }
};

fn bf16_to_f32(val: u16) f32 {
    const bits: u32 = @as(u32, val) << 16;
    return @bitCast(bits);
}
