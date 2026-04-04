//! Constant Operations
//! Ops that produce constant values (no inputs).
const types = @import("types.zig");
const pr = @import("../pr.zig");
const Aval = pr.Aval;
const std = @import("std");

pub const literal = struct {
    pub const arity = .{ .in = 0, .out = 1 };

    pub fn validate(op: *const pr.Op, lit: pr.Literal) pr.ValidationError!void {
        if (op.inputs.len != 0 or op.outputs.len != 1) return error.InvalidOpArity;

        const out = op.result(0).as_tensor();
        if (out.dtype != lit.dtype()) return error.LiteralTypeMismatch;
        if (out.shape.rank() != 0) return error.LiteralTypeMismatch;
    }

    pub fn infer_output(_: std.mem.Allocator, _: []const *pr.Var, lit: pr.Literal) pr.BuildError!Aval {
        return .{ .tensor = .{ .dtype = lit.dtype(), .shape = .{ .dims = &.{} } } };
    }

    pub fn vjp_forward(ctx: types.AdContext, op: *const pr.Op, lit: pr.Literal) types.AdError!void {
        const out = try ctx.builder.literal_scalar(lit);
        ctx.set_primal(op.result(0), out);
    }

    // No vjp_backward needed - constants have zero gradient

    /// JVP: constants have zero tangent.
    pub fn jvp(ctx: types.AdContext, op: *const pr.Op, _: pr.Literal) types.AdError!void {
        const out_primal = ctx.get_primal(op.result(0)) orelse return error.UnsupportedEqn;
        const out_tensor = out_primal.as_tensor();
        const z = try ctx.builder.scalar(out_tensor.dtype, 0.0);
        ctx.set_tangent(op.result(0), z);
    }

    pub fn format(writer: *types.Writer, _: *const pr.Op, lit: pr.Literal) types.FormatError!void {
        switch (lit) {
            .f16 => |v| try writer.print("{d}", .{pr.DType.f16.decode_f32(v)}),
            .bf16 => |v| try writer.print("{d}", .{pr.DType.bf16.decode_f32(v)}),
            inline .f32, .f64, .i8, .u8, .i32, .i64, .u32, .u64 => |v| try writer.print("{d}", .{v}),
            .bool => |v| try writer.print("{s}", .{if (v) "true" else "false"}),
        }
    }
};
