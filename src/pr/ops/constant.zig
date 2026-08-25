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

    // No VJP is needed because constants have zero gradient.

    pub fn format(writer: *types.Writer, _: *const pr.Op, lit: pr.Literal) types.FormatError!void {
        switch (lit) {
            .f16 => |v| try writer.print("{d}", .{pr.DType.f16.decode(f32, v)}),
            .bf16 => |v| try writer.print("{d}", .{pr.DType.bf16.decode(f32, v)}),
            inline .f32, .f64, .i8, .u8, .i32, .i64, .u32, .u64 => |v| try writer.print("{d}", .{v}),
            .bool => |v| try writer.print("{s}", .{if (v) "true" else "false"}),
        }
    }
};
