//! Special Operations
//! Ops with unique semantics (custom_call, call).
const std = @import("std");
const types = @import("types.zig");
const pr = @import("../pr.zig");
const Aval = pr.Aval;

// ============================================================================
// Custom Call
// ============================================================================

pub const custom_call = struct {
    pub const arity = .{ .in = .variadic, .out = .variadic };

    pub fn validate(op: *const pr.Op, cc: pr.CustomCallParams) pr.ValidationError!void {
        if (cc.out_avals.len != op.outputs.len) return error.InvalidOpArity;
    }

    pub fn infer_output(_: std.mem.Allocator, _: []const *pr.Var, cc: pr.CustomCallParams) pr.BuildError!Aval {
        if (cc.out_avals.len != 1) return error.InvalidOpArity;
        return cc.out_avals[0];
    }

    pub fn format(writer: *types.Writer, _: *const pr.Op, cc: pr.CustomCallParams) types.FormatError!void {
        try writer.print("target=\"{s}\"", .{cc.target_name});
        if (cc.has_side_effect) try writer.writeAll(", side_effect=true");
    }
};

// ============================================================================
// Call
// ============================================================================

pub const call = struct {
    pub const arity = .{ .in = .variadic, .out = .variadic };

    pub fn validate(_: *const pr.Op, _: pr.CallParams) pr.ValidationError!void {}

    pub fn infer_output(_: std.mem.Allocator, _: []const *pr.Var, _: pr.CallParams) pr.BuildError!Aval {
        // Call output types are determined by the callee, not inferred here.
        // The FunctionBuilder.call method handles this via emit_with_outputs.
        return error.InvalidOpArity;
    }

    pub fn format(writer: *types.Writer, _: *const pr.Op, cp: pr.CallParams) types.FormatError!void {
        try writer.print("callee=\"{s}\"", .{cp.callee});
    }
};
