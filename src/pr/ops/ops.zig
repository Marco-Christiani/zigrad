//! Op Registry
//! Central registry for all ops with comptime dispatch and validation.
//! Each handler receives the op and its typed params directly.
const std = @import("std");
const pr = @import("../pr.zig");

pub const types = @import("types.zig");
pub const constant = @import("constant.zig");
pub const elementwise = @import("elementwise.zig");
pub const unary = @import("unary.zig");
pub const compare = @import("compare.zig");
pub const contraction = @import("contraction.zig");
pub const shape = @import("shape.zig");
pub const special = @import("special.zig");

/// Map from Prim enum to op implementation struct.
pub fn OpFor(comptime prim: pr.Prim) type {
    return switch (prim) {
        .literal => constant.literal,
        .add => elementwise.add,
        .subtract => elementwise.subtract,
        .multiply => elementwise.multiply,
        .divide => elementwise.divide,
        .maximum => elementwise.maximum,
        .exp => unary.exp,
        .log => unary.log,
        .rsqrt => unary.rsqrt,
        .logistic => unary.logistic,
        .compare => compare.compare,
        .select => compare.select,
        .convert => unary.convert,
        .gather => shape.gather,
        .scatter => shape.scatter,
        .dot => contraction.dot,
        .dot_general => contraction.dot_general,
        .reshape => shape.reshape,
        .iota => shape.iota,
        .transpose => shape.transpose,
        .broadcast_in_dim => shape.broadcast_in_dim,
        .slice => shape.slice,
        .concatenate => shape.concatenate,
        .reduce_sum => shape.reduce_sum,
        .reduce_max => shape.reduce_max,
        .call => special.call,
        .custom_call => special.custom_call,
    };
}

// TODO: this should be refactored.
// TODO: its reasonable to have a comptime flag that logs op interface completeness to make it easier to check coverage

/// Comptime validation that all ops implement required interface.
fn validate_op_interface() void {
    inline for (comptime std.enums.values(pr.Prim)) |prim| {
        const Op = OpFor(prim);
        if (!@hasDecl(Op, "validate"))
            @compileError("Op " ++ @tagName(prim) ++ " missing validate()");
        if (!@hasDecl(Op, "infer_output"))
            @compileError("Op " ++ @tagName(prim) ++ " missing infer_output()");
    }
}

comptime {
    validate_op_interface();
}

// ============================================================================
// Dispatch Functions
// ============================================================================

/// Validate an op using the handler's validate function.
pub fn validate(op: *const pr.Op) pr.ValidationError!void {
    switch (op.params) {
        inline else => |typed_params, tag| {
            return OpFor(tag).validate(op, typed_params);
        },
    }
}

/// Infer output type for a set of params and inputs.
/// Called by FunctionBuilder before the Op exists.
pub fn infer_output(alloc: std.mem.Allocator, params: pr.Params, inputs: []const *pr.Var) pr.BuildError!pr.Aval {
    switch (params) {
        inline else => |typed_params, tag| {
            return OpFor(tag).infer_output(alloc, inputs, typed_params);
        },
    }
}

/// Check if an op supports VJP.
pub fn has_vjp(prim: pr.Prim) bool {
    inline for (comptime std.enums.values(pr.Prim)) |p| {
        if (prim == p) {
            const Op = OpFor(p);
            return @hasDecl(Op, "vjp_forward") and @hasDecl(Op, "vjp_backward");
        }
    }
    unreachable;
}

/// Check if an op has VJP forward (for primals computation).
pub fn has_vjp_forward(prim: pr.Prim) bool {
    inline for (comptime std.enums.values(pr.Prim)) |p| {
        if (prim == p) {
            return @hasDecl(OpFor(p), "vjp_forward");
        }
    }
    unreachable;
}

/// Execute VJP forward pass for an op.
pub fn vjp_forward(ctx: types.AdContext, op: *const pr.Op) types.AdError!void {
    switch (op.params) {
        inline else => |typed_params, tag| {
            const Handler = OpFor(tag);
            if (@hasDecl(Handler, "vjp_forward")) {
                return Handler.vjp_forward(ctx, op, typed_params);
            }
            return error.UnsupportedEqn;
        },
    }
}

/// Execute VJP backward pass for an op.
pub fn vjp_backward(ctx: types.AdContext, op: *const pr.Op) types.AdError!void {
    switch (op.params) {
        inline else => |typed_params, tag| {
            const Handler = OpFor(tag);
            if (@hasDecl(Handler, "vjp_backward")) {
                return Handler.vjp_backward(ctx, op, typed_params);
            }
            // No backward = zero gradient (e.g., literal)
            return;
        },
    }
}

/// Check if an op supports JVP.
pub fn has_jvp(prim: pr.Prim) bool {
    inline for (comptime std.enums.values(pr.Prim)) |p| {
        if (prim == p) {
            return @hasDecl(OpFor(p), "jvp");
        }
    }
    unreachable;
}

/// Execute JVP for an op (forward-mode tangent propagation).
pub fn jvp(ctx: types.AdContext, op: *const pr.Op) types.AdError!void {
    switch (op.params) {
        inline else => |typed_params, tag| {
            const Handler = OpFor(tag);
            if (@hasDecl(Handler, "jvp")) {
                return Handler.jvp(ctx, op, typed_params);
            }
            return error.UnsupportedEqn;
        },
    }
}

/// Format op-specific attributes.
pub fn format(writer: *types.Writer, op: *const pr.Op) types.FormatError!void {
    switch (op.params) {
        inline else => |typed_params, tag| {
            const Handler = OpFor(tag);
            if (@hasDecl(Handler, "format")) {
                return Handler.format(writer, op, typed_params);
            }
            return;
        },
    }
}

// ============================================================================
// Tests
// ============================================================================

test "all prims have op implementations" {
    inline for (comptime std.enums.values(pr.Prim)) |prim| {
        const Op = OpFor(prim);
        try std.testing.expect(@hasDecl(Op, "validate"));
        try std.testing.expect(@hasDecl(Op, "infer_output"));
    }
}

test "vjp support detection" {
    try std.testing.expect(has_vjp(.add));
    try std.testing.expect(has_vjp(.subtract));
    try std.testing.expect(has_vjp(.multiply));
    try std.testing.expect(has_vjp(.divide));
    try std.testing.expect(has_vjp(.dot));
    try std.testing.expect(has_vjp(.reshape));
    try std.testing.expect(has_vjp(.transpose));
    try std.testing.expect(has_vjp(.broadcast_in_dim));
    try std.testing.expect(has_vjp(.reduce_sum));
    try std.testing.expect(has_vjp(.exp));
    try std.testing.expect(has_vjp(.log));
    try std.testing.expect(has_vjp(.rsqrt));
    try std.testing.expect(has_vjp(.logistic));
    try std.testing.expect(has_vjp(.gather));
    try std.testing.expect(has_vjp(.select));
    try std.testing.expect(has_vjp(.reduce_max));
    try std.testing.expect(has_vjp(.dot_general));
    try std.testing.expect(has_vjp(.slice));
    try std.testing.expect(has_vjp(.concatenate));

    try std.testing.expect(has_vjp_forward(.literal));
    try std.testing.expect(!has_vjp(.literal));

    try std.testing.expect(has_vjp(.convert));

    try std.testing.expect(!has_vjp(.maximum));
    try std.testing.expect(!has_vjp(.scatter));
    try std.testing.expect(!has_vjp(.compare));
    try std.testing.expect(!has_vjp(.call));
    try std.testing.expect(!has_vjp(.custom_call));
}

test "jvp support detection" {
    try std.testing.expect(has_jvp(.add));
    try std.testing.expect(has_jvp(.subtract));
    try std.testing.expect(has_jvp(.multiply));
    try std.testing.expect(has_jvp(.divide));

    try std.testing.expect(has_jvp(.exp));
    try std.testing.expect(has_jvp(.log));
    try std.testing.expect(has_jvp(.rsqrt));
    try std.testing.expect(has_jvp(.logistic));
    try std.testing.expect(has_jvp(.convert));

    try std.testing.expect(has_jvp(.reshape));
    try std.testing.expect(has_jvp(.transpose));
    try std.testing.expect(has_jvp(.broadcast_in_dim));
    try std.testing.expect(has_jvp(.reduce_sum));
    try std.testing.expect(has_jvp(.reduce_max));
    try std.testing.expect(has_jvp(.slice));
    try std.testing.expect(has_jvp(.concatenate));
    try std.testing.expect(has_jvp(.gather));
    try std.testing.expect(has_jvp(.iota));

    try std.testing.expect(has_jvp(.dot));
    try std.testing.expect(has_jvp(.dot_general));

    try std.testing.expect(has_jvp(.literal));

    try std.testing.expect(has_jvp(.compare));
    try std.testing.expect(has_jvp(.select));

    try std.testing.expect(!has_jvp(.maximum));
    try std.testing.expect(!has_jvp(.scatter));
    try std.testing.expect(!has_jvp(.call));
    try std.testing.expect(!has_jvp(.custom_call));
}
