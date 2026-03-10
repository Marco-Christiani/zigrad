/// Op Registry
/// Central registry for all ops with comptime dispatch and validation.
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

/// Map from Prim enum to op implementation struct
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

/// Comptime validation that all ops implement required interface
fn validate_op_interface() void {
    inline for (comptime std.enums.values(pr.Prim)) |prim| {
        const Op = OpFor(prim);

        // Required methods
        if (!@hasDecl(Op, "validate")) {
            @compileError("Op " ++ @tagName(prim) ++ " missing validate()");
        }
        if (!@hasDecl(Op, "infer_output")) {
            @compileError("Op " ++ @tagName(prim) ++ " missing infer_output()");
        }
        // Optional: vjp_forward/vjp_backward (AD support)
        // These are checked at runtime when AD is requested
    }
}

// Run comptime validation
comptime {
    validate_op_interface();
}

// ============================================================================
// Dispatch Functions
// ============================================================================

/// Validate an equation using the op's validate function
pub fn validate(func: pr.Function, eqn: pr.Eqn) pr.ValidationError!void {
    const ctx = types.ValidateContext{ .func = func, .eqn = eqn };
    inline for (comptime std.enums.values(pr.Prim)) |prim| {
        if (eqn.prim == prim) {
            return OpFor(prim).validate(ctx);
        }
    }
    unreachable;
}

/// Infer output type for an equation
pub fn infer_output(builder: *pr.FunctionBuilder, prim: pr.Prim, inputs: []const pr.VarId, params: []const pr.Param) pr.BuildError!pr.Aval {
    const ctx = types.InferContext{ .builder = builder, .inputs = inputs, .params = params };
    inline for (comptime std.enums.values(pr.Prim)) |p| {
        if (prim == p) {
            return OpFor(p).infer_output(ctx);
        }
    }
    unreachable;
}

/// Check if an op supports VJP
pub fn has_vjp(prim: pr.Prim) bool {
    inline for (comptime std.enums.values(pr.Prim)) |p| {
        if (prim == p) {
            const Op = OpFor(p);
            return @hasDecl(Op, "vjp_forward") and @hasDecl(Op, "vjp_backward");
        }
    }
    unreachable;
}

/// Check if an op has VJP forward (for primals computation)
pub fn has_vjp_forward(prim: pr.Prim) bool {
    inline for (comptime std.enums.values(pr.Prim)) |p| {
        if (prim == p) {
            return @hasDecl(OpFor(p), "vjp_forward");
        }
    }
    unreachable;
}

/// Execute VJP forward pass for an equation
pub fn vjp_forward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
    inline for (comptime std.enums.values(pr.Prim)) |prim| {
        if (eqn.prim == prim) {
            const Op = OpFor(prim);
            if (@hasDecl(Op, "vjp_forward")) {
                return Op.vjp_forward(ctx, eqn);
            } else {
                return error.UnsupportedEqn;
            }
        }
    }
    unreachable;
}

/// Execute VJP backward pass for an equation
pub fn vjp_backward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
    inline for (comptime std.enums.values(pr.Prim)) |prim| {
        if (eqn.prim == prim) {
            const Op = OpFor(prim);
            if (@hasDecl(Op, "vjp_backward")) {
                return Op.vjp_backward(ctx, eqn);
            } else {
                // No backward = zero gradient (e.g., literal)
                return;
            }
        }
    }
    unreachable;
}

/// Check if an op supports JVP
pub fn has_jvp(prim: pr.Prim) bool {
    inline for (comptime std.enums.values(pr.Prim)) |p| {
        if (prim == p) {
            return @hasDecl(OpFor(p), "jvp");
        }
    }
    unreachable;
}

/// Execute JVP for an equation (forward-mode tangent propagation).
pub fn jvp(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
    inline for (comptime std.enums.values(pr.Prim)) |prim| {
        if (eqn.prim == prim) {
            const Op = OpFor(prim);
            if (@hasDecl(Op, "jvp")) {
                return Op.jvp(ctx, eqn);
            } else {
                return error.UnsupportedEqn;
            }
        }
    }
    unreachable;
}

/// Format op-specific attributes for an equation
pub fn format(writer: *types.Writer, func: pr.Function, eqn: pr.Eqn) types.FormatError!void {
    const ctx = types.FormatContext{ .func = func, .eqn = eqn };
    inline for (comptime std.enums.values(pr.Prim)) |prim| {
        if (eqn.prim == prim) {
            const Op = OpFor(prim);
            if (@hasDecl(Op, "format")) {
                return Op.format(writer, ctx);
            } else {
                return; // No format = nothing extra to show
            }
        }
    }
    unreachable;
}

// ============================================================================
// Tests
// ============================================================================

test "all prims have op implementations" {
    // This is validated at comptime, but let's also verify at runtime
    inline for (comptime std.enums.values(pr.Prim)) |prim| {
        const Op = OpFor(prim);
        try std.testing.expect(@hasDecl(Op, "validate"));
        try std.testing.expect(@hasDecl(Op, "infer_output"));
    }
}

test "vjp support detection" {
    // Ops with full VJP support
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

    // Ops with forward only (constants)
    try std.testing.expect(has_vjp_forward(.literal));
    try std.testing.expect(!has_vjp(.literal));

    // Ops with VJP (convert has forward + backward)
    try std.testing.expect(has_vjp(.convert));

    // Ops without VJP
    try std.testing.expect(!has_vjp(.maximum));
    try std.testing.expect(!has_vjp(.scatter));
    try std.testing.expect(!has_vjp(.compare));
    try std.testing.expect(!has_vjp(.call));
    try std.testing.expect(!has_vjp(.custom_call));
}

test "jvp support detection" {
    // Elementwise ops
    try std.testing.expect(has_jvp(.add));
    try std.testing.expect(has_jvp(.subtract));
    try std.testing.expect(has_jvp(.multiply));
    try std.testing.expect(has_jvp(.divide));

    // Unary ops
    try std.testing.expect(has_jvp(.exp));
    try std.testing.expect(has_jvp(.log));
    try std.testing.expect(has_jvp(.rsqrt));
    try std.testing.expect(has_jvp(.logistic));
    try std.testing.expect(has_jvp(.convert));

    // Shape ops
    try std.testing.expect(has_jvp(.reshape));
    try std.testing.expect(has_jvp(.transpose));
    try std.testing.expect(has_jvp(.broadcast_in_dim));
    try std.testing.expect(has_jvp(.reduce_sum));
    try std.testing.expect(has_jvp(.reduce_max));
    try std.testing.expect(has_jvp(.slice));
    try std.testing.expect(has_jvp(.concatenate));
    try std.testing.expect(has_jvp(.gather));
    try std.testing.expect(has_jvp(.iota));

    // Contraction ops
    try std.testing.expect(has_jvp(.dot));
    try std.testing.expect(has_jvp(.dot_general));

    // Constants
    try std.testing.expect(has_jvp(.literal));

    // Compare/select
    try std.testing.expect(has_jvp(.compare));
    try std.testing.expect(has_jvp(.select));

    // Unsupported
    try std.testing.expect(!has_jvp(.maximum));
    try std.testing.expect(!has_jvp(.scatter));
    try std.testing.expect(!has_jvp(.call));
    try std.testing.expect(!has_jvp(.custom_call));
}
