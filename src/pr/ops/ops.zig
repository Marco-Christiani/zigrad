/// Op Registry
/// Central registry for all ops with comptime dispatch and validation.
const std = @import("std");
const pr = @import("../pr.zig");

pub const types = @import("types.zig");
pub const constant = @import("constant.zig");
pub const elementwise = @import("elementwise.zig");
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
        .maximum => elementwise.maximum,
        .dot => contraction.dot,
        .reshape => shape.reshape,
        .transpose => shape.transpose,
        .broadcast_in_dim => shape.broadcast_in_dim,
        .custom_call => special.custom_call,
    };
}

/// Comptime validation that all ops implement required interface
fn validateOpInterface() void {
    inline for (comptime std.enums.values(pr.Prim)) |prim| {
        const Op = OpFor(prim);

        // Required methods
        if (!@hasDecl(Op, "validate")) {
            @compileError("Op " ++ @tagName(prim) ++ " missing validate()");
        }
        if (!@hasDecl(Op, "inferOutput")) {
            @compileError("Op " ++ @tagName(prim) ++ " missing inferOutput()");
        }
        if (!@hasDecl(Op, "lower")) {
            @compileError("Op " ++ @tagName(prim) ++ " missing lower()");
        }

        // Optional: vjpForward/vjpBackward (AD support)
        // These are checked at runtime when AD is requested
    }
}

// Run comptime validation
comptime {
    validateOpInterface();
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
pub fn inferOutput(builder: *pr.FunctionBuilder, prim: pr.Prim, inputs: []const pr.VarId, params: []const pr.Param) pr.BuildError!pr.Aval {
    const ctx = types.InferContext{ .builder = builder, .inputs = inputs, .params = params };
    inline for (comptime std.enums.values(pr.Prim)) |p| {
        if (prim == p) {
            return OpFor(p).inferOutput(ctx);
        }
    }
    unreachable;
}

/// Lower an equation to MLIR
pub fn lower(ctx: types.LowerContext, eqn: pr.Eqn) types.LowerError!void {
    inline for (comptime std.enums.values(pr.Prim)) |prim| {
        if (eqn.prim == prim) {
            return OpFor(prim).lower(ctx, eqn);
        }
    }
    unreachable;
}

/// Check if an op supports VJP
pub fn hasVjp(prim: pr.Prim) bool {
    inline for (comptime std.enums.values(pr.Prim)) |p| {
        if (prim == p) {
            const Op = OpFor(p);
            return @hasDecl(Op, "vjpForward") and @hasDecl(Op, "vjpBackward");
        }
    }
    unreachable;
}

/// Check if an op has VJP forward (for primals computation)
pub fn hasVjpForward(prim: pr.Prim) bool {
    inline for (comptime std.enums.values(pr.Prim)) |p| {
        if (prim == p) {
            return @hasDecl(OpFor(p), "vjpForward");
        }
    }
    unreachable;
}

/// Execute VJP forward pass for an equation
pub fn vjpForward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
    inline for (comptime std.enums.values(pr.Prim)) |prim| {
        if (eqn.prim == prim) {
            const Op = OpFor(prim);
            if (@hasDecl(Op, "vjpForward")) {
                return Op.vjpForward(ctx, eqn);
            } else {
                return error.UnsupportedEqn;
            }
        }
    }
    unreachable;
}

/// Execute VJP backward pass for an equation
pub fn vjpBackward(ctx: types.AdContext, eqn: pr.Eqn) types.AdError!void {
    inline for (comptime std.enums.values(pr.Prim)) |prim| {
        if (eqn.prim == prim) {
            const Op = OpFor(prim);
            if (@hasDecl(Op, "vjpBackward")) {
                return Op.vjpBackward(ctx, eqn);
            } else {
                // No backward = zero gradient (e.g., literal)
                return;
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
        try std.testing.expect(@hasDecl(Op, "inferOutput"));
        try std.testing.expect(@hasDecl(Op, "lower"));
    }
}

test "vjp support detection" {
    // Ops with full VJP support
    try std.testing.expect(hasVjp(.add));
    try std.testing.expect(hasVjp(.subtract));
    try std.testing.expect(hasVjp(.multiply));
    try std.testing.expect(hasVjp(.dot));
    try std.testing.expect(hasVjp(.reshape));
    try std.testing.expect(hasVjp(.transpose));

    // Ops with forward only (constants)
    try std.testing.expect(hasVjpForward(.literal));
    try std.testing.expect(!hasVjp(.literal));

    // Ops without VJP
    try std.testing.expect(!hasVjp(.maximum));
    try std.testing.expect(!hasVjp(.broadcast_in_dim));
    try std.testing.expect(!hasVjp(.custom_call));
}
