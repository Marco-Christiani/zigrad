//! Registers op implementations and provides runtime dispatch.
//!
//! - `OpFor(prim)`: map from `pr.Prim` enum variant to its implementation struct.
//! - Runtime dispatch helpers (`validate`, `infer_output`, `emit_primal`, ...)
//!
//! ## Op interface
//!
//! Every op implementation is a struct with methods named per `op_methods`
//!  below. `validate` and `infer_output` are required. Missing them fails the
//!  build via the `validate_op_interface` comptime block. The others
//!  (`format`, `emit_primal`, `vjp_backward`, `jvp`) are optional and probed
//!  via `@hasDecl` at dispatch time.
//!
//! Coverage: build with `-Demit-op-coverage=true` to have the registry
//!  `@compileLog` a per-op table of which interface methods are implemented.
//!  Used for tracking AD / lowering coverage without grepping the source.
//!
const std = @import("std");
const pr = @import("../pr.zig");
const build_options = @import("build_options");

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

// Interface spec

/// Describes one method in the op interface.
///
/// A required method must exist on every op. Optional methods are probed via
///  `@hasDecl` at dispatch time.
const MethodSpec = struct {
    name: []const u8,
    required: bool,
    doc: []const u8,
};

/// The canonical op interface. Order matters only for the coverage report
///  layout.
const op_methods: []const MethodSpec = &.{
    .{ .name = "validate", .required = true, .doc = "Op construction sanity check" },
    // TODO(pr): Make inference optional for operations with explicit result types.
    .{ .name = "infer_output", .required = true, .doc = "Shape/dtype inference" },
    .{ .name = "format", .required = false, .doc = "IR dump formatting" },
    .{ .name = "emit_primal", .required = false, .doc = "Primal re-emission for AD" },
    .{ .name = "vjp_backward", .required = false, .doc = "Cotangent propagation for reverse-mode AD" },
    .{ .name = "jvp", .required = false, .doc = "Tangent propagation for forward-mode AD" },
};

/// Comptime enforcement of required interface methods.
///
/// Missing any required method is a hard build error identifying the
///  op and the missing method.
fn validate_op_interface() void {
    inline for (comptime std.enums.values(pr.Prim)) |prim| {
        const Op = OpFor(prim);
        inline for (op_methods) |m| {
            if (m.required and !@hasDecl(Op, m.name)) {
                @compileError("op " ++ @tagName(prim) ++ " missing required method: " ++ m.name);
            }
        }
    }
}

/// Comptime coverage report.
///
/// Emits one `@compileLog` per row so each line shows up as its own output
///  entry since a single multi-line `@compileLog` would serialize with `\n`
///  escaped, which is unreadable.
///
/// Opt-in via `-Demit-op-coverage=true`.
fn emit_op_coverage_report() void {
    @setEvalBranchQuota(100000);
    @compileLog("== op interface coverage ==");

    // Header row.
    comptime var header: []const u8 = std.fmt.comptimePrint("  {s: <18}", .{"op"});
    inline for (op_methods) |m| {
        header = header ++ std.fmt.comptimePrint(" {s: <13}", .{m.name});
    }
    @compileLog(header);

    // inline for unrolls, each op gets a row
    inline for (comptime std.enums.values(pr.Prim)) |prim| {
        const Op = OpFor(prim);
        comptime var row: []const u8 = std.fmt.comptimePrint("  {s: <18}", .{@tagName(prim)});
        inline for (op_methods) |m| {
            row = row ++ std.fmt.comptimePrint(" {s: <13}", .{
                if (@hasDecl(Op, m.name)) "yes" else "-",
            });
        }
        @compileLog(row);
    }

    @compileLog('-' ** 30 ++ "totals" ++ '-' ** 30);
    inline for (op_methods) |m| {
        comptime var hits: usize = 0;
        inline for (comptime std.enums.values(pr.Prim)) |prim| {
            if (@hasDecl(OpFor(prim), m.name)) hits += 1;
        }
        const line = comptime std.fmt.comptimePrint("  {s: <13} {d}/{d}  {s}", .{
            m.name,
            hits,
            std.enums.values(pr.Prim).len,
            m.doc,
        });
        @compileLog(line);
    }
}

comptime {
    validate_op_interface();
    if (build_options.emit_op_coverage) emit_op_coverage_report();
}

// Dispatch Functions

/// Validate an op using the handler's validate function.
pub fn validate(op: *const pr.Op) pr.ValidationError!void {
    switch (op.params) {
        inline else => |typed_params, tag| {
            return try OpFor(tag).validate(op, typed_params);
        },
    }
}

/// Infer output type for a set of params and inputs.
/// Called by FunctionBuilder before the Op exists.
pub fn infer_output(alloc: std.mem.Allocator, params: pr.Params, inputs: []const *pr.Var) pr.BuildError!pr.Aval {
    switch (params) {
        inline else => |typed_params, tag| {
            return try OpFor(tag).infer_output(alloc, inputs, typed_params);
        },
    }
}

/// Check if an op supports VJP by providing `emit_primal` and `vjp_backward`.
pub fn has_vjp(prim: pr.Prim) bool {
    return switch (prim) {
        inline else => |p| @hasDecl(OpFor(p), "emit_primal") and @hasDecl(OpFor(p), "vjp_backward"),
    };
}

/// Check if an op can emit its primal computation into an AD-derived function.
pub fn has_emit_primal(prim: pr.Prim) bool {
    return switch (prim) {
        inline else => |p| @hasDecl(OpFor(p), "emit_primal"),
    };
}

/// Emit an op's primal computation into an AD-derived function.
pub fn emit_primal(ctx: types.AdContext, op: *const pr.Op) types.AdError!void {
    switch (op.params) {
        inline else => |typed_params, tag| {
            const Handler = OpFor(tag);
            if (@hasDecl(Handler, "emit_primal")) {
                return try Handler.emit_primal(ctx, op, typed_params);
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
                return try Handler.vjp_backward(ctx, op, typed_params);
            }
            // Missing rules contribute zero for nondifferentiable operations.
            return;
        },
    }
}

/// Check if an op supports JVP through both required AD hooks.
pub fn has_jvp(prim: pr.Prim) bool {
    return switch (prim) {
        inline else => |p| @hasDecl(OpFor(p), "emit_primal") and @hasDecl(OpFor(p), "jvp"),
    };
}

/// Execute JVP for an op (forward-mode tangent propagation).
pub fn jvp(ctx: types.AdContext, op: *const pr.Op) types.AdError!void {
    switch (op.params) {
        inline else => |typed_params, tag| {
            const Handler = OpFor(tag);
            if (@hasDecl(Handler, "jvp")) {
                return try Handler.jvp(ctx, op, typed_params);
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
                return try Handler.format(writer, op, typed_params);
            }
            return;
        },
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

    try std.testing.expect(has_emit_primal(.literal));
    try std.testing.expect(!has_vjp(.literal));

    try std.testing.expect(has_vjp(.convert));

    try std.testing.expect(has_vjp(.maximum));
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
