const std = @import("std");
const pr = @import("pr.zig");

/// Emit a Function as human-readable text.
/// Output format resembles JAX's jaxpr for familiarity.
pub fn emitText(func: pr.Function, writer: anytype) !void {
    try writer.print("{{ {s}\n", .{func.name});

    // Print parameters
    for (func.params) |param_id| {
        try emitVarDecl(func, param_id, writer);
        try writer.writeAll(" = param\n");
    }

    if (func.params.len > 0 and func.eqns.len > 0) {
        try writer.writeAll("  ----\n");
    }

    // Print equations
    for (func.eqns) |eqn| {
        try emitEqn(func, eqn, writer);
    }

    // Print returns
    if (func.returns.len > 0) {
        try writer.writeAll("  return ");
        for (func.returns, 0..) |ret_id, i| {
            if (i > 0) try writer.writeAll(", ");
            try emitVarRef(ret_id, writer);
        }
        try writer.writeAll("\n");
    }

    try writer.writeAll("}\n");
}

fn emitVarDecl(func: pr.Function, id: pr.VarId, writer: anytype) !void {
    try writer.writeAll("  ");
    try emitVarRef(id, writer);
    try writer.writeAll(": ");
    try emitAval(func.avals[@intCast(id)], writer);
}

fn emitVarRef(id: pr.VarId, writer: anytype) !void {
    try writer.print("v{d}", .{id});
}

fn emitAval(aval: pr.Aval, writer: anytype) !void {
    switch (aval) {
        .tensor => |t| {
            try writer.print("{s}[", .{@tagName(t.dtype)});
            for (t.shape.dims, 0..) |d, i| {
                if (i > 0) try writer.writeAll(",");
                try writer.print("{d}", .{d});
            }
            try writer.writeAll("]");
        },
    }
}

fn emitEqn(func: pr.Function, eqn: pr.Eqn, writer: anytype) !void {
    const inputs = eqn.inputs.slice(pr.VarId, func.varids_store);
    const outputs = eqn.outputs.slice(pr.VarId, func.varids_store);
    const params = eqn.params.slice(pr.Param, func.params_store);

    // Output variables
    for (outputs, 0..) |out_id, i| {
        if (i > 0) try writer.writeAll(", ");
        try emitVarDecl(func, out_id, writer);
    }

    try writer.writeAll(" = ");
    try writer.print("{s}", .{@tagName(eqn.prim)});

    // Input variables
    for (inputs) |in_id| {
        try writer.writeAll(" ");
        try emitVarRef(in_id, writer);
    }

    // Parameters
    try emitParams(params, writer);

    try writer.writeAll("\n");
}

fn emitParams(params: []const pr.Param, writer: anytype) !void {
    for (params) |param| {
        switch (param) {
            .literal => |lit| {
                try writer.writeAll(" ");
                try emitLiteral(lit, writer);
            },
            .out_shape => |shape| {
                try writer.writeAll(" shape=[");
                for (shape, 0..) |d, i| {
                    if (i > 0) try writer.writeAll(",");
                    try writer.print("{d}", .{d});
                }
                try writer.writeAll("]");
            },
            .broadcast_dimensions => |dims| {
                try writer.writeAll(" dims=[");
                for (dims, 0..) |d, i| {
                    if (i > 0) try writer.writeAll(",");
                    try writer.print("{d}", .{d});
                }
                try writer.writeAll("]");
            },
            .permutation => |perm| {
                try writer.writeAll(" perm=[");
                for (perm, 0..) |p, i| {
                    if (i > 0) try writer.writeAll(",");
                    try writer.print("{d}", .{p});
                }
                try writer.writeAll("]");
            },
            .call_target_name => |name| {
                try writer.print(" target=\"{s}\"", .{name});
            },
            .has_side_effect => |eff| {
                if (eff) try writer.writeAll(" side_effect");
            },
            .out_aval => {}, // Type already shown in output declaration
        }
    }
}

fn emitLiteral(lit: pr.Literal, writer: anytype) !void {
    switch (lit) {
        .f32 => |v| try writer.print("{d}", .{v}),
        .f64 => |v| try writer.print("{d}", .{v}),
        .i32 => |v| try writer.print("{d}", .{v}),
        .i64 => |v| try writer.print("{d}", .{v}),
        .u32 => |v| try writer.print("{d}", .{v}),
        .u64 => |v| try writer.print("{d}", .{v}),
    }
}

/// Convenience: emit to an ArrayList.
pub fn emitTextAlloc(allocator: std.mem.Allocator, func: pr.Function) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    errdefer buf.deinit();
    try emitText(func, buf.writer());
    return buf.toOwnedSlice();
}

test "emitText basic function" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a = try b.paramTensor(.f32, &.{ 2, 3 });
    const c = try b.paramTensor(.f32, &.{ 3, 2 });
    const d = try b.dot(a, c);

    const func = try b.finish(&.{d});

    const text = try emitTextAlloc(std.testing.allocator, func);
    defer std.testing.allocator.free(text);

    // Basic structure checks
    try std.testing.expect(std.mem.indexOf(u8, text, "{ main") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "param") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "dot") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "return") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "f32[2,3]") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "f32[3,2]") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "f32[2,2]") != null);
}

test "emitText with literal and broadcast" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    const one = try b.literalScalar(.{ .f32 = 1.0 });
    const broadcasted = try b.broadcastInDim(one, &.{ 2, 3 }, &.{});

    const func = try b.finish(&.{broadcasted});

    const text = try emitTextAlloc(std.testing.allocator, func);
    defer std.testing.allocator.free(text);

    try std.testing.expect(std.mem.indexOf(u8, text, "literal") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "broadcast_in_dim") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "shape=[2,3]") != null);
}

test "emitText with reshape and transpose" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "shapes");
    defer b.deinit();

    const x = try b.paramTensor(.f64, &.{ 2, 3 });
    const t = try b.transpose(x, &.{ 1, 0 });
    const r = try b.reshape(t, &.{6});

    const func = try b.finish(&.{r});

    const text = try emitTextAlloc(std.testing.allocator, func);
    defer std.testing.allocator.free(text);

    try std.testing.expect(std.mem.indexOf(u8, text, "transpose") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "perm=[1,0]") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "reshape") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "shape=[6]") != null);
}

test "emitText with custom_call" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "custom");
    defer b.deinit();

    const x = try b.paramTensor(.f32, &.{ 2, 2 });
    const y = try b.customCall("my.custom.op", &.{x}, x);

    const func = try b.finish(&.{y});

    const text = try emitTextAlloc(std.testing.allocator, func);
    defer std.testing.allocator.free(text);

    try std.testing.expect(std.mem.indexOf(u8, text, "custom_call") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "target=\"my.custom.op\"") != null);
}

test "emitText sample matmul_add" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "matmul_add");
    defer b.deinit();

    const a = try b.paramTensor(.f32, &.{ 2, 3 });
    const w = try b.paramTensor(.f32, &.{ 3, 4 });
    const bias = try b.paramTensor(.f32, &.{4});

    const prod = try b.dot(a, w);
    const bias_bc = try b.broadcastInDim(bias, &.{ 2, 4 }, &.{1});
    const out = try b.add(prod, bias_bc);

    const func = try b.finish(&.{out});

    const text = try emitTextAlloc(std.testing.allocator, func);
    defer std.testing.allocator.free(text);

    // Print to see format (only in verbose test mode)
    std.debug.print("\n--- PR pretty-print output ---\n{s}--- end ---\n", .{text});

    // Verify structure
    try std.testing.expect(std.mem.indexOf(u8, text, "matmul_add") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "dot") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "broadcast_in_dim") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "add") != null);
}
