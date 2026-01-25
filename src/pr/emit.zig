const std = @import("std");
const pr = @import("pr.zig");

/// Emit a Function as human-readable text.
/// Output format resembles JAX's jaxpr for familiarity.
pub fn emit_text(func: pr.Function, writer: anytype) !void {
    try writer.print("{{ {s}\n", .{func.name});

    // Print parameters
    for (func.params) |param_id| {
        try emit_var_decl(func, param_id, writer);
        try writer.writeAll(" = param\n");
    }

    if (func.params.len > 0 and func.eqns.len > 0) {
        try writer.writeAll("  ----\n");
    }

    // Print equations
    for (func.eqns) |eqn| {
        try emit_eqn(func, eqn, writer);
    }

    // Print returns
    if (func.returns.len > 0) {
        try writer.writeAll("  return ");
        for (func.returns, 0..) |ret_id, i| {
            if (i > 0) try writer.writeAll(", ");
            try emit_var_ref(ret_id, writer);
        }
        try writer.writeAll("\n");
    }

    try writer.writeAll("}\n");
}

fn emit_var_decl(func: pr.Function, id: pr.VarId, writer: anytype) !void {
    try writer.writeAll("  ");
    try emit_var_ref(id, writer);
    try writer.writeAll(": ");
    try emit_aval(func.avals[@intCast(id)], writer);
}

fn emit_var_ref(id: pr.VarId, writer: anytype) !void {
    try writer.print("v{d}", .{id});
}

fn emit_aval(aval: pr.Aval, writer: anytype) !void {
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

fn emit_eqn(func: pr.Function, eqn: pr.Eqn, writer: anytype) !void {
    const inputs = eqn.inputs.slice(pr.VarId, func.varids_store);
    const outputs = eqn.outputs.slice(pr.VarId, func.varids_store);
    const params = eqn.params.slice(pr.Param, func.params_store);

    // Output variables
    for (outputs, 0..) |out_id, i| {
        if (i > 0) try writer.writeAll(", ");
        try emit_var_decl(func, out_id, writer);
    }

    try writer.writeAll(" = ");
    try writer.print("{s}", .{@tagName(eqn.prim)});

    // Input variables
    for (inputs) |in_id| {
        try writer.writeAll(" ");
        try emit_var_ref(in_id, writer);
    }

    // Parameters
    try emit_params(params, writer);

    try writer.writeAll("\n");
}

fn emit_params(params: []const pr.Param, writer: anytype) !void {
    for (params) |param| {
        switch (param) {
            .literal => |lit| {
                try writer.writeAll(" ");
                try emit_literal(lit, writer);
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
            .call_callee => |name| {
                try writer.print(" callee=\"{s}\"", .{name});
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

fn emit_literal(lit: pr.Literal, writer: anytype) !void {
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
pub fn emit_text_alloc(allocator: std.mem.Allocator, func: pr.Function) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    errdefer buf.deinit();
    try emit_text(func, buf.writer());
    return buf.toOwnedSlice();
}

test "emit_text basic function" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a = try b.param_tensor(.f32, &.{ 2, 3 });
    const c = try b.param_tensor(.f32, &.{ 3, 2 });
    const d = try b.dot(a, c);

    const func = try b.finish(&.{d});

    const text = try emit_text_alloc(std.testing.allocator, func);
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

test "emit_text with literal and broadcast" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    const one = try b.literal_scalar(.{ .f32 = 1.0 });
    const broadcasted = try b.broadcast_in_dim(one, &.{ 2, 3 }, &.{});

    const func = try b.finish(&.{broadcasted});

    const text = try emit_text_alloc(std.testing.allocator, func);
    defer std.testing.allocator.free(text);

    try std.testing.expect(std.mem.indexOf(u8, text, "literal") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "broadcast_in_dim") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "shape=[2,3]") != null);
}

test "emit_text with reshape and transpose" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "shapes");
    defer b.deinit();

    const x = try b.param_tensor(.f64, &.{ 2, 3 });
    const t = try b.transpose(x, &.{ 1, 0 });
    const r = try b.reshape(t, &.{6});

    const func = try b.finish(&.{r});

    const text = try emit_text_alloc(std.testing.allocator, func);
    defer std.testing.allocator.free(text);

    try std.testing.expect(std.mem.indexOf(u8, text, "transpose") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "perm=[1,0]") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "reshape") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "shape=[6]") != null);
}

test "emit_text with custom_call" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "custom");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 2 });
    const y = try b.custom_call("my.custom.op", &.{x}, x);

    const func = try b.finish(&.{y});

    const text = try emit_text_alloc(std.testing.allocator, func);
    defer std.testing.allocator.free(text);

    try std.testing.expect(std.mem.indexOf(u8, text, "custom_call") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "target=\"my.custom.op\"") != null);
}

test "emit_text sample matmul_add" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "matmul_add");
    defer b.deinit();

    const a = try b.param_tensor(.f32, &.{ 2, 3 });
    const w = try b.param_tensor(.f32, &.{ 3, 4 });
    const bias = try b.param_tensor(.f32, &.{4});

    const prod = try b.dot(a, w);
    const bias_bc = try b.broadcast_in_dim(bias, &.{ 2, 4 }, &.{1});
    const out = try b.add(prod, bias_bc);

    const func = try b.finish(&.{out});

    const text = try emit_text_alloc(std.testing.allocator, func);
    defer std.testing.allocator.free(text);

    // Print to see format (only in verbose test mode)
    std.debug.print("\n--- PR pretty-print output ---\n{s}--- end ---\n", .{text});

    // Verify structure
    try std.testing.expect(std.mem.indexOf(u8, text, "matmul_add") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "dot") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "broadcast_in_dim") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "add") != null);
}
