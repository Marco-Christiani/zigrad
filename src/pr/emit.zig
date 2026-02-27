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
                try emit_i64_or_usize_list(usize, writer, shape);
                try writer.writeAll("]");
            },
            .broadcast_dimensions => |dims| {
                try writer.writeAll(" dims=[");
                try emit_i64_or_usize_list(i64, writer, dims);
                try writer.writeAll("]");
            },
            .permutation => |perm| {
                try writer.writeAll(" perm=[");
                try emit_i64_or_usize_list(i64, writer, perm);
                try writer.writeAll("]");
            },
            .reduce_axes => |axes| {
                try writer.writeAll(" axes=[");
                try emit_i64_or_usize_list(i64, writer, axes);
                try writer.writeAll("]");
            },
            .iota_dimension => |dim| try writer.print(" iota_dim={d}", .{dim}),
            .out_dtype => |dt| try writer.print(" dtype={s}", .{@tagName(dt)}),
            .concat_axis => |axis| try writer.print(" axis={d}", .{axis}),
            .call_callee => |name| try writer.print(" callee=\"{s}\"", .{name}),
            .call_target_name => |name| try writer.print(" target=\"{s}\"", .{name}),
            .has_side_effect => |eff| {
                if (eff) try writer.writeAll(" side_effect");
            },
            .compare => |cp| try writer.print(" dir={s} type={s}", .{ @tagName(cp.direction), @tagName(cp.compare_type) }),
            .dot_general => |dg| {
                try writer.writeAll(" batch_lhs=[");
                try emit_i64_or_usize_list(i64, writer, dg.lhs_batch_dims);
                try writer.writeAll("] batch_rhs=[");
                try emit_i64_or_usize_list(i64, writer, dg.rhs_batch_dims);
                try writer.writeAll("] contract_lhs=[");
                try emit_i64_or_usize_list(i64, writer, dg.lhs_contracting_dims);
                try writer.writeAll("] contract_rhs=[");
                try emit_i64_or_usize_list(i64, writer, dg.rhs_contracting_dims);
                try writer.writeAll("]");
            },
            .slice => |s| {
                try writer.writeAll(" start=[");
                try emit_i64_or_usize_list(i64, writer, s.start_indices);
                try writer.writeAll("] limit=[");
                try emit_i64_or_usize_list(i64, writer, s.limit_indices);
                try writer.writeAll("]");
            },
            .call_kernel_key => |k| try writer.print(" kernel_key=\"{s}\"", .{k}),
            .call_kernel_id => |id| try writer.print(" kernel_id={d}", .{id}),
            .call_provider_name => |p| try writer.print(" provider=\"{s}\"", .{p}),
            .call_carrier_hint => |hint| try writer.print(" carrier=\"{s}\"", .{hint}),
            .gather, .scatter => {}, // Complex params; ZXPR format handles these
            .out_aval, .out_avals => {}, // Type already shown in output declaration
        }
    }
}

fn emit_i64_or_usize_list(comptime T: type, writer: anytype, items: []const T) !void {
    for (items, 0..) |v, i| {
        if (i > 0) try writer.writeAll(",");
        try writer.print("{d}", .{v});
    }
}

fn emit_literal(lit: pr.Literal, writer: anytype) !void {
    switch (lit) {
        .bf16 => |v| try writer.print("bf16(0x{x:0>4})", .{v}),
        inline .f32, .f64, .i32, .i64, .u32, .u64 => |v| try writer.print("{d}", .{v}),
        .bool => |v| try writer.writeAll(if (v) "true" else "false"),
    }
}

/// Convenience: emit to a heap-allocated buffer.
pub fn emit_text_alloc(allocator: std.mem.Allocator, func: pr.Function) ![]u8 {
    var buf: std.ArrayList(u8) = try .initCapacity(allocator, 256);
    errdefer buf.deinit(allocator);
    try emit_text(func, buf.writer(allocator));
    return try buf.toOwnedSlice(allocator);
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
