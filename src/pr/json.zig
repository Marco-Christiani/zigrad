//! JSON graph emitter for PR.
//!
//! Graph model: nodes = ops (params + equations), edges = vars (typed data flow).
//!
//! Each op becomes one node regardless of output count. Each variable
//!  flowing between ops becomes an edge carrying the var's name, dtype, shape,
//!  and ZXPR snippet. This matches the PR's SSA structure where variables are
//!  the named, typed connections between operations.
const std = @import("std");
const pr = @import("pr.zig");
const ops = @import("ops/ops.zig");
const zxpr = @import("zxpr/root.zig");

const Writer = std.Io.Writer;
const var_name = zxpr.var_name;

/// Emit a Function as a JSON graph.
///
/// Nodes represent ops (params and equations). Edges represent variables.
/// A param node has one outgoing var edge per consumer. An op node
///  has outgoing var edges for each of its output variables.
pub fn emit(func: pr.Function, writer: *Writer) !void {
    try writer.writeAll("{");

    // "name"
    try writer.writeAll("\"name\":");
    try write_json_string(writer, func.name);

    // "nodes"
    try writer.writeAll(",\"nodes\":[");
    var node_idx: usize = 0;

    for (func.params, 0..) |param_var, pi| {
        if (node_idx > 0) try writer.writeAll(",");
        try emit_param_node(writer, param_var, pi);
        node_idx += 1;
    }

    for (func.ops, 0..) |op, oi| {
        if (node_idx > 0) try writer.writeAll(",");
        try emit_op_node(writer, op, oi);
        node_idx += 1;
    }
    try writer.writeAll("]");

    // Pre-build var.id -> producer lookup (O(1) per edge instead of O(N) scan).
    var producer_buf: [4096]u32 = undefined;
    const map_len = @min(func.var_count, producer_buf.len);
    const producer_map = producer_buf[0..map_len];
    buildProducerMap(func, producer_map);

    // "edges" - each variable flowing between ops
    try writer.writeAll(",\"edges\":[");
    var edge_idx: usize = 0;
    for (func.ops, 0..) |op, oi| {
        for (op.inputs, 0..) |operand, port| {
            if (edge_idx > 0) try writer.writeAll(",");
            try emit_var_edge(writer, operand.value, oi, port, producer_map);
            edge_idx += 1;
        }
    }
    try writer.writeAll("]");

    // "regions"
    try writer.writeAll(",\"regions\":[");
    for (func.regions, 0..) |region, ri| {
        if (ri > 0) try writer.writeAll(",");
        try emit_region(writer, func, region);
    }
    try writer.writeAll("]");

    // "returns"
    try writer.writeAll(",\"returns\":[");
    for (func.returns, 0..) |ret_var, i| {
        if (i > 0) try writer.writeAll(",");
        try write_json_string(writer, var_name(ret_var.id));
    }
    try writer.writeAll("]");

    try writer.writeAll("}\n");
}

/// Emit a Program as a JSON array of function graphs.
pub fn emit_program(program: *const pr.Program, writer: *Writer) !void {
    if (program.functions.len == 1) {
        try emit(program.functions[0], writer);
        return;
    }
    try writer.writeAll("[");
    for (program.functions, 0..) |func, i| {
        if (i > 0) try writer.writeAll(",");
        try emit(func, writer);
    }
    try writer.writeAll("]\n");
}

// ============================================================================
// Node emitters - nodes represent ops
// ============================================================================

/// Op node id: "p0", "p1", ... for params; "e0", "e1", ... for ops.
fn op_node_id(writer: *Writer, prefix: []const u8, index: usize) !void {
    try writer.writeAll("\"");
    try writer.writeAll(prefix);
    try writer.print("{d}", .{index});
    try writer.writeAll("\"");
}

fn emit_param_node(writer: *Writer, v: *const pr.Var, param_index: usize) !void {
    try writer.writeAll("{\"id\":");
    try op_node_id(writer, "p", param_index);
    try writer.writeAll(",\"kind\":\"param\"");
    try writer.writeAll(",\"label\":");
    try write_json_string(writer, var_name(v.id));
    try emit_zxpr_field_param(writer, v);
    try writer.writeAll("}");
}

fn emit_op_node(writer: *Writer, op: *const pr.Op, op_index: usize) !void {
    const prim = op.prim();
    try writer.writeAll("{\"id\":");
    try op_node_id(writer, "e", op_index);
    try writer.writeAll(",\"kind\":");
    try write_json_string(writer, @tagName(prim));
    try emit_param_attrs(writer, op);
    if (ops.has_vjp(prim)) {
        try writer.writeAll(",\"vjp\":true");
    }
    // Label: "out_var = prim" (first output for display)
    if (op.outputs.len > 0) {
        try writer.writeAll(",\"label\":\"");
        try writer.writeAll(var_name(op.outputs[0].id));
        try writer.writeAll(" = ");
        try writer.writeAll(@tagName(prim));
        try writer.writeAll("\"");
    }
    try emit_zxpr_field_op(writer, op);
    try writer.writeAll("}");
}

// ============================================================================
// Edge emitter - edges represent variables
// ============================================================================

/// Packed producer reference: bit 31 selects param (0) or op (1), bits 0..30 hold the index.
/// `no_producer` sentinel means the var has no known producer.
const no_producer: u32 = std.math.maxInt(u32);
const op_flag: u32 = 1 << 31;

fn buildProducerMap(func: pr.Function, map: []u32) void {
    @memset(map, no_producer);
    for (func.params, 0..) |param_var, pi| {
        if (param_var.id < map.len) {
            map[param_var.id] = @intCast(pi);
        }
    }
    for (func.ops, 0..) |op, oi| {
        for (op.outputs) |out_var| {
            if (out_var.id < map.len) {
                map[out_var.id] = op_flag | @as(u32, @intCast(oi));
            }
        }
    }
}

fn lookupProducer(map: []const u32, var_id: u32) ?struct { id_prefix: []const u8, index: usize } {
    if (var_id >= map.len) return null;
    const encoded = map[var_id];
    if (encoded == no_producer) return null;
    return if (encoded & op_flag != 0)
        .{ .id_prefix = "e", .index = @intCast(encoded & 0x7FFF_FFFF) }
    else
        .{ .id_prefix = "p", .index = @intCast(encoded) };
}

fn emit_var_edge(writer: *Writer, v: *const pr.Var, target_op_idx: usize, port: usize, producer_map: []const u32) !void {
    const producer = lookupProducer(producer_map, v.id) orelse return;

    try writer.writeAll("{\"source\":");
    try op_node_id(writer, producer.id_prefix, producer.index);
    try writer.writeAll(",\"target\":");
    try op_node_id(writer, "e", target_op_idx);
    try writer.print(",\"port\":{d}", .{port});

    // Var identity
    try writer.writeAll(",\"var\":");
    try write_json_string(writer, var_name(v.id));

    // Var type
    try emit_aval_fields(writer, v.aval);

    try writer.writeAll("}");
}

fn emit_aval_fields(writer: *Writer, aval: pr.Aval) !void {
    switch (aval) {
        .tensor => |t| {
            try writer.writeAll(",\"dtype\":");
            try write_json_string(writer, @tagName(t.dtype));
            try writer.writeAll(",\"shape\":[");
            for (t.shape.dims, 0..) |d, i| {
                if (i > 0) try writer.writeAll(",");
                try writer.print("{d}", .{d});
            }
            try writer.writeAll("]");
        },
    }
}

// ============================================================================
// ZXPR snippet embedding
// ============================================================================

fn emit_zxpr_field_param(writer: *Writer, v: *const pr.Var) !void {
    var buf: [512]u8 = undefined;
    var zw: Writer = .fixed(&buf);
    zxpr.emit_param_line(v, &zw) catch return;
    const text = zw.buffered();
    if (text.len > 0) {
        try writer.writeAll(",\"zxpr\":");
        try write_json_string(writer, text);
    }
}

fn emit_zxpr_field_op(writer: *Writer, op: *const pr.Op) !void {
    var buf: [512]u8 = undefined;
    var zw: Writer = .fixed(&buf);
    zxpr.emit_op_line(op, &zw) catch return;
    const text = zw.buffered();
    if (text.len > 0) {
        try writer.writeAll(",\"zxpr\":");
        try write_json_string(writer, text);
    }
}

fn emit_zxpr_field_region(writer: *Writer, func: pr.Function, region: pr.Region) !void {
    var buf: [512]u8 = undefined;
    var zw: Writer = .fixed(&buf);
    zxpr.emit_region_block(func, region, &zw) catch return;
    const text = zw.buffered();
    if (text.len > 0) {
        try writer.writeAll(",\"zxpr\":");
        try write_json_string(writer, text);
    }
}

// ============================================================================
// Params -> JSON attrs
//
// Walks the typed Params union and emits format-specific JSON attributes.
// The exhaustive switch ensures new Params variants cause a compile error.
// ============================================================================

fn emit_param_attrs(writer: *Writer, op: *const pr.Op) !void {
    var has_attr = false;
    switch (op.params) {
        // No-param ops
        .add, .subtract, .multiply, .divide, .maximum => {},
        .exp, .log, .rsqrt, .logistic => {},
        .select, .dot => {},

        .literal => |lit| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"value\":");
            try emit_literal_json(writer, lit);
        },
        .reshape => |rp| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"out_shape\":");
            try emit_i64_array(writer, rp.out_shape);
        },
        .broadcast_in_dim => |bp| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"out_shape\":");
            try emit_i64_array(writer, bp.out_shape);
            try writer.writeAll(",\"broadcast_dims\":");
            try emit_i64_array(writer, bp.dimensions);
        },
        .transpose => |tp| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"permutation\":");
            try emit_i64_array(writer, tp.permutation);
        },
        .reduce_sum => |rp| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"reduce_axes\":");
            try emit_i64_array(writer, rp.axes);
        },
        .reduce_max => |rp| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"reduce_axes\":");
            try emit_i64_array(writer, rp.axes);
        },
        .iota => |ip| {
            try open_attrs(writer, &has_attr);
            try writer.print("\"iota_dim\":{d}", .{ip.dimension});
            try writer.writeAll(",\"out_shape\":");
            try emit_i64_array(writer, ip.out_shape);
            try writer.writeAll(",\"out_dtype\":");
            try write_json_string(writer, @tagName(ip.out_dtype));
        },
        .convert => |dt| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"out_dtype\":");
            try write_json_string(writer, @tagName(dt));
        },
        .compare => |cp| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"direction\":");
            try write_json_string(writer, @tagName(cp.direction));
            try writer.writeAll(",\"compare_type\":");
            try write_json_string(writer, @tagName(cp.compare_type));
        },
        .dot_general => |dg| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"lhs_contracting\":");
            try emit_i64_array(writer, dg.lhs_contracting_dims);
            try writer.writeAll(",\"rhs_contracting\":");
            try emit_i64_array(writer, dg.rhs_contracting_dims);
            try writer.writeAll(",\"lhs_batch\":");
            try emit_i64_array(writer, dg.lhs_batch_dims);
            try writer.writeAll(",\"rhs_batch\":");
            try emit_i64_array(writer, dg.rhs_batch_dims);
        },
        .gather => |g| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"slice_sizes\":");
            try emit_i64_array(writer, g.slice_sizes);
            try writer.writeAll(",\"offset_dims\":");
            try emit_i64_array(writer, g.offset_dims);
            try writer.writeAll(",\"collapsed_slice_dims\":");
            try emit_i64_array(writer, g.collapsed_slice_dims);
            try writer.writeAll(",\"start_index_map\":");
            try emit_i64_array(writer, g.start_index_map);
            try writer.print(",\"index_vector_dim\":{d}", .{g.index_vector_dim});
        },
        .scatter => |s| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"update_window_dims\":");
            try emit_i64_array(writer, s.update_window_dims);
            try writer.writeAll(",\"inserted_window_dims\":");
            try emit_i64_array(writer, s.inserted_window_dims);
            try writer.writeAll(",\"scatter_dims_to_operand_dims\":");
            try emit_i64_array(writer, s.scatter_dims_to_operand_dims);
            try writer.print(",\"index_vector_dim\":{d}", .{s.index_vector_dim});
            try writer.writeAll(",\"reduction\":");
            try write_json_string(writer, @tagName(s.reduction));
        },
        .slice => |s| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"start\":");
            try emit_i64_array(writer, s.start_indices);
            try writer.writeAll(",\"limit\":");
            try emit_i64_array(writer, s.limit_indices);
            try writer.writeAll(",\"strides\":");
            try emit_i64_array(writer, s.strides);
        },
        .concatenate => |cp| {
            try open_attrs(writer, &has_attr);
            try writer.print("\"axis\":{d}", .{cp.axis});
        },
        .call => |cp| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"callee\":");
            try write_json_string(writer, cp.callee);
        },
        .custom_call => |cc| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"target\":");
            try write_json_string(writer, cc.target_name);
            if (cc.kernel_key) |k| {
                try writer.writeAll(",\"kernel_key\":");
                try write_json_string(writer, k);
            }
            if (cc.provider_name) |p| {
                try writer.writeAll(",\"provider\":");
                try write_json_string(writer, p);
            }
            if (cc.has_side_effect) {
                try writer.writeAll(",\"side_effect\":true");
            }
        },
    }
    if (has_attr) try writer.writeAll("}");
}

fn open_attrs(writer: *Writer, has_attr: *bool) !void {
    if (!has_attr.*) {
        try writer.writeAll(",\"attrs\":{");
        has_attr.* = true;
    } else {
        try writer.writeAll(",");
    }
}

// ============================================================================
// Region emitter
// ============================================================================

fn emit_region(writer: *Writer, func: pr.Function, region: pr.Region) !void {
    try writer.writeAll("{\"name\":");
    try write_json_string(writer, region.name);

    if (region.annotation.kernelize) |provider| {
        try writer.writeAll(",\"kernelize\":");
        try write_json_string(writer, provider);
    }
    if (region.annotation.outline) {
        try writer.writeAll(",\"outline\":true");
    }

    // node_ids reference op node ids (e0, e1, ...)
    try writer.writeAll(",\"node_ids\":[");
    const op_end = region.op_start + region.op_len;
    var first = true;
    var op_i: u32 = region.op_start;
    while (op_i < op_end and op_i < func.ops.len) : (op_i += 1) {
        if (!first) try writer.writeAll(",");
        try writer.print("\"e{d}\"", .{op_i});
        first = false;
    }
    try writer.writeAll("]");

    try emit_zxpr_field_region(writer, func, region);
    try writer.writeAll("}");
}

// ============================================================================
// JSON primitives
// ============================================================================

fn write_json_string(writer: *Writer, s: []const u8) !void {
    try writer.writeAll("\"");
    var start: usize = 0;
    for (s, 0..) |c, i| {
        switch (c) {
            '"', '\\', 0x00...0x1f => {
                if (i > start) try writer.writeAll(s[start..i]);
                switch (c) {
                    '"' => try writer.writeAll("\\\""),
                    '\\' => try writer.writeAll("\\\\"),
                    '\n' => try writer.writeAll("\\n"),
                    '\r' => try writer.writeAll("\\r"),
                    '\t' => try writer.writeAll("\\t"),
                    else => try writer.print("\\u{x:0>4}", .{@as(u16, c)}),
                }
                start = i + 1;
            },
            else => {},
        }
    }
    if (start < s.len) try writer.writeAll(s[start..]);
    try writer.writeAll("\"");
}

fn emit_i64_array(writer: *Writer, items: []const i64) !void {
    try writer.writeAll("[");
    for (items, 0..) |v, i| {
        if (i > 0) try writer.writeAll(",");
        try writer.print("{d}", .{v});
    }
    try writer.writeAll("]");
}

/// JSON literal values. bf16 is widened to f32 for JSON compatibility.
fn emit_literal_json(writer: *Writer, lit: pr.Literal) !void {
    switch (lit) {
        .f16 => |v| try writer.print("{d}", .{pr.DType.f16.decode(f32, v)}),
        .bf16 => |v| try writer.print("{d}", .{pr.DType.bf16.decode(f32, v)}),
        .bool => |v| try writer.writeAll(if (v) "true" else "false"),
        inline .f32, .f64 => |v| try writer.print("{d}", .{v}),
        inline .i8, .u8, .i32, .i64, .u32, .u64 => |v| try writer.print("{d}", .{v}),
    }
}

// ============================================================================
// Tests
// ============================================================================

test emit {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a = try b.param_tensor(.f32, &.{ 2, 3 });
    const c = try b.param_tensor(.f32, &.{ 3, 2 });
    const d = try b.dot(a, c);
    const func = try b.finish(&.{d});

    var buf: [2048]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try emit(func, &w);

    const result = w.buffered();

    try std.testing.expect(result.len > 0);
    try std.testing.expect(result[0] == '{');
    try std.testing.expect(std.mem.indexOf(u8, result, "\"name\":\"main\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"nodes\":[") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"edges\":[") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"returns\":[\"c\"]") != null);

    // Nodes are ops: params "p0","p1", op "e0"
    try std.testing.expect(std.mem.indexOf(u8, result, "\"id\":\"p0\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"id\":\"p1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"id\":\"e0\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"kind\":\"param\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"kind\":\"dot\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"vjp\":true") != null);

    // Param nodes have var-name labels
    try std.testing.expect(std.mem.indexOf(u8, result, "\"label\":\"a\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"label\":\"b\"") != null);

    // Edges are vars: carry var name, dtype, shape
    try std.testing.expect(std.mem.indexOf(u8, result, "\"source\":\"p0\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"source\":\"p1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"target\":\"e0\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"var\":\"a\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"var\":\"b\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"shape\":[2,3]") != null);

    // ZXPR snippets on nodes
    try std.testing.expect(std.mem.indexOf(u8, result, "\"zxpr\":\"a: 2x3<f32>\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "dot[contracting") != null);
}

test "json with regions" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "k");
    defer b.deinit();

    const a = try b.param_tensor(.f32, &.{ 2, 2 });
    const c = try b.param_tensor(.f32, &.{ 2, 2 });

    try b.push_region("tvm-kernel", .{ .kernelize = "tvm" });
    const add1 = try b.add(a, c);
    const add2 = try b.add(add1, c);
    try b.pop_region();

    _ = try b.add(add2, c);
    const func = try b.finish(&.{add2});

    var buf: [4096]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try emit(func, &w);

    const result = w.buffered();

    try std.testing.expect(std.mem.indexOf(u8, result, "\"kernelize\":\"tvm\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"name\":\"tvm-kernel\"") != null);
    // Region node_ids now reference op nodes (e0, e1)
    try std.testing.expect(std.mem.indexOf(u8, result, "\"node_ids\":[\"e0\",\"e1\"]") != null);
}

test "json with reshape and transpose" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "shapes");
    defer b.deinit();

    const x = try b.param_tensor(.f64, &.{ 2, 3 });
    const t = try b.transpose(x, &.{ 1, 0 });
    const r = try b.reshape(t, &.{6});
    const func = try b.finish(&.{r});

    var buf: [2048]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try emit(func, &w);

    const result = w.buffered();

    // Op nodes
    try std.testing.expect(std.mem.indexOf(u8, result, "\"kind\":\"transpose\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"permutation\":[1,0]") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"kind\":\"reshape\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"out_shape\":[6]") != null);
    // Var edges carry type info
    try std.testing.expect(std.mem.indexOf(u8, result, "\"var\":\"a\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"dtype\":\"f64\"") != null);
}

test "json with broadcast and literal" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "bcast");
    defer b.deinit();

    const one = try b.literal_scalar(.{ .f32 = 1.0 });
    const broadcasted = try b.broadcast_in_dim(one, &.{ 2, 3 }, &.{});
    const func = try b.finish(&.{broadcasted});

    var buf: [2048]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try emit(func, &w);

    const result = w.buffered();

    // Op nodes
    try std.testing.expect(std.mem.indexOf(u8, result, "\"kind\":\"literal\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"kind\":\"broadcast_in_dim\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"broadcast_dims\":[]") != null);
    // Literal is e0, broadcast is e1; edge from e0 -> e1 carries var "a"
    try std.testing.expect(std.mem.indexOf(u8, result, "\"source\":\"e0\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"target\":\"e1\"") != null);
}

test "json output is valid JSON" {
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

    var buf: [8192]u8 = undefined;
    var wr: Writer = .fixed(&buf);
    try emit(func, &wr);

    const result = wr.buffered();

    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, result, .{});
    defer parsed.deinit();

    const root = parsed.value.object;
    try std.testing.expectEqualStrings("matmul_add", root.get("name").?.string);

    // 3 params + 3 ops = 6 nodes
    const nodes = root.get("nodes").?.array;
    try std.testing.expectEqual(6, nodes.items.len);

    // Edges = vars flowing between ops
    const edges = root.get("edges").?.array;
    try std.testing.expect(edges.items.len > 0);

    // Each edge has var name, dtype, shape
    const first_edge = edges.items[0].object;
    try std.testing.expect(first_edge.get("var") != null);
    try std.testing.expect(first_edge.get("dtype") != null);
    try std.testing.expect(first_edge.get("shape") != null);

    const returns = root.get("returns").?.array;
    try std.testing.expectEqual(1, returns.items.len);
}

test emit_program {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b1 = try pr.FunctionBuilder.init(&program, "f1");
    defer b1.deinit();
    const x = try b1.param_tensor(.f32, &.{4});
    const func1 = try b1.finish(&.{x});
    try program.add_function(func1);

    var b2 = try pr.FunctionBuilder.init(&program, "f2");
    defer b2.deinit();
    const y = try b2.param_tensor(.f32, &.{8});
    const func2 = try b2.finish(&.{y});
    try program.add_function(func2);

    var buf: [4096]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try emit_program(&program, &w);

    const result = w.buffered();

    try std.testing.expect(result[0] == '[');
    try std.testing.expect(std.mem.indexOf(u8, result, "\"name\":\"f1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"name\":\"f2\"") != null);
}
