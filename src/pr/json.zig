///! JSON graph emitter for PR.
///!
///! Graph model: nodes = ops (params + equations), edges = vars (typed data flow).
///!
///! Each equation becomes one node regardless of output count. Each variable
///! flowing between ops becomes an edge carrying the var's name, dtype, shape,
///! and ZXPR snippet. This matches the PR's SSA structure where variables are
///! the named, typed connections between operations.
///!
///! Param serialization: each emitter (zxpr via ops.format, this file)
///! necessarily renders the same Param union in a different syntax.
const std = @import("std");
const pr = @import("pr.zig");
const ops = @import("ops/ops.zig");
const zxpr = @import("zxpr/root.zig");

const Writer = std.Io.Writer;
const var_name = zxpr.var_name;

/// Emit a Function as a JSON graph.
///
/// Nodes represent ops (params and equations). Edges represent variables.
/// A param node has one outgoing var edge per consumer. An equation node
/// has outgoing var edges for each of its output variables.
pub fn emit(func: pr.Function, writer: *Writer) !void {
    try writer.writeAll("{");

    // "name"
    try writer.writeAll("\"name\":");
    try write_json_string(writer, func.name);

    // Build producer map: VarId -> node id string
    // Params get "p0", "p1", ...; equations get "e0", "e1", ...
    // Also track which VarIds are consumed as inputs (for edge generation).

    // "nodes"
    try writer.writeAll(",\"nodes\":[");
    var node_idx: usize = 0;

    for (func.params, 0..) |param_id, pi| {
        if (node_idx > 0) try writer.writeAll(",");
        try emit_param_node(writer, func, param_id, pi);
        node_idx += 1;
    }

    for (func.eqns, 0..) |eqn, ei| {
        if (node_idx > 0) try writer.writeAll(",");
        try emit_eqn_node(writer, func, eqn, ei);
        node_idx += 1;
    }
    try writer.writeAll("]");

    // Pre-build VarId -> producer lookup (O(1) per edge instead of O(N) scan).
    var producer_buf: [4096]u32 = undefined;
    const producer_map = producer_buf[0..@min(func.avals.len, producer_buf.len)];
    buildProducerMap(func, producer_map);

    // "edges" - each variable flowing between ops
    // For each equation, its inputs are vars produced by earlier ops.
    // Edge source = producing op, edge target = consuming op.
    try writer.writeAll(",\"edges\":[");
    var edge_idx: usize = 0;
    for (func.eqns, 0..) |eqn, ei| {
        const input_ids = eqn.inputs.slice(pr.VarId, func.varids_store);
        for (input_ids, 0..) |in_id, port| {
            if (edge_idx > 0) try writer.writeAll(",");
            try emit_var_edge(writer, func, in_id, ei, port, producer_map);
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
    for (func.returns, 0..) |ret_id, i| {
        if (i > 0) try writer.writeAll(",");
        try write_json_string(writer, var_name(ret_id));
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

/// Op node id: "p0", "p1", ... for params; "e0", "e1", ... for equations.
fn op_node_id(writer: *Writer, prefix: []const u8, index: usize) !void {
    try writer.writeAll("\"");
    try writer.writeAll(prefix);
    try writer.print("{d}", .{index});
    try writer.writeAll("\"");
}

fn emit_param_node(writer: *Writer, func: pr.Function, id: pr.VarId, param_index: usize) !void {
    try writer.writeAll("{\"id\":");
    try op_node_id(writer, "p", param_index);
    try writer.writeAll(",\"kind\":\"param\"");
    // Label: var name for readability
    try writer.writeAll(",\"label\":");
    try write_json_string(writer, var_name(id));
    try emit_zxpr_field(writer, func, .{ .param = id });
    try writer.writeAll("}");
}

fn emit_eqn_node(writer: *Writer, func: pr.Function, eqn: pr.Eqn, eqn_index: usize) !void {
    try writer.writeAll("{\"id\":");
    try op_node_id(writer, "e", eqn_index);
    try writer.writeAll(",\"kind\":");
    try write_json_string(writer, @tagName(eqn.prim));
    try emit_param_attrs(writer, eqn.params.slice(pr.Param, func.params_store));
    if (ops.has_vjp(eqn.prim)) {
        try writer.writeAll(",\"vjp\":true");
    }
    // Label: "out_var = prim" (first output for display)
    const output_ids = eqn.outputs.slice(pr.VarId, func.varids_store);
    if (output_ids.len > 0) {
        try writer.writeAll(",\"label\":\"");
        try writer.writeAll(var_name(output_ids[0]));
        try writer.writeAll(" = ");
        try writer.writeAll(@tagName(eqn.prim));
        try writer.writeAll("\"");
    }
    try emit_zxpr_field(writer, func, .{ .eqn = eqn });
    try writer.writeAll("}");
}

// ============================================================================
// Edge emitter - edges represent variables
// ============================================================================

/// Packed producer reference: bit 31 selects param (0) or eqn (1), bits 0..30 hold the index.
/// `no_producer` sentinel means the VarId has no known producer.
const no_producer: u32 = std.math.maxInt(u32);
const eqn_flag: u32 = 1 << 31;

fn buildProducerMap(func: pr.Function, map: []u32) void {
    @memset(map, no_producer);
    for (func.params, 0..) |pid, pi| {
        map[@intCast(pid)] = @intCast(pi);
    }
    for (func.eqns, 0..) |eqn, ei| {
        for (eqn.outputs.slice(pr.VarId, func.varids_store)) |out_id| {
            map[@intCast(out_id)] = eqn_flag | @as(u32, @intCast(ei));
        }
    }
}

fn lookupProducer(map: []const u32, var_id: pr.VarId) ?struct { id_prefix: []const u8, index: usize } {
    const encoded = map[@intCast(var_id)];
    if (encoded == no_producer) return null;
    return if (encoded & eqn_flag != 0)
        .{ .id_prefix = "e", .index = @intCast(encoded & 0x7FFF_FFFF) }
    else
        .{ .id_prefix = "p", .index = @intCast(encoded) };
}

fn emit_var_edge(writer: *Writer, func: pr.Function, var_id: pr.VarId, target_eqn_idx: usize, port: usize, producer_map: []const u32) !void {
    const producer = lookupProducer(producer_map, var_id) orelse return;

    try writer.writeAll("{\"source\":");
    try op_node_id(writer, producer.id_prefix, producer.index);
    try writer.writeAll(",\"target\":");
    try op_node_id(writer, "e", target_eqn_idx);
    try writer.print(",\"port\":{d}", .{port});

    // Var identity
    try writer.writeAll(",\"var\":");
    try write_json_string(writer, var_name(var_id));

    // Var type
    try emit_aval_fields(writer, func.avals[@intCast(var_id)]);

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

const ZxprTarget = union(enum) {
    param: pr.VarId,
    eqn: pr.Eqn,
    region: pr.Region,
};

fn emit_zxpr_field(writer: *Writer, func: pr.Function, target: ZxprTarget) !void {
    var buf: [512]u8 = undefined;
    var zw: Writer = .fixed(&buf);
    switch (target) {
        .param => |id| zxpr.emit_param_line(func, id, &zw) catch return,
        .eqn => |eqn| zxpr.emit_eqn_line(func, eqn, &zw) catch return,
        .region => |region| zxpr.emit_region_block(func, region, &zw) catch return,
    }
    const text = zw.buffered();
    if (text.len > 0) {
        try writer.writeAll(",\"zxpr\":");
        try write_json_string(writer, text);
    }
}

// ============================================================================
// Param -> JSON attrs
//
// Each output format (zxpr via ops.format, this JSON emitter)
// walks the Param union with format-specific syntax. The exhaustive switch
// ensures new Param variants cause a compile error here.
// ============================================================================

fn emit_param_attrs(writer: *Writer, params: []const pr.Param) !void {
    var has_attr = false;
    for (params) |param| {
        switch (param) {
            .literal => |lit| {
                try open_attrs(writer, &has_attr);
                try writer.writeAll("\"value\":");
                try emit_literal_json(writer, lit);
            },
            .out_shape => |shape| {
                try open_attrs(writer, &has_attr);
                try writer.writeAll("\"out_shape\":");
                try emit_usize_array(writer, shape);
            },
            .broadcast_dimensions => |dims| {
                try open_attrs(writer, &has_attr);
                try writer.writeAll("\"broadcast_dims\":");
                try emit_i64_array(writer, dims);
            },
            .permutation => |perm| {
                try open_attrs(writer, &has_attr);
                try writer.writeAll("\"permutation\":");
                try emit_i64_array(writer, perm);
            },
            .reduce_axes => |axes| {
                try open_attrs(writer, &has_attr);
                try writer.writeAll("\"reduce_axes\":");
                try emit_i64_array(writer, axes);
            },
            .iota_dimension => |dim| {
                try open_attrs(writer, &has_attr);
                try writer.print("\"iota_dim\":{d}", .{dim});
            },
            .out_dtype => |dt| {
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
            .concat_axis => |axis| {
                try open_attrs(writer, &has_attr);
                try writer.print("\"axis\":{d}", .{axis});
            },
            .call_callee => |name| {
                try open_attrs(writer, &has_attr);
                try writer.writeAll("\"callee\":");
                try write_json_string(writer, name);
            },
            .call_target_name => |name| {
                try open_attrs(writer, &has_attr);
                try writer.writeAll("\"target\":");
                try write_json_string(writer, name);
            },
            .call_kernel_key => |k| {
                try open_attrs(writer, &has_attr);
                try writer.writeAll("\"kernel_key\":");
                try write_json_string(writer, k);
            },
            .call_provider_name => |p| {
                try open_attrs(writer, &has_attr);
                try writer.writeAll("\"provider\":");
                try write_json_string(writer, p);
            },
            .call_carrier_hint => |h| {
                try open_attrs(writer, &has_attr);
                try writer.writeAll("\"carrier\":");
                try write_json_string(writer, h);
            },
            .has_side_effect => |eff| {
                if (eff) {
                    try open_attrs(writer, &has_attr);
                    try writer.writeAll("\"side_effect\":true");
                }
            },
            // Type info already captured in shape/dtype fields on the node
            .out_aval, .out_avals => {},
        }
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
    const eqn_end = region.eqn_start + region.eqn_len;
    var first = true;
    var eqn_i: u32 = region.eqn_start;
    while (eqn_i < eqn_end and eqn_i < func.eqns.len) : (eqn_i += 1) {
        if (!first) try writer.writeAll(",");
        try writer.print("\"e{d}\"", .{eqn_i});
        first = false;
    }
    try writer.writeAll("]");

    try emit_zxpr_field(writer, func, .{ .region = region });
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

fn emit_usize_array(writer: *Writer, items: []const usize) !void {
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
        .bf16 => |v| try writer.print("{d}", .{@as(f32, @bitCast(@as(u32, v) << 16))}),
        .bool => |v| try writer.writeAll(if (v) "true" else "false"),
        inline .f32, .f64 => |v| try writer.print("{d}", .{v}),
        inline .i32, .i64, .u32, .u64 => |v| try writer.print("{d}", .{v}),
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

    // Nodes are ops: params "p0","p1", equation "e0"
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

    // 3 params + 3 equations = 6 op nodes
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
