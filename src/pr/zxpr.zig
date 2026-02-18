///! ZXPR: Zigrad Expression Representation
///!
///! A human-readable text format for zigrad's Program Representation (PR).
///! Inspired by jaxpr.
///!
///! Features:
///! - Clean functional syntax with `let`/`in` bindings
///! - Op-specific attribute annotations
///! - AD-aware annotations (VJP support indicators)
///!
///! Example:
///!
///! zxpr main {
///!   ; params (2)
///!   a: 2x3<f32>
///!   b: 3x2<f32>
///!   ; body (1 ops)
///!   let
///!     c: 2x2<f32> = dot[contracting=([1], [0]), K=3](a, b)  ; vjp
///!   in c
///! }
const std = @import("std");
const pr = @import("pr.zig");
const ops = @import("ops/ops.zig");
const zxpr_style = @import("zxpr_style.zig");

const Writer = std.Io.Writer;

pub const Symbols = zxpr_style.Symbols;
pub const FormatConfig = zxpr_style.Config;
pub const FormatMode = zxpr_style.ConfigMode;
pub const FormatOpts = zxpr_style.ConfigOpts;
pub const format_config = zxpr_style.config;
pub const ShapeFormat = zxpr_style.ShapeFormat;
pub const Palette = zxpr_style.Palette;

const max_region_stack = 8;

// ============================================================================
// Emitter
// ============================================================================

pub const Emitter = struct {
    const Self = @This();

    writer: *Writer,
    func: pr.Function,
    indent: []const u8,
    styler: zxpr_style.Styler,

    pub fn init(writer: *Writer, func: pr.Function, cfg: FormatConfig) Self {
        return .{
            .writer = writer,
            .func = func,
            .indent = "  ",
            .styler = zxpr_style.Styler.init(writer, cfg),
        };
    }

    pub fn emit(self: *Self) !void {
        const w = self.writer;
        const ind = self.indent;

        try self.styler.write_keyword("zxpr");
        try w.writeAll(" ");
        try self.styler.write_var_name(self.func.name);
        try w.writeAll(" {\n");

        // Parameters section
        if (self.func.params.len > 0) {
            try w.writeAll(ind);
            try self.styler.write_section("; params");
            try w.print(" ({d})\n", .{self.func.params.len});
            for (self.func.params) |param_id| {
                try w.writeAll(ind);
                try self.emit_var_with_type(param_id);
                try w.writeAll("\n");
            }
        }

        // Body section
        if (self.func.eqns.len > 0) {
            try w.writeAll(ind);
            try self.styler.write_section("; body");
            try w.print(" ({d} ops)\n", .{self.func.eqns.len});
            try w.writeAll(ind);
            try self.styler.write_keyword("let");
            try w.writeAll("\n");

            // Track which regions are currently open (by index into func.regions)
            var open_stack: [max_region_stack]usize = undefined;
            var open_len: usize = 0;

            for (self.func.eqns, 0..) |eqn, eqn_idx| {
                // Close regions that end before this eqn
                while (open_len > 0) {
                    const ri = open_stack[open_len - 1];
                    const region = self.func.regions[ri];
                    if (eqn_idx >= region.eqn_start + region.eqn_len) {
                        open_len -= 1;
                        try self.emit_region_end(open_len);
                    } else break;
                }

                // Open regions that start at this eqn
                for (self.func.regions, 0..) |region, ri| {
                    if (region.eqn_start == eqn_idx) {
                        // Check not already open
                        var already = false;
                        for (open_stack[0..open_len]) |oi| {
                            if (oi == ri) {
                                already = true;
                                break;
                            }
                        }
                        if (!already and open_len < max_region_stack) {
                            try self.emit_region_start(open_len, region);
                            open_stack[open_len] = ri;
                            open_len += 1;
                        }
                    }
                }

                try w.writeAll(ind);
                try w.writeAll(ind);
                try self.emit_gutters(open_len);
                try self.emit_binding(eqn);
                try w.writeAll("\n");
            }

            // Close remaining open regions
            while (open_len > 0) {
                open_len -= 1;
                try self.emit_region_end(open_len);
            }
        }

        // Return section
        if (self.func.returns.len > 0) {
            if (self.func.eqns.len == 0) {
                try w.writeAll(ind);
                try w.writeAll(";\n");
            }
            try w.writeAll(ind);
            try self.styler.write_keyword("in");
            try w.writeAll(" ");
            for (self.func.returns, 0..) |ret_id, i| {
                if (i > 0) try w.writeAll(", ");
                try self.emit_var_name(ret_id);
            }
            try w.writeAll("\n");
        }

        try w.writeAll("}\n");
    }

    // ====================================================================
    // Helpers
    // ====================================================================

    fn emit_var_name(self: *Self, id: pr.VarId) !void {
        try self.styler.write_var_name(var_name(id));
    }

    fn emit_var_with_type(self: *Self, id: pr.VarId) !void {
        try self.emit_var_name(id);
        try self.writer.writeAll(": ");
        try self.emit_type(self.func.avals[@intCast(id)]);
    }

    fn emit_type(self: *Self, aval: pr.Aval) !void {
        switch (aval) {
            .tensor => |t| {
                switch (self.styler.cfg.shape_format) {
                    .dtype_suffix => {
                        if (t.shape.dims.len > 0) {
                            for (t.shape.dims, 0..) |d, i| {
                                if (i > 0) try self.writer.writeAll("x");
                                try self.writer.print("{d}", .{d});
                            }
                            try self.writer.writeAll("<");
                            try self.styler.write_type_name(@tagName(t.dtype));
                            try self.writer.writeAll(">");
                        } else {
                            try self.styler.write_type_name(@tagName(t.dtype));
                        }
                    },
                }
            },
        }
    }

    fn emit_binding(self: *Self, eqn: pr.Eqn) !void {
        const outputs = eqn.outputs.slice(pr.VarId, self.func.varids_store);
        const inputs = eqn.inputs.slice(pr.VarId, self.func.varids_store);
        const params = eqn.params.slice(pr.Param, self.func.params_store);

        // Output binding(s)
        for (outputs, 0..) |out_id, i| {
            if (i > 0) try self.writer.writeAll(", ");
            try self.emit_var_with_type(out_id);
        }

        try self.writer.writeAll(" = ");
        try self.styler.write_op_name(@tagName(eqn.prim));
        if (is_dtype_only_attr(eqn.prim) and !self.styler.cfg.include_dtype_attrs) {
            try self.writer.writeAll("(");
        } else {
            try self.writer.writeAll("[");
            if (is_dtype_only_attr(eqn.prim)) {
                _ = try self.emit_dtype_attr(eqn.prim, inputs, params);
            } else {
                try ops.format(self.writer, self.func, eqn);
            }
            try self.writer.writeAll("](");
        }

        // Inputs as function args
        for (inputs, 0..) |in_id, i| {
            if (i > 0) try self.writer.writeAll(", ");
            try self.emit_var_name(in_id);
        }

        try self.writer.writeAll(")");

        // VJP annotation
        if (ops.has_vjp(eqn.prim)) {
            try self.styler.write_comment("  ; vjp");
        }
    }

    fn emit_region_start(self: *Self, depth: usize, region: pr.Region) !void {
        try self.writer.writeAll(self.indent);
        try self.writer.writeAll(self.indent);
        try self.emit_gutters(depth);
        try self.styler.write_region(self.styler.cfg.symbols.region_start);
        try self.styler.write_region(" ");
        try self.styler.write_region(region.name);
        try self.styler.write_region("[");
        try self.emit_annotation(region.annotation);
        try self.styler.write_region("]\n");
    }

    fn emit_region_end(self: *Self, depth: usize) !void {
        try self.writer.writeAll(self.indent);
        try self.writer.writeAll(self.indent);
        try self.emit_gutters(depth);
        try self.styler.write_region(self.styler.cfg.symbols.region_end);
        try self.styler.write_region("\n");
    }

    fn emit_annotation(self: *Self, ann: pr.Annotation) !void {
        var first = true;
        if (ann.kernelize) |provider| {
            try self.styler.write_region("kernelize=");
            try self.styler.write_region(provider);
            first = false;
        }
        if (ann.outline) {
            if (!first) try self.styler.write_region(", ");
            try self.styler.write_region("outline");
        }
    }

    fn emit_gutters(self: *Self, count: usize) !void {
        var i: usize = 0;
        while (i < count) : (i += 1) {
            try self.styler.write_region(self.styler.cfg.symbols.region_gutter);
            try self.writer.writeAll(" ");
        }
    }

    fn emit_dtype_attr(self: *Self, prim: pr.Prim, inputs: []const pr.VarId, params: []const pr.Param) !?bool {
        const use_dtype = switch (prim) {
            .add, .subtract, .multiply, .divide, .maximum => true,
            .exp, .log, .rsqrt, .logistic, .convert => true,
            else => false,
        };
        if (!use_dtype) return null;
        if (inputs.len == 0) return false;
        const dtype = if (prim == .convert) blk: {
            const out_dtype = pr.param_out_dtype(params) orelse return false;
            break :blk out_dtype;
        } else blk: {
            const tensor = self.func.avals[@intCast(inputs[0])].as_tensor() orelse return false;
            break :blk tensor.dtype;
        };
        try self.writer.writeAll("dtype=");
        try self.styler.write_type_name(@tagName(dtype));
        return true;
    }
};

fn is_dtype_only_attr(prim: pr.Prim) bool {
    return switch (prim) {
        .add, .subtract, .multiply, .divide, .maximum => true,
        .exp, .log, .rsqrt, .logistic, .convert => true,
        else => false,
    };
}

// ============================================================================
// Helpers
// ============================================================================

/// Generate readable variable name from ID: 0->a, 1->b, ..., 26->aa, etc.
fn var_name(id: pr.VarId) []const u8 {
    const names = comptime blk: {
        @setEvalBranchQuota(20000);
        const single = 26;
        const double = 26 * 26;
        const triple = 26 * 26 * 26;
        const total = single + double + triple;
        var arr: [total][]const u8 = undefined;
        for (0..total) |i| {
            if (i < single) {
                arr[i] = &[_]u8{'a' + @as(u8, @intCast(i))};
            } else if (i < single + double) {
                const idx = i - single;
                const first = 'a' + @as(u8, @intCast(idx / 26));
                const second = 'a' + @as(u8, @intCast(idx % 26));
                arr[i] = &[_]u8{ first, second };
            } else {
                const idx = i - single - double;
                const first = 'a' + @as(u8, @intCast(idx / (26 * 26)));
                const second = 'a' + @as(u8, @intCast((idx / 26) % 26));
                const third = 'a' + @as(u8, @intCast(idx % 26));
                arr[i] = &[_]u8{ first, second, third };
            }
        }
        break :blk arr;
    };
    if (id < names.len) return names[id];
    return "???"; // Fallback for very large programs
}

// ============================================================================
// Public API
// ============================================================================

/// Emit ZXPR representation of a function.
pub fn emit(func: pr.Function, writer: *Writer, mode: FormatMode, opts: FormatOpts) !void {
    const cfg = format_config(mode, opts);
    var emitter = Emitter.init(writer, func, cfg);
    try emitter.emit();
}

// ============================================================================
// Tests
// ============================================================================

test "zxpr format" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a = try b.param_tensor(.f32, &.{ 2, 3 });
    const c = try b.param_tensor(.f32, &.{ 3, 2 });
    const d = try b.dot(a, c);
    const func = try b.finish(&.{d});

    var buf: [512]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try emit(func, &w, .plain, .{});

    const result = w.buffered();
    try std.testing.expect(std.mem.indexOf(u8, result, "zxpr main {") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "; params (2)") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "a: 2x3<f32>") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "let\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "dot[contracting=([1], [0])") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "; vjp") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "in c") != null);
}

test "zxpr variable naming" {
    try std.testing.expectEqualStrings("a", var_name(0));
    try std.testing.expectEqualStrings("b", var_name(1));
    try std.testing.expectEqualStrings("z", var_name(25));
    try std.testing.expectEqualStrings("aa", var_name(26));
    try std.testing.expectEqualStrings("ab", var_name(27));
    try std.testing.expectEqualStrings("zz", var_name(701));
    try std.testing.expectEqualStrings("aaa", var_name(702));
}

test "zxpr with transpose shows permutation" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "t");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.transpose(x, &.{ 1, 0 });
    const func = try b.finish(&.{y});

    var buf: [256]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try emit(func, &w, .plain, .{});

    const result = w.buffered();
    try std.testing.expect(std.mem.indexOf(u8, result, "transpose[perm=[1, 0]]") != null);
}

test "zxpr kernelize region annotations" {
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

    const out = try b.add(add2, c);
    const func = try b.finish(&.{out});

    var buf: [512]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try emit(func, &w, .plain, .{});

    const result = w.buffered();
    try std.testing.expect(std.mem.indexOf(u8, result, "> tvm-kernel[kernelize=tvm]") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "<") != null);
}
