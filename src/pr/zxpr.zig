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

const max_region_tags = 4;
const max_region_stack = 8;

const RegionTag = struct {
    key: []const u8,
    value: ?[]const u8,
};

const RegionSpec = struct {
    tags: [max_region_tags]RegionTag = undefined,
    len: usize = 0,
};

const RegionStackSpec = struct {
    items: [max_region_stack]RegionSpec = undefined,
    len: usize = 0,
};

const Region = struct {
    name_buf: [24]u8,
    name_len: usize,
    spec: RegionSpec,
};

fn region_name(region: *const Region) []const u8 {
    return region.name_buf[0..region.name_len];
}

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
            var region_stack: [8]Region = undefined;
            var region_len: usize = 0;
            var region_counter: usize = 0;
            for (self.func.eqns) |eqn| {
                const desired = self.region_stack_for_eqn(eqn);
                const common = common_region_prefix(region_stack[0..region_len], desired);

                var i: usize = region_len;
                while (i > common) : (i -= 1) {
                    try self.close_region(region_stack[0 .. i - 1], region_stack[i - 1]);
                }
                region_len = common;

                i = common;
                while (i < desired.len) : (i += 1) {
                    region_len = try self.open_region(
                        region_stack[0..region_len],
                        &region_stack,
                        region_len,
                        &region_counter,
                        desired.items[i],
                    );
                }

                try self.emit_binding_prefix(region_stack[0..region_len]);
                try self.emit_binding(eqn);
                try w.writeAll("\n");
            }
            var i: usize = region_len;
            while (i > 0) : (i -= 1) {
                try self.close_region(region_stack[0 .. i - 1], region_stack[i - 1]);
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

    fn emit_binding_prefix(self: *Self, regions: []const Region) !void {
        try self.writer.writeAll(self.indent);
        try self.writer.writeAll(self.indent);
        try self.emit_gutters(regions.len);
    }

    fn emit_region_start(self: *Self, regions: []const Region, region: Region) !void {
        try self.writer.writeAll(self.indent);
        try self.writer.writeAll(self.indent);
        try self.emit_gutters(regions.len);
        try self.styler.write_region(self.styler.cfg.symbols.region_start);
        try self.styler.write_region(" ");
        try self.styler.write_region(region_name(&region));
        try self.styler.write_region("[");
        for (region.spec.tags[0..region.spec.len], 0..) |tag, i| {
            if (i > 0) try self.styler.write_region(", ");
            try self.styler.write_region(tag.key);
            if (tag.value) |value| {
                try self.styler.write_region("=");
                try self.styler.write_region(value);
            }
        }
        try self.styler.write_region("]\n");
    }

    fn emit_region_end(self: *Self, regions: []const Region, region: Region) !void {
        try self.writer.writeAll(self.indent);
        try self.writer.writeAll(self.indent);
        try self.emit_gutters(regions.len);
        try self.styler.write_region(self.styler.cfg.symbols.region_end);
        try self.styler.write_region(" ");
        try self.styler.write_region(region_name(&region));
        try self.styler.write_region("\n");
    }

    fn emit_gutters(self: *Self, count: usize) !void {
        var i: usize = 0;
        while (i < count) : (i += 1) {
            try self.styler.write_region(self.styler.cfg.symbols.region_gutter);
            try self.writer.writeAll(" ");
        }
    }

    fn region_stack_for_eqn(self: *Self, eqn: pr.Eqn) RegionStackSpec {
        const params = eqn.params.slice(pr.Param, self.func.params_store);
        return region_stack_from_params(params);
    }

    fn open_region(
        self: *Self,
        active: []const Region,
        stack: *[8]Region,
        len: usize,
        counter: *usize,
        spec: RegionSpec,
    ) !usize {
        var region = Region{ .name_buf = undefined, .name_len = 0, .spec = spec };
        const name_slice = std.fmt.bufPrint(&region.name_buf, "region{d}", .{counter.*}) catch "region?";
        region.name_len = name_slice.len;
        try self.emit_region_start(active, region);
        stack[len] = region;
        counter.* += 1;
        return len + 1;
    }

    fn close_region(self: *Self, active: []const Region, region: Region) !void {
        try self.emit_region_end(active, region);
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

fn region_stack_from_params(params: []const pr.Param) RegionStackSpec {
    var stack: RegionStackSpec = .{};
    if (pr.param_kernelize_provider(params)) |provider| {
        stack.items[stack.len] = region_spec_kernelize(provider);
        stack.len += 1;
    }
    if (pr.param_outline(params) orelse false) {
        stack.items[stack.len] = region_spec_outline();
        stack.len += 1;
    }
    return stack;
}

fn region_spec_kernelize(provider: []const u8) RegionSpec {
    var spec: RegionSpec = .{};
    spec.tags[0] = .{ .key = "kernelize", .value = provider };
    spec.len = 1;
    return spec;
}

fn region_spec_outline() RegionSpec {
    var spec: RegionSpec = .{};
    spec.tags[0] = .{ .key = "outline", .value = null };
    spec.len = 1;
    return spec;
}

fn region_spec_equal(a: RegionSpec, b: RegionSpec) bool {
    if (a.len != b.len) return false;
    for (a.tags[0..a.len], 0..) |tag, i| {
        const other = b.tags[i];
        if (!std.mem.eql(u8, tag.key, other.key)) return false;
        if (tag.value) |value| {
            const other_value = other.value orelse return false;
            if (!std.mem.eql(u8, value, other_value)) return false;
        } else {
            if (other.value != null) return false;
        }
    }
    return true;
}

fn common_region_prefix(current: []const Region, desired: RegionStackSpec) usize {
    const max_len = if (current.len < desired.len) current.len else desired.len;
    var i: usize = 0;
    while (i < max_len) : (i += 1) {
        if (!region_spec_equal(current[i].spec, desired.items[i])) break;
    }
    return i;
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
    const kparams = &.{pr.Param{ .kernelize_provider = "tvm" }};

    const add1 = try b.emit(.add, &.{ a, c }, kparams);
    const add2 = try b.emit(.add, &.{ add1, c }, kparams);
    const out = try b.add(add2, c);
    const func = try b.finish(&.{out});

    var buf: [512]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try emit(func, &w, .plain, .{});

    const result = w.buffered();
    try std.testing.expect(std.mem.indexOf(u8, result, "> region0[kernelize=tvm]") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "< region0") != null);
}
