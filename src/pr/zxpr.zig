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
///!   a: f32[2,3]
///!   b: f32[3,2]
///!   ; body (1 ops)
///!   let c: f32[2,2] = dot[contracting=([1], [0]), K=3](a, b)  ; vjp
///!   ;
///!   in c
///! }
const std = @import("std");
const pr = @import("pr.zig");
const ops = @import("ops/ops.zig");

const Writer = std.Io.Writer;

// ============================================================================
// Emitter
// ============================================================================

pub const Emitter = struct {
    const Self = @This();

    writer: *Writer,
    func: pr.Function,
    indent: []const u8,

    pub fn init(writer: *Writer, func: pr.Function) Self {
        return .{
            .writer = writer,
            .func = func,
            .indent = "  ",
        };
    }

    pub fn emit(self: *Self) !void {
        const w = self.writer;
        const ind = self.indent;

        try w.print("zxpr {s} {{\n", .{self.func.name});

        // Parameters section
        if (self.func.params.len > 0) {
            try w.writeAll(ind);
            try w.print("; params ({d})\n", .{self.func.params.len});
            for (self.func.params) |param_id| {
                try w.writeAll(ind);
                try self.emitVarWithType(param_id);
                try w.writeAll("\n");
            }
        }

        // Body section
        if (self.func.eqns.len > 0) {
            try w.writeAll(ind);
            try w.print("; body ({d} ops)\n", .{self.func.eqns.len});
            for (self.func.eqns) |eqn| {
                try w.writeAll(ind);
                try self.emitLet(eqn);
                try w.writeAll("\n");
            }
        }

        // Return section
        if (self.func.returns.len > 0) {
            try w.writeAll(ind);
            try w.writeAll(";\n");
            try w.writeAll(ind);
            try w.writeAll("in ");
            for (self.func.returns, 0..) |ret_id, i| {
                if (i > 0) try w.writeAll(", ");
                try self.emitVarName(ret_id);
            }
            try w.writeAll("\n");
        }

        try w.writeAll("}\n");
    }

    // ====================================================================
    // Helpers
    // ====================================================================

    fn emitVarName(self: *Self, id: pr.VarId) !void {
        try self.writer.print("{s}", .{varName(id)});
    }

    fn emitVarWithType(self: *Self, id: pr.VarId) !void {
        try self.emitVarName(id);
        try self.writer.writeAll(": ");
        try self.emitType(self.func.avals[@intCast(id)]);
    }

    fn emitType(self: *Self, aval: pr.Aval) !void {
        switch (aval) {
            .tensor => |t| {
                try self.writer.print("{s}[", .{@tagName(t.dtype)});
                for (t.shape.dims, 0..) |d, i| {
                    if (i > 0) try self.writer.writeAll(",");
                    try self.writer.print("{d}", .{d});
                }
                try self.writer.writeAll("]");
            },
        }
    }

    fn emitLet(self: *Self, eqn: pr.Eqn) !void {
        const outputs = eqn.outputs.slice(pr.VarId, self.func.varids_store);
        const inputs = eqn.inputs.slice(pr.VarId, self.func.varids_store);
        const params = eqn.params.slice(pr.Param, self.func.params_store);

        try self.writer.writeAll("let ");

        // Output binding(s)
        for (outputs, 0..) |out_id, i| {
            if (i > 0) try self.writer.writeAll(", ");
            try self.emitVarWithType(out_id);
        }

        try self.writer.writeAll(" = ");
        try self.writer.print("{s}[", .{@tagName(eqn.prim)});

        // Op-specific attributes via dispatch
        try ops.format(self.writer, self.func, eqn);

        if (pr.paramOutline(params) orelse false) {
            try self.writer.writeAll(", outline=true");
        }

        try self.writer.writeAll("](");

        // Inputs as function args
        for (inputs, 0..) |in_id, i| {
            if (i > 0) try self.writer.writeAll(", ");
            try self.emitVarName(in_id);
        }

        try self.writer.writeAll(")");

        // VJP annotation
        if (ops.hasVjp(eqn.prim)) {
            try self.writer.writeAll("  ; vjp");
        }
    }
};

// ============================================================================
// Helpers
// ============================================================================

/// Generate readable variable name from ID: 0->a, 1->b, ..., 26->aa, etc.
fn varName(id: pr.VarId) []const u8 {
    const names = comptime blk: {
        var arr: [256][]const u8 = undefined;
        for (0..256) |i| {
            if (i < 26) {
                arr[i] = &[_]u8{'a' + @as(u8, @intCast(i))};
            } else {
                const first = 'a' + @as(u8, @intCast((i - 26) / 26));
                const second = 'a' + @as(u8, @intCast((i - 26) % 26));
                arr[i] = &[_]u8{ first, second };
            }
        }
        break :blk arr;
    };
    if (id < 256) return names[id];
    return "??"; // Fallback for very large programs
}

// ============================================================================
// Public API
// ============================================================================

/// Emit ZXPR representation of a function
pub fn emit(func: pr.Function, writer: *Writer) !void {
    var emitter = Emitter.init(writer, func);
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

    const a = try b.paramTensor(.f32, &.{ 2, 3 });
    const c = try b.paramTensor(.f32, &.{ 3, 2 });
    const d = try b.dot(a, c);
    const func = try b.finish(&.{d});

    var buf: [512]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try emit(func, &w);

    const result = w.buffered();
    try std.testing.expect(std.mem.indexOf(u8, result, "zxpr main {") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "; params (2)") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "a: f32[2,3]") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "dot[contracting=([1], [0])") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "; vjp") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "in c") != null);
}

test "zxpr variable naming" {
    try std.testing.expectEqualStrings("a", varName(0));
    try std.testing.expectEqualStrings("b", varName(1));
    try std.testing.expectEqualStrings("z", varName(25));
    try std.testing.expectEqualStrings("aa", varName(26));
    try std.testing.expectEqualStrings("ab", varName(27));
}

test "zxpr with transpose shows permutation" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "t");
    defer b.deinit();

    const x = try b.paramTensor(.f32, &.{ 2, 3 });
    const y = try b.transpose(x, &.{ 1, 0 });
    const func = try b.finish(&.{y});

    var buf: [256]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try emit(func, &w);

    const result = w.buffered();
    try std.testing.expect(std.mem.indexOf(u8, result, "transpose[perm=[1, 0]]") != null);
}
