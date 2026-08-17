//! Read-only command support for serialized PR programs.

const std = @import("std");
const json = @import("json.zig");
const pr = @import("pr.zig");
const serialize = @import("serialize.zig");
const zxpr = @import("zxpr.zig");

const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;
const max_input_bytes = 1024 * 1024 * 1024;

/// Textual rendering produced from a serialized PR program.
pub const RenderFormat = enum {
    zxpr,
    json,
};

/// Read-only operation to apply to a serialized PR program.
pub const Action = union(enum) {
    render: RenderFormat,
    info,
};

/// Parse a serialized PR file and write the selected result to standard output.
pub fn run(io: std.Io, allocator: Allocator, path: []const u8, action: Action) !void {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(max_input_bytes),
    );
    defer allocator.free(bytes);

    var program = try serialize.parse(allocator, bytes);
    defer program.deinit();

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    try emit_action(&program, bytes.len, action, &stdout_writer.interface);
    try stdout_writer.interface.flush();
}

fn emit_action(
    program: *const pr.Program,
    input_size: usize,
    action: Action,
    writer: *Writer,
) !void {
    switch (action) {
        .render => |format| switch (format) {
            .zxpr => try zxpr.emit_program(program, writer, .{
                .spec = .{ .zxpr = .{ .mode = .plain } },
            }),
            .json => try json.emit_program(program, writer),
        },
        .info => {
            try writer.print("size: {d} bytes\n", .{input_size});
            try writer.print("functions: {d}\n", .{program.functions().len});
            for (program.functions()) |func| {
                try writer.print(
                    "function {s}: {d} ops, {d} values\n",
                    .{ func.name, func.ops.len, func.var_count },
                );
            }
        },
    }
}

test "emit_action renders ZXPR and JSON" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{1});
    const y = try b.param_tensor(.f32, &.{1});
    const sum = try b.add(x, y);
    const func = try b.finish(&.{sum});
    _ = try program.add_function(func);

    for ([_]RenderFormat{ .zxpr, .json }) |format| {
        var output: Writer.Allocating = .init(std.testing.allocator);
        defer output.deinit();
        try emit_action(&program, 0, .{ .render = format }, &output.writer);

        const expected = switch (format) {
            .zxpr => "zxpr @0 main",
            .json => "\"name\":\"main\"",
        };
        try std.testing.expect(std.mem.indexOf(u8, output.written(), expected) != null);
    }
}

test "emit_action reports program contents" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "summary");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{1});
    const y = try b.add(x, x);
    const func = try b.finish(&.{y});
    _ = try program.add_function(func);

    var output: Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    try emit_action(&program, 123, .info, &output.writer);

    try std.testing.expectEqualStrings(
        "size: 123 bytes\n" ++
            "functions: 1\n" ++
            "function summary: 1 ops, 2 values\n",
        output.written(),
    );
}
