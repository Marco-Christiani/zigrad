const std = @import("std");

const pass = @import("pass.zig");
const pr = @import("../pr/pr.zig");
const zxpr = @import("../pr/zxpr.zig");

pub const DumpTarget = enum {
    stdout,
    file,
};

pub const DumpConfig = struct {
    target: DumpTarget = .stdout,
    path: ?[]const u8 = null,

    /// Optional label identifying which PR function was selected as entry.
    /// Printed as "entry: <name>" header in dump output for user reference.
    /// Does not affect the actual function names in the dumped content.
    entry_name: ?[]const u8 = null,
};

fn dump_pr_pass(ptr: *anyopaque, artifact: *pass.Artifact, ctx: *pass.PassContext) pass.PassError!void {
    _ = ctx;
    if (artifact.kind() != .pr) return error.ArtifactKindMismatch;

    const cfg: *DumpConfig = @ptrCast(@alignCast(ptr));

    const program = artifact.pr;
    const task = struct {
        program: *const pr.Program,
        entry: ?[]const u8,
        fn run(self: @This(), out: *std.Io.Writer) !void {
            try emit_program(out, self.program, self.entry);
        }
    }{
        .program = program,
        .entry = cfg.entry_name,
    };

    with_writer(cfg, task) catch return error.ValidationFailed;
}

fn dump_mlir_pass(ptr: *anyopaque, artifact: *pass.Artifact, ctx: *pass.PassContext) pass.PassError!void {
    _ = ctx;
    if (artifact.kind() != .mlir) return error.ArtifactKindMismatch;

    const cfg: *DumpConfig = @ptrCast(@alignCast(ptr));

    const mlir = artifact.mlir;
    if (mlir.encoding != .text) return error.ValidationFailed;

    const task = struct {
        bytes: []const u8,
        entry: ?[]const u8,
        fn run(self: @This(), out: *std.Io.Writer) !void {
            try emit_mlir(out, self.bytes, self.entry);
        }
    }{
        .bytes = mlir.bytes,
        .entry = cfg.entry_name,
    };

    with_writer(cfg, task) catch return error.ValidationFailed;
}

pub fn dump_pr_pass_with_config(config: *DumpConfig) pass.Pass {
    return .{
        .ptr = @ptrCast(config),
        .run_fn = dump_pr_pass,
        .name = "dump_pr",
        .input_kind = .pr,
        .output_kind = .pr,
    };
}

pub fn dump_mlir_pass_with_config(config: *DumpConfig) pass.Pass {
    return .{
        .ptr = @ptrCast(config),
        .run_fn = dump_mlir_pass,
        .name = "dump_mlir",
        .input_kind = .mlir,
        .output_kind = .mlir,
    };
}

fn emit_program(out: *std.Io.Writer, program: *const pr.Program, entry: ?[]const u8) !void {
    if (entry) |name| {
        try out.print("entry: {s}\n", .{name});
    }
    for (program.functions, 0..) |func, i| {
        if (i > 0) try out.writeAll("\n");
        try zxpr.emit(func, out, .auto_stdout, .{});
    }
}

fn emit_mlir(out: *std.Io.Writer, bytes: []const u8, entry: ?[]const u8) !void {
    if (entry) |name| {
        try out.print("entry: {s}\n", .{name});
    }
    try out.writeAll(bytes);
    if (bytes.len == 0 or bytes[bytes.len - 1] != '\n') try out.writeAll("\n");
}

fn with_writer(config: *const DumpConfig, task: anytype) !void {
    var buffer: [8192]u8 = undefined;

    if (config.target == .stdout) {
        var stdout_writer = std.fs.File.stdout().writer(&buffer);
        const out = &stdout_writer.interface;
        try task.run(out);
        try out.flush();
        return;
    }

    const path = config.path orelse return error.MissingContext;
    var file = if (std.fs.path.isAbsolute(path))
        try std.fs.createFileAbsolute(path, .{ .truncate = true })
    else
        try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();

    var file_writer = file.writer(&buffer);
    const out = &file_writer.interface;
    try task.run(out);
    try out.flush();
}

test "emit_program includes entry header and zxpr output" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{1});
    const func = try b.finish(&.{x});
    try program.add_function(func);

    var writer_state = std.Io.Writer.Allocating.init(testing.allocator);
    defer writer_state.deinit();

    try emit_program(&writer_state.writer, &program, "main");
    const output = try writer_state.toOwnedSlice();
    defer testing.allocator.free(output);

    try testing.expect(std.mem.indexOf(u8, output, "entry: main") != null);
    try testing.expect(std.mem.indexOf(u8, output, "zxpr main") != null);
}

test "emit_mlir appends newline" {
    const testing = std.testing;

    var writer_state = std.Io.Writer.Allocating.init(testing.allocator);
    defer writer_state.deinit();

    try emit_mlir(&writer_state.writer, "module {}", null);
    const output = try writer_state.toOwnedSlice();
    defer testing.allocator.free(output);

    try testing.expect(output.len > 0);
    try testing.expectEqual(@as(u8, '\n'), output[output.len - 1]);
}
