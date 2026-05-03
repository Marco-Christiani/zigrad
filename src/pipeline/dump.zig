const std = @import("std");

const pass = @import("pass.zig");
const pr = @import("../pr/pr.zig");
const zxpr = @import("../pr/zxpr.zig");
const hlo_decode = @import("../c/xla/hlo_decode.zig");

pub const DumpTarget = zxpr.OutputTarget;
pub const DumpSpec = zxpr.DumpSpec;
pub const DumpConfig = zxpr.Config;

fn dump_pr_pass(ptr: *anyopaque, artifact: *pass.Artifact, ctx: *pass.PassContext) pass.PassError!void {
    if (artifact.kind() != .pr) return error.ArtifactKindMismatch;

    const cfg: *DumpConfig = @ptrCast(@alignCast(ptr));

    const program = artifact.pr;
    const task = struct {
        program: *const pr.Program,
        cfg: DumpConfig,
        fn run(self: @This(), out: *std.Io.Writer) !void {
            try zxpr.emit_program(self.program, out, self.cfg);
        }
    }{
        .program = program,
        .cfg = cfg.*,
    };

    with_writer(ctx.io, cfg, task) catch return error.ValidationFailed;
}

fn dump_mlir_pass(ptr: *anyopaque, artifact: *pass.Artifact, ctx: *pass.PassContext) pass.PassError!void {
    if (artifact.kind() != .stablehlo) return error.ArtifactKindMismatch;

    const cfg: *DumpConfig = @ptrCast(@alignCast(ptr));

    const sh = artifact.stablehlo;
    if (sh.encoding != .text) return error.ValidationFailed;

    const task = struct {
        bytes: []const u8,
        entry: ?[]const u8,
        fn run(self: @This(), out: *std.Io.Writer) !void {
            try emit_mlir(out, self.bytes, self.entry);
        }
    }{
        .bytes = sh.bytes,
        .entry = cfg.entry_name,
    };

    with_writer(ctx.io, cfg, task) catch return error.ValidationFailed;
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
        .input_kind = .stablehlo,
        .output_kind = .stablehlo,
    };
}

fn emit_mlir(out: *std.Io.Writer, bytes: []const u8, entry: ?[]const u8) !void {
    if (entry) |name| {
        try out.print("entry: {s}\n", .{name});
    }

    try out.writeAll(bytes);
    if (bytes.len == 0 or bytes[bytes.len - 1] != '\n') try out.writeAll("\n");
}

/// Dump backend-optimized program output (post-compilation, not a pipeline pass).
///
/// Attempts to decode the protobuf bytes into human-readable HLO text first.
/// On decode failure, falls back to a size summary (stdout) or raw bytes (file).
pub fn dump_optimized_program(io: std.Io, config: *const DumpConfig, code: []const u8, format: []const u8, allocator: std.mem.Allocator) !void {
    const task = struct {
        code: []const u8,
        format: []const u8,
        is_stdout: bool,
        allocator: std.mem.Allocator,

        fn run(self: @This(), out: *std.Io.Writer) !void {
            // Try decoding protobuf into human-readable HLO text.
            if (hlo_decode.decode_and_print(self.code, self.allocator, out)) return;

            // Decode failed -- fall back to raw output.
            if (self.is_stdout) {
                try out.print("optimized program format: {s}\n", .{self.format});
                try out.print("optimized program size: {d} bytes\n", .{self.code.len});
            } else {
                try out.writeAll(self.code);
            }
        }
    }{
        .code = code,
        .format = format,
        .is_stdout = config.target == .stdout,
        .allocator = allocator,
    };

    try with_writer(io, config, task);
}

fn with_writer(io: std.Io, config: *const DumpConfig, task: anytype) !void {
    var buffer: [8192]u8 = undefined;

    switch (config.target) {
        .stdout => {
            var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
            const out = &stdout_writer.interface;
            try task.run(out);
            try out.flush();
        },
        .file => |path| {
            var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
            defer file.close(io);

            var file_writer = file.writer(io, &buffer);
            const out = &file_writer.interface;
            try task.run(out);
            try out.flush();
        },
    }
}

test "emit_program includes entry header and zxpr output" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main_fn");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{1});
    const func = try b.finish(&.{x});
    try program.add_function(func);

    var writer_state = std.Io.Writer.Allocating.init(testing.allocator);
    defer writer_state.deinit();

    try zxpr.emit_program(&program, &writer_state.writer, .{
        .target = .stdout,
        .entry_name = "test_main",
        .spec = .{ .zxpr = .{ .mode = .plain } },
    });
    const output = try writer_state.toOwnedSlice();
    defer testing.allocator.free(output);

    try testing.expect(std.mem.indexOf(u8, output, "entry: test_main") != null);
    try testing.expect(std.mem.indexOf(u8, output, "zxpr main_fn") != null);
}

test "emit_program json format" {
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

    try zxpr.emit_program(&program, &writer_state.writer, .{
        .target = .stdout,
        .entry_name = "main",
        .spec = .json,
    });
    const output = try writer_state.toOwnedSlice();
    defer testing.allocator.free(output);

    // JSON format: no entry header, produces JSON object
    try testing.expect(std.mem.indexOf(u8, output, "entry:") == null);
    try testing.expect(std.mem.indexOf(u8, output, "\"name\":\"main\"") != null);
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
