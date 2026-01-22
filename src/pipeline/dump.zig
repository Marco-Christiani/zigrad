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
    entry_name: ?[]const u8 = null,
};

pub fn dumpPrPass(artifact: *pass.Artifact, ctx: *pass.PassContext, userdata: ?*anyopaque) pass.PassError!void {
    _ = ctx;
    if (artifact.kind() != .pr) return error.ArtifactKindMismatch;

    const cfg_ptr = userdata orelse return error.MissingContext;
    const cfg: *DumpConfig = @ptrCast(@alignCast(cfg_ptr));

    const program = artifact.pr;
    const task = struct {
        program: *const pr.Program,
        entry: ?[]const u8,
        fn run(self: @This(), out: *std.Io.Writer) !void {
            try emitProgram(out, self.program, self.entry);
        }
    }{
        .program = program,
        .entry = cfg.entry_name,
    };

    withWriter(cfg, task) catch return error.ValidationFailed;
}

pub fn dumpMlirPass(artifact: *pass.Artifact, ctx: *pass.PassContext, userdata: ?*anyopaque) pass.PassError!void {
    _ = ctx;
    if (artifact.kind() != .mlir) return error.ArtifactKindMismatch;

    const cfg_ptr = userdata orelse return error.MissingContext;
    const cfg: *DumpConfig = @ptrCast(@alignCast(cfg_ptr));

    const mlir = artifact.mlir;
    if (mlir.encoding != .text) return error.ValidationFailed;

    const task = struct {
        bytes: []const u8,
        entry: ?[]const u8,
        fn run(self: @This(), out: *std.Io.Writer) !void {
            try emitMlir(out, self.bytes, self.entry);
        }
    }{
        .bytes = mlir.bytes,
        .entry = cfg.entry_name,
    };

    withWriter(cfg, task) catch return error.ValidationFailed;
}

pub fn dumpPrPassWithConfig(config: *DumpConfig) pass.Pass {
    return .{
        .name = "dump_pr",
        .input_kind = .pr,
        .output_kind = .pr,
        .run = dumpPrPass,
        .userdata = config,
    };
}

pub fn dumpMlirPassWithConfig(config: *DumpConfig) pass.Pass {
    return .{
        .name = "dump_mlir",
        .input_kind = .mlir,
        .output_kind = .mlir,
        .run = dumpMlirPass,
        .userdata = config,
    };
}

fn emitProgram(out: *std.Io.Writer, program: *const pr.Program, entry: ?[]const u8) !void {
    if (entry) |name| {
        try out.print("entry: {s}\n", .{name});
    }
    for (program.functions, 0..) |func, i| {
        if (i > 0) try out.writeAll("\n");
        var emitter = zxpr.Emitter.init(out, func);
        try emitter.emit();
    }
}

fn emitMlir(out: *std.Io.Writer, bytes: []const u8, entry: ?[]const u8) !void {
    if (entry) |name| {
        try out.print("entry: {s}\n", .{name});
    }
    try out.writeAll(bytes);
    if (bytes.len == 0 or bytes[bytes.len - 1] != '\n') try out.writeAll("\n");
}

fn withWriter(config: *const DumpConfig, task: anytype) !void {
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
