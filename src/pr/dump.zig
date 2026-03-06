const std = @import("std");
const pr = @import("pr.zig");
const json = @import("json.zig");
const zxpr = @import("zxpr/root.zig");

const Writer = std.Io.Writer;

pub const OutputTarget = union(enum) {
    stdout,
    file: []const u8,
};

pub const ZxprMode = enum {
    auto,
    plain,
};

pub const ZxprSpec = struct {
    mode: ZxprMode = .auto,
    style_opts: zxpr.style.ConfigOpts = .{},
};

pub const DumpSpec = union(enum) {
    zxpr: ZxprSpec,
    json,
};

pub const Config = struct {
    target: OutputTarget = .stdout,
    spec: DumpSpec = .{ .zxpr = .{} },

    /// Optional label identifying which PR function was selected as entry.
    /// Printed as "entry: <name>" header in ZXPR output for user reference.
    /// Does not affect actual function names in the dumped content.
    entry_name: ?[]const u8 = null,
};

pub fn emit_program(program: *const pr.Program, writer: *Writer, cfg: Config) !void {
    switch (cfg.spec) {
        .json => try json.emit_program(program, writer),
        .zxpr => |zx| {
            if (cfg.entry_name) |name| {
                try writer.print("entry: {s}\n", .{name});
            }

            const mode: zxpr.style.ConfigMode = switch (zx.mode) {
                .plain => .plain,
                .auto => switch (cfg.target) {
                    .stdout => .auto_stdout,
                    .file => .plain,
                },
            };
            const zx_cfg = zxpr.style.config(mode, zx.style_opts);

            for (program.functions, 0..) |func, i| {
                if (i > 0) try writer.writeAll("\n");
                try zxpr.emit(func, writer, zx_cfg);
            }
        },
    }
}

