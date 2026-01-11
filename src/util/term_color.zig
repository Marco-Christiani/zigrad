const std = @import("std");

pub const Color = std.Io.tty.Color;
pub const Config = std.Io.tty.Config;

/// Small helper for colorized output in example/non-core code paths.
pub const Tty = struct {
    conf: Config,
    writer: *std.Io.Writer,

    pub fn initForStdout(writer: *std.Io.Writer) Tty {
        return .{
            .conf = Config.detect(std.fs.File.stdout()),
            .writer = writer,
        };
    }

    pub fn initForStderr(writer: *std.Io.Writer) Tty {
        return .{
            .conf = Config.detect(std.fs.File.stderr()),
            .writer = writer,
        };
    }

    pub fn set(self: *Tty, color: Color) !void {
        try Config.setColor(self.conf, self.writer, color);
    }

    pub fn reset(self: *Tty) !void {
        try Config.setColor(self.conf, self.writer, .reset);
    }

    pub fn print(self: *Tty, color: Color, comptime fmt: []const u8, args: anytype) !void {
        try self.set(color);
        try self.writer.print(fmt, args);
        try self.reset();
        // IMMEDIATE flush
        try self.writer.flush();
    }
};
