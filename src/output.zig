//! Shared mechanics for writing optional compiler output.

const std = @import("std");

/// Destination for compiler output.
pub const Target = union(enum) {
    stdout,
    file: []const u8,
};

/// Shared output destination and optional entry label.
pub const Config = struct {
    /// Selects stdout or a file path.
    target: Target = .stdout,

    /// Labels output associated with one entry function.
    entry_name: ?[]const u8 = null,
};

/// Writes one output task to its selected destination.
pub fn write(io: std.Io, target: Target, task: anytype) !void {
    var buffer: [8192]u8 = undefined;

    switch (target) {
        .stdout => {
            var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
            const out = &stdout_writer.interface;
            try task.emit(out);
            try out.flush();
        },
        .file => |path| {
            var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
            defer file.close(io);

            var file_writer = file.writer(io, &buffer);
            const out = &file_writer.interface;
            try task.emit(out);
            try out.flush();
        },
    }
}
