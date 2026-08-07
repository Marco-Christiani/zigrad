//! Native object linker capability.

const std = @import("std");

const log = std.log.scoped(.@"zg/linker");

/// Environment key used to configure the ELF linker executable.
pub const path_env = "ZG_ELF_LINKER_PATH";

/// Errors returned while resolving linker configuration.
pub const ConfigError = error{MissingLinker};

/// Errors returned while invoking the configured linker.
pub const Error = std.process.RunError || error{
    LinkFailed,
    NoObjects,
};

/// Output formats supported by the linker capability.
pub const OutputFormat = enum {
    elf_shared,
};

/// Inputs for one native link operation.
pub const Request = struct {
    /// Format of the linked output.
    format: OutputFormat,

    /// Path replaced or created by the linker.
    output: []const u8,

    /// Native object paths borrowed for the link operation.
    objects: []const []const u8,
};

/// Explicit external linker executable.
pub const Linker = struct {
    /// Linker executable path borrowed by this value.
    executable: []const u8,

    /// Resolve the linker path from an explicit environment map.
    ///
    /// The returned path is borrowed from `environ`.
    pub fn from_environ(
        environ: *const std.process.Environ.Map,
    ) ConfigError!Linker {
        return .{
            .executable = environ.get(path_env) orelse
                return error.MissingLinker,
        };
    }

    /// Link native objects into the requested output.
    ///
    /// A failed linker process may leave a partial output file.
    pub fn link(
        self: Linker,
        io: std.Io,
        allocator: std.mem.Allocator,
        request: Request,
    ) Error!void {
        if (request.objects.len == 0) return error.NoObjects;

        const arguments = try make_arguments(allocator, self, request);
        defer allocator.free(arguments);

        const result = try std.process.run(allocator, io, .{
            .argv = arguments,
            .stderr_limit = .limited(1024 * 1024),
            .stdout_limit = .limited(64 * 1024),
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);

        switch (result.term) {
            .exited => |code| if (code != 0) {
                log.err("linker exited with code {d}: {s}", .{ code, result.stderr });
                return error.LinkFailed;
            },
            else => {
                log.err("linker terminated abnormally", .{});
                return error.LinkFailed;
            },
        }
        log.debug("linked {d} objects into {s}", .{
            request.objects.len,
            request.output,
        });
    }
};

fn make_arguments(
    allocator: std.mem.Allocator,
    linker: Linker,
    request: Request,
) std.mem.Allocator.Error![][]const u8 {
    const fixed = switch (request.format) {
        .elf_shared => [_][]const u8{
            linker.executable,
            "-shared",
            "-o",
            request.output,
        },
    };
    const arguments = try allocator.alloc([]const u8, fixed.len + request.objects.len);
    @memcpy(arguments[0..fixed.len], &fixed);
    @memcpy(arguments[fixed.len..], request.objects);
    return arguments;
}

test "Linker resolves its executable" {
    var environ: std.process.Environ.Map = .init(std.testing.allocator);
    defer environ.deinit();

    try std.testing.expectError(error.MissingLinker, Linker.from_environ(&environ));
    try environ.put(path_env, "/toolchain/bin/ld.lld");

    const linker = try Linker.from_environ(&environ);
    try std.testing.expectEqualStrings("/toolchain/bin/ld.lld", linker.executable);
}

test "Linker constructs an ELF shared-library command" {
    const linker = Linker{ .executable = "/toolchain/bin/ld.lld" };
    const arguments = try make_arguments(std.testing.allocator, linker, .{
        .format = .elf_shared,
        .output = "/tmp/kernel.so",
        .objects = &.{ "/tmp/host.o", "/tmp/device.o" },
    });
    defer std.testing.allocator.free(arguments);

    try std.testing.expectEqualDeep(&[_][]const u8{
        "/toolchain/bin/ld.lld",
        "-shared",
        "-o",
        "/tmp/kernel.so",
        "/tmp/host.o",
        "/tmp/device.o",
    }, arguments);
}
