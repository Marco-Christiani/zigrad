//! Process-wide runtime inputs.
//!
//! `main` constructs these values from `std.process.Init`.
//!
//! CLI dispatch passes them to subsystems that need I/O, environment access,
//!  or allocation. Pure subsystems do not receive them.
const std = @import("std");

/// Path policy for one runtime-loaded library.
///
/// The path is borrowed. Its owner must keep it alive until the integration
///  opens the library.
pub const RuntimeLibrary = struct {
    path: []const u8,

    /// Resolve one optional path override from an explicit environment map.
    pub fn from_environ(
        environ: *const std.process.Environ.Map,
        env_name: []const u8,
        default_path: []const u8,
    ) RuntimeLibrary {
        const configured = environ.get(env_name) orelse return .{ .path = default_path };
        return .{ .path = if (configured.len == 0) default_path else configured };
    }
};

/// Process services passed explicitly from application entry.
pub const RuntimeEnv = struct {
    /// Allocator supplied by process initialization.
    allocator: std.mem.Allocator,
    /// I/O implementation supplied by process initialization.
    io: std.Io,
    /// Mutable environment map supplied by process initialization.
    environ: *std.process.Environ.Map,

    /// Construct runtime services from Zig process initialization.
    pub fn from_init(init: std.process.Init) RuntimeEnv {
        return .{
            .allocator = init.gpa,
            .io = init.io,
            .environ = init.environ_map,
        };
    }
};

test RuntimeLibrary {
    var environ: std.process.Environ.Map = .init(std.testing.allocator);
    defer environ.deinit();

    const fallback = RuntimeLibrary.from_environ(&environ, "ZG_TEST_LIBRARY", "libtest.so");
    try std.testing.expectEqualStrings("libtest.so", fallback.path);

    try environ.put("ZG_TEST_LIBRARY", "/runtime/libtest.so");
    const configured = RuntimeLibrary.from_environ(&environ, "ZG_TEST_LIBRARY", "libtest.so");
    try std.testing.expectEqualStrings("/runtime/libtest.so", configured.path);
}
