const std = @import("std");

pub const Error = error{
    InvalidCpuDeviceCount,
};

/// Process-independent options for the PJRT plugin loader.
pub const PluginOptions = struct {
    /// Emit dynamic-loader diagnostics.
    debug: bool = false,

    /// Trace PJRT execute calls.
    trace_execute: bool = false,

    /// Close the plugin DSO when the final client reference is released.
    ///
    /// This defaults to false because some PJRT plugins retain process-global
    ///  state that is unsafe to tear down before process exit.
    close_on_unload: bool = false,
};

/// Inputs required to initialize a PJRT client.
pub const InitOptions = struct {
    /// Override the number of devices created by the CPU plugin.
    cpu_device_count: ?usize = null,

    /// Dynamic-loader and diagnostic policy.
    plugin: PluginOptions = .{},
};

/// Resolve PJRT initialization policy from an explicit environment map.
pub fn from_environ(environ: *const std.process.Environ.Map) Error!InitOptions {
    return .{
        .cpu_device_count = try optional_cpu_device_count(environ),
        .plugin = .{
            .debug = flag(environ, "ZG_PJRT_DEBUG"),
            .trace_execute = flag(environ, "ZG_PJRT_TRACE_EXECUTE"),
            .close_on_unload = flag(environ, "ZG_PJRT_DLCLOSE"),
        },
    };
}

fn flag(environ: *const std.process.Environ.Map, name: []const u8) bool {
    const value = environ.get(name) orelse return false;
    return value.len != 0 and !std.mem.eql(u8, value, "0");
}

fn optional_cpu_device_count(environ: *const std.process.Environ.Map) Error!?usize {
    const value = environ.get("ZG_CPU_DEVICE_COUNT") orelse return null;
    if (value.len == 0) return null;
    return std.fmt.parseInt(usize, value, 10) catch error.InvalidCpuDeviceCount;
}

test "from_environ resolves PJRT initialization options" {
    var environ: std.process.Environ.Map = .init(std.testing.allocator);
    defer environ.deinit();

    try std.testing.expectEqualDeep(InitOptions{}, try from_environ(&environ));

    try environ.put("ZG_CPU_DEVICE_COUNT", "4");
    try environ.put("ZG_PJRT_DEBUG", "1");
    try environ.put("ZG_PJRT_TRACE_EXECUTE", "true");
    try environ.put("ZG_PJRT_DLCLOSE", "1");

    try std.testing.expectEqualDeep(InitOptions{
        .cpu_device_count = 4,
        .plugin = .{
            .debug = true,
            .trace_execute = true,
            .close_on_unload = true,
        },
    }, try from_environ(&environ));
}

test "from_environ rejects an invalid CPU device count" {
    var environ: std.process.Environ.Map = .init(std.testing.allocator);
    defer environ.deinit();

    try environ.put("ZG_CPU_DEVICE_COUNT", "four");
    try std.testing.expectError(error.InvalidCpuDeviceCount, from_environ(&environ));
}
