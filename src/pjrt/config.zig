const std = @import("std");

pub const Error = error{
    /// `ZG_CPU_DEVICE_COUNT` is not a positive 32-bit integer.
    InvalidCpuDeviceCount,
    /// `ZG_PJRT_XLA_GPU_MEMORY_FRACTION` is not a positive finite number.
    InvalidXlaGpuMemoryFraction,
    /// Environment configuration selects more than one client policy.
    ConflictingClientOptions,
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

/// Plugin-specific options used while creating a PJRT client.
pub const ClientOptions = union(enum) {
    /// Let the plugin use its default client policy.
    default,
    /// XLA CPU client configuration.
    xla_cpu: struct {
        /// Number of CPU devices exposed by the client.
        device_count: u32,
    },
    /// XLA GPU client configuration.
    xla_gpu: struct {
        /// Positive scaling factor applied to device memory by XLA's allocator.
        ///
        /// Values above one request unified memory and require device support.
        memory_fraction: f32,
    },
};

/// Inputs required to initialize a PJRT client.
pub const InitOptions = struct {
    /// Options interpreted by the selected plugin during client creation.
    client: ClientOptions = .default,

    /// Dynamic-loader and diagnostic policy.
    plugin: PluginOptions = .{},
};

/// Resolve PJRT initialization policy from an explicit environment map.
pub fn from_environ(environ: *const std.process.Environ.Map) Error!InitOptions {
    const cpu_device_count = try optional_cpu_device_count(environ);
    const gpu_memory_fraction = try optional_xla_gpu_memory_fraction(environ);
    if (cpu_device_count != null and gpu_memory_fraction != null)
        return error.ConflictingClientOptions;

    return .{
        .client = if (cpu_device_count) |count|
            .{ .xla_cpu = .{ .device_count = count } }
        else if (gpu_memory_fraction) |fraction|
            .{ .xla_gpu = .{ .memory_fraction = fraction } }
        else
            .default,
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

fn optional_cpu_device_count(environ: *const std.process.Environ.Map) Error!?u32 {
    const value = environ.get("ZG_CPU_DEVICE_COUNT") orelse return null;
    if (value.len == 0) return null;
    const count = std.fmt.parseInt(u32, value, 10) catch
        return error.InvalidCpuDeviceCount;
    if (count == 0) return error.InvalidCpuDeviceCount;
    return count;
}

fn optional_xla_gpu_memory_fraction(
    environ: *const std.process.Environ.Map,
) Error!?f32 {
    const value = environ.get("ZG_PJRT_XLA_GPU_MEMORY_FRACTION") orelse return null;
    if (value.len == 0) return null;
    const fraction = std.fmt.parseFloat(f32, value) catch
        return error.InvalidXlaGpuMemoryFraction;
    if (!std.math.isFinite(fraction) or fraction <= 0)
        return error.InvalidXlaGpuMemoryFraction;
    return fraction;
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
        .client = .{ .xla_cpu = .{ .device_count = 4 } },
        .plugin = .{
            .debug = true,
            .trace_execute = true,
            .close_on_unload = true,
        },
    }, try from_environ(&environ));
}

test "from_environ resolves XLA GPU memory policy" {
    var environ: std.process.Environ.Map = .init(std.testing.allocator);
    defer environ.deinit();

    try environ.put("ZG_PJRT_XLA_GPU_MEMORY_FRACTION", "0.5");
    try std.testing.expectEqualDeep(InitOptions{
        .client = .{ .xla_gpu = .{ .memory_fraction = 0.5 } },
    }, try from_environ(&environ));
}

test "from_environ rejects an invalid CPU device count" {
    var environ: std.process.Environ.Map = .init(std.testing.allocator);
    defer environ.deinit();

    try environ.put("ZG_CPU_DEVICE_COUNT", "four");
    try std.testing.expectError(error.InvalidCpuDeviceCount, from_environ(&environ));

    try environ.put("ZG_CPU_DEVICE_COUNT", "0");
    try std.testing.expectError(error.InvalidCpuDeviceCount, from_environ(&environ));
}

test "from_environ rejects an invalid XLA GPU memory fraction" {
    var environ: std.process.Environ.Map = .init(std.testing.allocator);
    defer environ.deinit();

    try environ.put("ZG_PJRT_XLA_GPU_MEMORY_FRACTION", "0");
    try std.testing.expectError(error.InvalidXlaGpuMemoryFraction, from_environ(&environ));
}

test "from_environ rejects incompatible client options" {
    var environ: std.process.Environ.Map = .init(std.testing.allocator);
    defer environ.deinit();

    try environ.put("ZG_CPU_DEVICE_COUNT", "2");
    try environ.put("ZG_PJRT_XLA_GPU_MEMORY_FRACTION", "0.5");
    try std.testing.expectError(error.ConflictingClientOptions, from_environ(&environ));
}
