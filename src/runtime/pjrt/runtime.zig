/// PJRT Runtime
///
/// Manages device enumeration, buffer transfers, and execution.
/// Compilation is handled by the toolchain layer - this module focuses on execution.
const std = @import("std");

const plugin = @import("../../ffi/pjrt/plugin.zig");
const pjrt_api = @import("../../ffi/pjrt/api.zig");
const pjrt_types = @import("../../ffi/pjrt/types.zig");

// Re-export common types for convenience
pub const LoadedExecutable = pjrt_types.LoadedExecutable;
pub const Buffer = pjrt_types.Buffer;
pub const Device = pjrt_types.Device;
pub const Event = pjrt_types.Event;
pub const BufferType = pjrt_types.BufferType;

pub const Runtime = struct {
    api: *pjrt_api.Api,
    client: pjrt_types.Client,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, plugin_path: []const u8) !Runtime {
        const api_ptr = try allocator.create(pjrt_api.Api);
        errdefer allocator.destroy(api_ptr);

        api_ptr.* = try plugin.loadPlugin(plugin_path);
        errdefer plugin.unloadPlugin(api_ptr.*);

        var client = try pjrt_types.Client.create(api_ptr);
        errdefer client.deinit();

        return .{
            .api = api_ptr,
            .client = client,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Runtime) void {
        self.client.deinit();
        plugin.unloadPlugin(self.api.*);
        self.allocator.destroy(self.api);
    }

    /// Get all available devices
    pub fn getDevices(self: *Runtime, allocator: std.mem.Allocator) ![]Device {
        return self.client.getDevices(allocator);
    }

    /// Deprecated: use getDevices instead
    pub fn devices(self: *Runtime, allocator: std.mem.Allocator) ![]Device {
        return self.getDevices(allocator);
    }

    /// Create a buffer on device from host data
    pub fn bufferFromHost(
        self: *Runtime,
        device: *const Device,
        data: []const u8,
        dtype: BufferType,
        shape: []const i64,
    ) !Buffer {
        return self.client.bufferFromHost(device, data, dtype, shape);
    }

    /// Load a serialized executable (JIT cache path) into a ready-to-run loaded executable.
    pub fn loadSerializedExecutable(
        self: *Runtime,
        serialized_executable: []const u8,
        overridden_compile_options: ?[]const u8,
    ) !LoadedExecutable {
        return self.client.deserializeAndLoad(serialized_executable, overridden_compile_options);
    }

    /// Get the underlying client (for toolchain access)
    pub fn getClient(self: *Runtime) *pjrt_types.Client {
        return &self.client;
    }
};
