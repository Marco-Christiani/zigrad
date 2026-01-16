const std = @import("std");

const plugin = @import("../../bridge/pjrt/plugin.zig");
const pjrt_api = @import("../../bridge/pjrt/api.zig");
const pjrt_types = @import("../../bridge/pjrt/types.zig");

pub const Runtime = struct {
    api: *pjrt_api.Api,
    client: pjrt_types.Client,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, plugin_path: []const u8) !Runtime {
        const api_ptr = try allocator.create(pjrt_api.Api);
        errdefer allocator.destroy(api_ptr);

        api_ptr.* = try plugin.loadPlugin(plugin_path);
        errdefer plugin.unloadPlugin(api_ptr.*);

        const client = try pjrt_types.Client.create(api_ptr);
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

    pub fn devices(self: *Runtime, allocator: std.mem.Allocator) ![]pjrt_types.Device {
        return self.client.getDevices(allocator);
    }
};
