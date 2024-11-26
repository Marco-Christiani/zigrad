const std = @import("std");

pub const HostDevice = @import("host.zig").HostDevice;

pub const CudaDevice = struct {
    pub fn print(_: CudaDevice) void {
        std.debug.print("Using Cuda Device\n", .{});
    }
    pub fn reference(self: *const CudaDevice) DeviceReference {
        return .{ .cuda = self };
    }
};

// callback to replace host reference to union
pub fn host_reference(self: *const HostDevice) DeviceReference {
    return .{ .host = self };
}

pub const DeviceReference = union(enum) {
    cuda: *const CudaDevice,
    host: *const HostDevice,
    pub fn print(self: DeviceReference) void {
        switch (self) {
            inline else => |dev| dev.print(),
        }
    }
};
