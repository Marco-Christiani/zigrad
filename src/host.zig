const std = @import("std");
const root = @import("root");
const backend = @import("backend.zig").backend;

pub const HostDevice = struct {
    pub fn print(_: HostDevice) void {
        std.debug.print("Using Host Device\n", .{});
    }
    pub const reference = switch (backend) {
        .HOST => host_reference,
        .CUDA => @import("cuda.zig").host_reference,
    };
};

// default reference if cuda device is not specified
fn host_reference(self: *const HostDevice) DeviceReference {
    return self;
}

pub const DeviceReference = *const HostDevice;
