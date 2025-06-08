const std = @import("std");

const zg = @import("zigrad");

const Tensor = zg.NDTensor(f32);
const Array = zg.NDArray(f32);

pub fn main() !void {
    const total_devices = zg.device.CudaDevice.device_count();
    std.debug.print("total_devices: {}\n", .{total_devices});

    for (0..total_devices) |i| {
        std.debug.print("Working: {}...\n", .{i});
    }

    var gpu1 = zg.device.CudaDevice.init(0);
    defer gpu1.deinit();

    var gpu2 = zg.device.CudaDevice.init(0);
    defer gpu2.deinit();
}
