const std = @import("std");
pub const zg = @import("zigrad");
const mnist = zg.mnist;
const layer = zg.layer;
pub const std_options = zg.std_options;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var cpu = zg.device.HostDevice.init(gpa.allocator());
    defer cpu.deinit();

    const x = try zg.NDTensor(f32).from_cache(&.{ 10, 10 }, true, cpu.reference());
    defer x.to_cache();
}
