const std = @import("std");
pub const zg = @import("zigrad");
const mnist = zg.mnist;
const layer = zg.layer;
pub const std_options = zg.std_options;

pub fn transpose(t: anytype) [2]usize {
    const shape = t.get_shape();
    var s: [2]usize = shape[0..2].*;
    std.mem.swap(usize, &s[0], &s[1]);
    return s;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var cpu = zg.device.HostDevice.init(0, gpa.allocator());
    defer cpu.deinit();
}
