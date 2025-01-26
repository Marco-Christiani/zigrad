const std = @import("std");
pub const zg = @import("zigrad");
const mnist = zg.mnist;
const layer = zg.layer;
pub const std_options = zg.std_options;

pub const zigrad_settings: zg.Settings = .{
    .caching_policy = null,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var gpu = zg.device.CudaDevice.init(0, gpa.allocator());
    defer gpu.deinit();

    const A_host = [_]f32{ 1, 2, 3, 4, 1, 2, 3, 4 };
    const B_host = [_]f32{ 1, 2, 3, 4, 1, 2, 3, 4 };

    const A = try zg.NDTensor(f32).empty(&.{ 2, 2, 2 }, false, gpu.reference());
    defer A.deinit();

    const x = try zg.NDTensor(f32).empty(&.{ 2, 2, 2 }, false, gpu.reference());
    defer A.deinit();

    gpu.mem_transfer(f32, A_host[0..], A.get_data(), .HtoD);
    gpu.mem_transfer(f32, B_host[0..], x.get_data(), .HtoD);

    const y = try A.bmm(x, .{
        .trans_a = false,
        .trans_b = false,
    });
    defer y.deinit();

    y.print();
}
