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

    const reference = gpu.reference();

    const A = try zg.NDTensor(f32).empty(&.{ 3, 64, 64 }, false, reference);
    const x = try zg.NDTensor(f32).empty(&.{ 3, 64 }, false, reference);
    gpu.mem_sequence(f32, A.get_data(), 0.0, 1.0);

    gpu.blas.reduce(f32, A.get_data(), A.get_shape(), x.get_data(), x.get_shape(), &.{1}, 1.0, 0.0, .add);

    x.print();
}
