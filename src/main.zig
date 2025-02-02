const std = @import("std");
pub const zg = @import("zigrad");
const mnist = zg.mnist;
const layer = zg.layer;
pub const std_options = zg.std_options;

const Model = struct {
    pub fn forward(_: anytype) void {}
};

const Tensor = zg.NDTensor(f32);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    //
    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();
    //
    var gpu = zg.device.CudaDevice.init(0, arena.allocator());
    defer gpu.deinit();

    const A = try Tensor.sequence(0.0, 1.0, &.{ 1, 2, 2 }, false, gpu.reference());
    defer A.deinit();

    const B = try Tensor.sequence(0.0, 1.0, &.{ 1, 2, 2 }, false, gpu.reference());
    defer B.deinit();

    A.print();
    B.print();
    //
    // gpu.capture.open(.{});
    //
    // const C = try A.bmm(B, .{
    //     .trans_a = false,
    //     .trans_b = false,
    // });
    //
    // defer C.deinit();
    //
    // gpu.capture.save();
    //
    // std.time.sleep(std.time.ns_per_s * 3);
    //
    // try gpu.capture.run();

    //C.print();
}
