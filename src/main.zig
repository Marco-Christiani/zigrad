const std = @import("std");
pub const zg = @import("zigrad");
const mnist = zg.mnist;
const layer = zg.layer;
pub const std_options = zg.std_options;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var gpu = zg.device.CudaDevice.init(0, gpa.allocator());
    defer gpu.deinit();

    const loss = gpu.loss.nll(f32, .{
        .dimensions = 1,
        .input_logits = true,
        .target_type = .index,
        .reduce_type = .SUM,
    });

    loss.forward(...);
    loss.backward(...);
}
