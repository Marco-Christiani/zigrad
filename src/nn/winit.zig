const zg = @import("../zigrad.zig");
const NDTensor = zg.NDTensor;
// const random = zg.random;
const std = @import("std");
const random = std.crypto.random;

pub fn he_init(comptime T: type, tensor: *const NDTensor(T)) void {
    const fan_in: T = @floatFromInt(tensor.data.shape.shape[1]);
    const std_dev = @sqrt(2.0 / fan_in);
    for (tensor.data.data) |*value| {
        value.* = random.floatNorm(T) * std_dev;
    }
}
