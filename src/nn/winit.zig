const std = @import("std");
const zg = @import("../root.zig");
const NDTensor = zg.NDTensor;
var rng = std.Random.DefaultPrng.init(zg.settings.seed);
const random = rng.random();

pub fn kaimingUniformInit(comptime T: type, tensor: *NDTensor(T), a: T) void {
    const fan_in = calculateFanIn(tensor);
    const bound = @sqrt(6.0 / fan_in) * a;

    for (tensor.data.data) |*value| {
        value.* = random.float(T) * 2 * bound - bound;
    }
}

pub fn biasInit(comptime T: type, tensor: *NDTensor(T)) void {
    const fan_in = calculateFanIn(tensor);
    const bound = 1 / @sqrt(fan_in);

    for (tensor.data.data) |*value| {
        value.* = random.float(T) * 2 * bound - bound;
    }
}

fn calculateFanIn(tensor: anytype) @TypeOf(tensor.data.data[0]) {
    const T = @TypeOf(tensor.data.data[0]);
    return switch (tensor.data.shape.len()) {
        1 => @as(T, @floatFromInt(tensor.data.shape.shape[0])),
        2 => @as(T, @floatFromInt(tensor.data.shape.shape[1])),
        4 => @as(T, @floatFromInt(tensor.data.shape.shape[1] * tensor.data.shape.shape[2] * tensor.data.shape.shape[3])),
        else => @panic("Unsupported tensor shape for fan_in calculation"),
    };
}
