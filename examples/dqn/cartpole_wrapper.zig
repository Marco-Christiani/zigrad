const std = @import("std");
const CartPole = @import("src/CartPole.zig");

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();
const T = f32;

pub export fn cartpole_init(seed: u64) ?*CartPole {
    const result = allocator.create(CartPole) catch @panic("init alloc failed");
    result.* = CartPole.init(seed);
    return result;
}

pub export fn cartpole_reset(pole: *CartPole) [*c]T {
    var state = pole.reset();
    const result = allocator.alloc(T, 4) catch return null;
    for (result, &state) |*r, *s| r.* = s.*;
    return result.ptr;
}

pub export fn cartpole_step(pole: *CartPole, action: u32, state: [*c]T, reward: [*c]T, done: [*c]u8) void {
    const result = pole.step(action);
    var dest = result.state;
    std.mem.copyForwards(T, &dest, state[0..4]);
    reward.* = result.reward;
    done.* = if (result.done == 1) 1 else 0;
}

pub export fn cartpole_delete(pole: *CartPole) void {
    allocator.destroy(pole);
}
