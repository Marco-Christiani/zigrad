const std = @import("std");
const zg = @import("zigrad");
const Py = @import("python.zig");
const NDTensor = zg.NDTensor;
const GraphManager = zg.GraphManager;

pub fn main() !void {
    const T = f32;
    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();
    try test_sum(T, cpu.reference(), std.heap.smp_allocator);
    switch (zg.backend) {
        .CUDA => {
            var cuda = zg.device.CudaDevice.init();
            defer cuda.deinit();
            try test_sum(T, cuda.reference(), std.heap.smp_allocator);
        },
        .HOST => {},
    }
}

fn test_sum(T: type, device: zg.DeviceReference, allocator: std.mem.Allocator) !void {
    var gm = GraphManager.init(allocator, .{});
    defer gm.deinit();

    const Tensor = NDTensor(T);

    const input = try Tensor.from_slice(&.{ 1, 2, 3, 4 }, &.{4}, .{
        .requires_grad = true,
        .device = device,
        .node_allocator = gm.heap(),
    });
    defer input.deinit();

    const sum_result = try input.sum();

    try std.testing.expectEqualSlices(f32, &.{10}, sum_result.data.data);

    try gm.backward(sum_result);

    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1 }, input.assume_grad_data());

    var py_mod = try Py.create_module("test_module");
    defer py_mod.deinit();
    try py_mod.run(
        \\import torch
        \\inp = torch.tensor([1, 2, 3, 4], dtype=torch.float32, requires_grad=True)
        \\inp.retain_grad()
        \\sum_result = inp.sum()
        \\sum_result.retain_grad()
        \\sum_result.backward()
        \\print("sum_result:", sum_result.detach().numpy())
        \\print("sum_result.grad:", sum_result.grad.numpy())
        \\print("inp.grad:", inp.grad)
        \\print("inp:", inp)
    );
    const expected_sum = try py_mod.eval_float("sum_result.detach().numpy()");
    const expected_sum_grad = try py_mod.eval_float("sum_result.grad.numpy()");
    std.debug.print("sum_result: {d}\n", .{sum_result.get_data()});
    std.debug.print("expected_sum: {d}\n", .{expected_sum});
    std.debug.print("sum_result.grad: {d}\n", .{sum_result.assume_grad_data()});
    std.debug.print("expected_sum_grad: {d}\n", .{expected_sum_grad});
    try std.testing.expectEqualSlices(T, &.{@floatCast(expected_sum)}, sum_result.get_data());
    try std.testing.expectEqualSlices(T, &.{@floatCast(expected_sum_grad)}, sum_result.assume_grad_data());
}
