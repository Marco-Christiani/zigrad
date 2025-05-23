const std = @import("std");
const zg = @import("zigrad");
const Py = @import("python.zig");
const NDTensor = zg.NDTensor;
const Graph = zg.Graph;

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
    var graph = Graph.init(allocator, .{});
    defer graph.deinit();

    const Tensor = NDTensor(T);

    const input = try Tensor.from_slice(&graph, device, &.{ 1, 2, 3, 4 }, null, .{
        .requires_grad = true,
    });
    defer input.deinit();
    defer input.deinit();

    const sum_result = try input.sum();

    try std.testing.expectEqualSlices(f32, &.{10}, sum_result.data.data);
    try sum_result.backward();

    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1 }, input.assume_grad_data());

    var py_mod = try Py.create_module("test_sum");
    defer py_mod.deinit();
    try py_mod.run(
        \\import torch
        \\inp = torch.tensor([1, 2, 3, 4], dtype=torch.float32, requires_grad=True)
        \\inp.retain_grad()
        \\sum_result = inp.sum()
        \\sum_result.retain_grad()
        \\sum_result.backward()
    );
    const expected_sum = try py_mod.eval_float("sum_result.detach().numpy()");
    const expected_sum_grad = try py_mod.eval_float("sum_result.grad.numpy()");
    try std.testing.expectEqualSlices(T, &.{@floatCast(expected_sum)}, sum_result.get_data());
    try std.testing.expectEqualSlices(T, &.{@floatCast(expected_sum_grad)}, sum_result.assume_grad_data());
}
