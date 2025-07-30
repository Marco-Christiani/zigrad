const std = @import("std");
const zg = @import("zigrad");
const Py = @import("python.zig");
const lib = @import("lib.zig");
const NDTensor = zg.NDTensor;
const Graph = zg.Graph;

pub fn run_tests(T: type, device: zg.DeviceReference) !void {
    try test_smce_1d(T, device);
}

fn test_smce_1d(T: type, device: zg.DeviceReference) !void {
    var graph = Graph.init(std.heap.smp_allocator, .{});
    defer graph.deinit();

    const Tensor = NDTensor(T);
    const opts: zg.TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const input = try Tensor.from_slice(device, &.{ 1.0, 2.0, 3.0, 4.0 }, &.{4}, opts);
    defer input.deinit();

    const target = try Tensor.from_slice(device, &.{ 0.0, 1.0, 0.0, 0.0 }, &.{4}, opts);
    defer target.deinit();

    const loss = try zg.loss.softmax_cross_entropy_loss(T, input, target);
    defer loss.deinit();

    try loss.backward();

    var py_mod = try Py.create_module("test_smce_1d");
    defer py_mod.deinit();

    try py_mod.run(
        \\import torch
        \\input = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        \\target = torch.tensor([0.0, 1.0, 0.0, 0.0])
        \\loss_fn = torch.nn.CrossEntropyLoss()
        \\loss = loss_fn(input.unsqueeze(0), target.argmax().unsqueeze(0))
        \\loss.backward()
        \\loss_val = loss.item()
        \\input_grad = input.grad.numpy()
    );

    // const expected_loss = (try py_mod.get_slice(T, "loss_val"))[0];
    const loss_val = try py_mod.get_var("loss_val");
    const expected_loss = try loss_val.float(f32);
    const expected_grad = try py_mod.get_slice(T, "input_grad");

    try std.testing.expectApproxEqAbs(expected_loss, loss.get_data()[0], 1e-4);
    try lib.expectApproxEqRelSlices(T, expected_grad, input.assume_grad_data(), 1e-4);
}
