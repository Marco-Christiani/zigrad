const std = @import("std");
const zg = @import("zigrad");
const Py = @import("python.zig");

const c = switch (@import("builtin").target.os.tag) {
    // .macos => @cImport(@cInclude("/opt/homebrew/Cellar/python@3.11/3.11.10/Frameworks/Python.framework/Versions/3.11/include/python3.11/Python.h")),
    inline else => @cImport(@cInclude("Python.h")),
};

pub fn main() !void {
    Py.init();
    defer Py.finalize();

    Py.init();
    defer Py.finalize();
    try unit_test(std.heap.smp_allocator);
    try test_with_clean_module();

    try @import("ndtensor.zig").main();
}

test {
    Py.init();
    defer Py.finalize();
    try unit_test(std.testing.allocator);
    try test_with_clean_module();
}

fn unit_test(allocator: std.mem.Allocator) !void {
    const T = f32;
    const Tensor = zg.NDTensor(T);

    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = false });
    defer graph.deinit();

    // Zigrad
    const x = try Tensor.from_slice(&graph, cpu.reference(), &.{ -2.0, -0.5, 0.5, 2.0 }, &.{ 2, 2 }, .{
        .requires_grad = true,
    });
    defer x.deinit();

    const y = try x.clamp(-1.0, 1.0);
    defer y.deinit();

    try y.backward();

    const zigrad_out = y.get_data();
    const zigrad_grad = x.assume_grad_data();

    // Torch
    const mod = try Py.import_main();

    const script =
        \\import torch
        \\x = torch.tensor([-2.0, -0.5, 0.5, 2.0], dtype=torch.float32, requires_grad=True).reshape(2, 2)
        \\x.retain_grad()
        \\y = x.clamp(-1.0, 1.0)
        \\y.retain_grad()
        \\y.sum().backward()
        \\yval = y.detach().numpy()
    ;
    try mod.run(script);

    var py_x = try mod.get_var("x");
    defer py_x.deinit();

    var py_y = try mod.get_var("y");
    defer py_y.deinit();

    var py_y_detached = try py_y.call_method("detach");
    defer py_y_detached.deinit();

    var py_y_np = try py_y_detached.call_method("numpy");
    defer py_y_np.deinit();

    var py_x_grad = try py_x.get_attr("grad");
    defer py_x_grad.deinit();

    var py_x_grad_np = try py_x_grad.call_method("numpy");
    defer py_x_grad_np.deinit();

    var y_buffer = try py_y_np.get_buffer();
    defer y_buffer.deinit();

    var grad_buffer = try py_x_grad_np.get_buffer();
    defer grad_buffer.deinit();

    const py_y_slice = try y_buffer.slice(f32);
    const py_grad_slice = try grad_buffer.slice(f32);
    const y_buffer2 = try py_y_np.get_slice(f32);
    const y_slice3 = try mod.get_slice(f32, "yval");

    try std.testing.expectEqualSlices(f32, py_y_slice, y_buffer2);
    try std.testing.expectEqualSlices(f32, py_y_slice, y_slice3);
    // Compare
    try std.testing.expectEqualSlices(f32, py_y_slice, zigrad_out);
    try std.testing.expectEqualSlices(f32, py_grad_slice, zigrad_grad);
}

fn test_with_clean_module() !void {
    var py_mod = try Py.create_module("test_module");
    defer py_mod.deinit();
    try py_mod.run(
        \\import torch
        \\import numpy as np
        \\x = torch.tensor([1, 2, 3], dtype=torch.float32)
        \\y = x * 2
        \\result = y.numpy()
    );
    const result_slice = try py_mod.get_slice(f32, "result");
    try std.testing.expectEqualSlices(f32, &[_]f32{ 2, 4, 6 }, result_slice);
}
