const std = @import("std");
const zg = @import("zigrad");
const Py = @import("python.zig");
const lib = @import("lib.zig");
const c = @cImport(@cInclude("Python.h"));

pub fn main() !void {
    Py.init();
    defer Py.finalize();

    const T = f32;
    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();

    try self_test();
    try integration_test(T, cpu.reference());

    try @import("ndtensor.zig").run_tests(T, cpu.reference());
    try @import("loss.zig").run_tests(T, cpu.reference());

    if (zg.has_cuda) {
        var cuda = zg.device.CudaDevice.init();
        defer cuda.deinit();
        try integration_test(T, cuda.reference());
        try @import("ndtensor.zig").main();
        try @import("loss.zig").main();
    }
    std.debug.print("Verified.\n", .{});
}

test {
    Py.init();
    defer Py.finalize();
    try self_test();

    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();
    try integration_test(f32, cpu.reference());
}

fn self_test() !void {
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

fn integration_test(T: type, device: zg.DeviceReference) !void {
    const Tensor = zg.NDTensor(T);

    var graph = zg.Graph.init(std.heap.smp_allocator, .{ .eager_teardown = false });
    defer graph.deinit();

    // Zigrad
    const x = try Tensor.from_slice(device, &.{ -2.0, -0.5, 0.5, 2.0 }, &.{ 2, 2 }, .{
        .requires_grad = true,
        .graph = &graph,
    });
    defer x.deinit();

    const y = try x.clamp(-1.0, 1.0);
    defer y.deinit();

    try y.backward();

    const zigrad_out = y.get_data();
    const zigrad_grad = x.assume_grad_data();

    // Torch
    const mod = try Py.import_main();

    const script_fmt =
        \\import torch
        \\x = torch.tensor([-2.0, -0.5, 0.5, 2.0], dtype=torch.{s}, requires_grad=True).reshape(2, 2)
        \\x.retain_grad()
        \\y = x.clamp(-1.0, 1.0)
        \\y.retain_grad()
        \\y.sum().backward()
        \\yval = y.detach().numpy()
    ;

    const torch_dtype = switch (T) {
        f32 => "float32",
        f64 => "float64",
        inline else => @compileError("Unsupported type" ++ @typeName(T)),
    };

    var buf: [script_fmt.len + torch_dtype.len:0]u8 = undefined;

    const script = try std.fmt.bufPrintZ(&buf, script_fmt, .{torch_dtype});
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
