const std = @import("std");
const zg = @import("zigrad");
const Py = @import("python.zig");
const c = @cImport({
    @cInclude("Python.h");
});

pub fn main() !void {
    Py.init();
    defer Py.finalize();

    const main_mod = try Py.import_main();
    const globals = c.PyModule_GetDict(main_mod);

    const script =
        \\import torch
        \\x = torch.arange(0, 10000, dtype=torch.float32).reshape(100, 100)
        \\x.requires_grad_()
        \\y = (x * x).sum()
        \\y.backward()
    ;
    try Py.run(script, globals);

    const x_obj = try Py.get_var(globals, "x");
    const grad_obj = try x_obj.get_attr("grad");
    const numpy_obj = try grad_obj.call_method("numpy");

    const pybuf = try numpy_obj.get_buffer();
    const pybuf_slice = try pybuf.slice(f32);

    std.debug.print("grad w.r.t x [0..8] = ", .{});
    for (0..8) |i| {
        std.debug.print("{d:.1} ", .{pybuf_slice[i]});
    }
    std.debug.print("\n", .{});

    const code_obj = try Py.check(c.Py_CompileString("x.grad.sum().item()", "<string>", c.Py_eval_input));
    const result_obj = try Py.check(c.PyEval_EvalCode(code_obj, globals, globals));
    if (c.PyFloat_Check(result_obj) == 1) {
        const value = c.PyFloat_AsDouble(result_obj);
        std.debug.print("{d}\n", .{value});
        const v2 = try Py.eval_float("x.grad.sum().item()", globals);
        std.debug.assert(value == v2);
    }
}

test {
    Py.init();
    defer Py.finalize();
    try unit_test();
}

fn unit_test() !void {
    const T = f32;
    const Tensor = zg.NDTensor(T);

    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();

    var gm = zg.GraphManager.init(std.testing.allocator, .{ .eager_teardown = false });
    defer gm.deinit();

    // Zigrad
    const x = try Tensor.from_slice(&.{ -2.0, -0.5, 0.5, 2.0 }, &.{ 2, 2 }, .{
        .device = cpu.reference(),
        .node_allocator = gm.heap(),
        .requires_grad = true,
    });
    defer x.deinit();

    const y = try x.clamp(-1.0, 1.0);
    defer y.deinit();

    try gm.backward(y);

    const zigrad_out = y.get_data();
    const zigrad_grad = x.assume_grad_data();

    // Torch
    const globals = try Py.import_main();
    const dict = c.PyModule_GetDict(globals);

    const script =
        \\import torch
        \\x = torch.tensor([-2.0, -0.5, 0.5, 2.0], dtype=torch.float32, requires_grad=True).reshape(2, 2)
        \\x.retain_grad()
        \\y = x.clamp(-1.0, 1.0)
        \\y.retain_grad()
        \\y.sum().backward()
        \\yval = y.detach().numpy()
    ;
    try Py.run(script, dict);

    var py_x = try Py.get_var(dict, "x");
    defer py_x.deinit();

    var py_y = try Py.get_var(dict, "y");
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
    const y_slice3 = try Py.get_var_slice(dict, "yval", f32);

    try std.testing.expectEqualSlices(f32, py_y_slice, y_buffer2);
    try std.testing.expectEqualSlices(f32, py_y_slice, y_slice3);
    // Compare
    try std.testing.expectEqualSlices(f32, py_y_slice, zigrad_out);
    try std.testing.expectEqualSlices(f32, py_grad_slice, zigrad_grad);
}
