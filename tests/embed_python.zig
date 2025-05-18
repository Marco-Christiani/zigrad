const std = @import("std");
const zg = @import("zigrad");
const python = @import("python.zig");
const Py = python.Py;
const c = @cImport({
    @cInclude("Python.h");
});

pub fn main() !void {
    Py.init();
    defer Py.finalize();

    const main_mod = try Py.importMain();
    const globals = c.PyModule_GetDict(main_mod);

    const script =
        \\import torch
        \\x = torch.arange(0, 10000, dtype=torch.float32).reshape(100, 100)
        \\x.requires_grad_()
        \\y = (x * x).sum()
        \\y.backward()
    ;
    try Py.run(script, globals);

    const x_obj = try Py.getVar(globals, "x");
    const grad_obj = try Py.getattr(x_obj, "grad");
    const numpy_obj = try Py.callMethod(grad_obj, "numpy");

    var view: c.Py_buffer = .{};
    try Py.getBuffer(numpy_obj, &view);
    defer Py.releaseBuffer(&view);

    if (view.ndim != 2 or view.itemsize != @sizeOf(f32)) {
        std.debug.print("Unexpected tensor shape/itemsize\n", .{});
        return error.UnexpectedShape;
    }

    const rows: usize = @intCast(view.shape[0]);
    const cols: usize = @intCast(view.shape[1]);
    const count = rows * cols;

    const flat_ptr: [*]const f32 = @ptrCast(@alignCast(view.buf orelse return error.BadBuffer));
    const sample = flat_ptr[0..count];

    std.debug.print("grad w.r.t x [0..8] = ", .{});
    for (0..8) |i| {
        std.debug.print("{d:.1} ", .{sample[i]});
    }
    std.debug.print("\n", .{});

    const code_obj = try Py.check(c.Py_CompileString("x.grad.sum().item()", "<string>", c.Py_eval_input));

    _ = c.PyObject_Print(code_obj, c.stderr, 0);
    std.debug.print("\n", .{});
    const result_obj = try Py.check(c.PyEval_EvalCode(code_obj, globals, globals));
    _ = c.PyObject_Print(result_obj, c.stderr, 0);
    std.debug.print("\n", .{});
}

test {
    Py.init();
    defer Py.finalize();
    try test2();
}

fn test2() !void {
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
    const globals = try Py.importMain();
    const dict = c.PyModule_GetDict(globals);

    const script =
        \\import torch
        \\x = torch.tensor([-2.0, -0.5, 0.5, 2.0], dtype=torch.float32, requires_grad=True).reshape(2, 2)
        \\x.retain_grad()
        \\y = x.clamp(-1.0, 1.0)
        \\y.retain_grad()
        \\y.sum().backward()
    ;
    try Py.run(script, dict);

    var py_x = try Py.getVar(dict, "x");
    defer py_x.deinit();

    var py_y = try Py.getVar(dict, "y");
    defer py_y.deinit();

    var py_y_detached = try py_y.callMethod("detach");
    defer py_y_detached.deinit();

    var py_y_np = try py_y_detached.callMethod("numpy");
    defer py_y_np.deinit();

    var py_x_grad = try py_x.getattr("grad");
    defer py_x_grad.deinit();

    var py_x_grad_np = try py_x_grad.callMethod("numpy");
    defer py_x_grad_np.deinit();

    var y_buffer = try py_y_np.getBuffer(c.PyBUF_FORMAT | c.PyBUF_ANY_CONTIGUOUS);
    defer y_buffer.deinit();

    var grad_buffer = try py_x_grad_np.getBuffer(c.PyBUF_FORMAT | c.PyBUF_ANY_CONTIGUOUS);
    defer grad_buffer.deinit();

    const py_y_slice = try y_buffer.asReshapedSlice(f32, &.{ 2, 2 });
    const py_grad_slice = try grad_buffer.asReshapedSlice(f32, &.{ 2, 2 });

    // Compare
    try std.testing.expectEqualSlices(f32, py_y_slice, zigrad_out);
    try std.testing.expectEqualSlices(f32, py_grad_slice, zigrad_grad);
}
