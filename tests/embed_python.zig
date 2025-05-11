const std = @import("std");
const python = @import("python.zig");
const c = @cImport({
    @cInclude("Python.h");
});

pub fn main() !void {
    python.init();
    defer python.finalize();

    const main_mod = try python.importMain();
    const globals = c.PyModule_GetDict(main_mod);

    const script =
        \\import torch
        \\x = torch.arange(0, 10000, dtype=torch.float32).reshape(100, 100)
        \\x.requires_grad_()
        \\y = (x * x).sum()
        \\y.backward()
    ;
    try python.run(script, globals);

    const x_obj = try python.getVar(globals, "x");
    const grad_obj = try python.getattr(x_obj, "grad");
    const numpy_obj = try python.callMethod(grad_obj, "numpy");

    var view: c.Py_buffer = .{};
    try python.getBuffer(numpy_obj, &view);
    defer python.releaseBuffer(&view);

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

    const code_obj = try python.check(c.Py_CompileString("x.grad.sum().item()", "<string>", c.Py_eval_input));

    _ = c.PyObject_Print(code_obj, c.stderr, 0);
    std.debug.print("\n", .{});
    const result_obj = try python.check(c.PyEval_EvalCode(code_obj, globals, globals));
    _ = c.PyObject_Print(result_obj, c.stderr, 0);
    std.debug.print("\n", .{});
}
