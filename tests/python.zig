const std = @import("std");
const c = @cImport({
    @cInclude("Python.h");
});

pub const PyError = error{PythonError};

pub fn init() void {
    _ = c.Py_InitializeEx(0);
}

pub fn finalize() void {
    _ = c.Py_FinalizeEx();
}

pub fn check(obj: ?*c.PyObject) !*c.PyObject {
    if (obj == null) {
        c.PyErr_Print();
        return PyError.PythonError;
    }
    return obj.?;
}

pub fn importMain() !*c.PyObject {
    return check(c.PyImport_AddModule("__main__"));
}

pub fn run(script: [:0]const u8, globals: *c.PyObject) !void {
    const res = try check(c.PyRun_String(script, c.Py_file_input, globals, globals));
    _ = res;
}

pub fn getVar(globals: *c.PyObject, name: [:0]const u8) !*c.PyObject {
    return check(c.PyDict_GetItemString(globals, name));
}

pub fn getattr(obj: *c.PyObject, name: [:0]const u8) !*c.PyObject {
    return check(c.PyObject_GetAttrString(obj, name));
}

pub fn callMethod(obj: *c.PyObject, method: [:0]const u8) !*c.PyObject {
    return check(c.PyObject_CallMethod(obj, method, null));
}

pub fn getBuffer(obj: *c.PyObject, view: *c.Py_buffer) !void {
    if (c.PyObject_GetBuffer(obj, view, c.PyBUF_FORMAT | c.PyBUF_ANY_CONTIGUOUS) != 0) {
        c.PyErr_Print();
        return PyError.PythonError;
    }
}

pub fn releaseBuffer(view: *c.Py_buffer) void {
    c.PyBuffer_Release(view);
}
