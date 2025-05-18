const std = @import("std");
const c = @cImport({
    @cInclude("Python.h");
});

pub const PyError = error{
    PythonError,
    BufferError,
    TypeError,
};

pub const Py = struct {
    pub fn init() void {
        _ = c.Py_InitializeEx(0);
    }

    pub fn finalize() void {
        _ = c.Py_FinalizeEx();
    }

    pub fn importMain() !*c.PyObject {
        return check(c.PyImport_AddModule("__main__"));
    }

    pub fn run(script: [:0]const u8, globals: *c.PyObject) !void {
        const res = try check(c.PyRun_String(script, c.Py_file_input, globals, globals));
        c.Py_DECREF(res);
    }

    pub fn eval(expr: [:0]const u8, globals: *c.PyObject) !PyObject {
        const obj = try check(c.PyRun_String(expr, c.Py_eval_input, globals, globals));
        return PyObject.fromBorrowed(obj);
    }

    pub fn dict() !PyObject {
        return PyObject.fromBorrowed(try check(c.PyDict_New()));
    }

    pub fn getVar(globals: *c.PyObject, name: [:0]const u8) !PyObject {
        const obj = try check(c.PyDict_GetItemString(globals, name));
        // PyDict_GetItemString returns a borrowed reference
        return PyObject.fromBorrowed(obj);
    }

    pub fn check(obj: ?*c.PyObject) !*c.PyObject {
        if (obj == null) {
            c.PyErr_Print();
            return PyError.PythonError;
        }
        return obj.?;
    }
};

pub const PyObject = struct {
    obj: *c.PyObject,
    owned: bool,

    pub fn fromBorrowed(obj: *c.PyObject) PyObject {
        c.Py_INCREF(obj);
        return PyObject{ .obj = obj, .owned = true };
    }

    pub fn fromOwned(obj: *c.PyObject) PyObject {
        return PyObject{ .obj = obj, .owned = true };
    }

    pub fn deinit(self: *PyObject) void {
        if (self.owned) {
            c.Py_DECREF(self.obj);
            self.owned = false;
        }
    }

    pub fn getattr(self: PyObject, name: [:0]const u8) !PyObject {
        const obj = try Py.check(c.PyObject_GetAttrString(self.obj, name));
        return PyObject.fromOwned(obj);
    }

    pub fn callMethod(self: PyObject, method: [:0]const u8) !PyObject {
        const obj = try Py.check(c.PyObject_CallMethod(self.obj, method, null));
        return PyObject.fromOwned(obj);
    }

    pub fn getBuffer(self: PyObject, format: c_int) !PyBuffer {
        var buffer = PyBuffer{};
        const result = c.PyObject_GetBuffer(self.obj, &buffer.view, format);
        if (result != 0) {
            c.PyErr_Print();
            return PyError.BufferError;
        }
        return buffer;
    }

    pub fn toFloat(self: PyObject) !f64 {
        const result = c.PyFloat_AsDouble(self.obj);
        if (result == -1.0 and c.PyErr_Occurred() != null) {
            c.PyErr_Print();
            return PyError.TypeError;
        }
        return result;
    }
};

pub const PyBuffer = struct {
    view: c.Py_buffer = .{},

    pub fn deinit(self: *PyBuffer) void {
        c.PyBuffer_Release(&self.view);
    }

    pub fn asSlice(self: PyBuffer, comptime T: type) ![]const T {
        if (self.view.ndim < 1 or self.view.itemsize != @sizeOf(T)) {
            return PyError.BufferError;
        }

        const data_ptr: [*]const T = @ptrCast(@alignCast(self.view.buf orelse return PyError.BufferError));
        const len = computeBufferLength(&self.view);
        return data_ptr[0..len];
    }

    pub fn asReshapedSlice(self: PyBuffer, comptime T: type, dims: []const usize) ![]const T {
        const slice = try self.asSlice(T);

        var expected_len: usize = 1;
        for (dims) |dim| {
            expected_len *= dim;
        }

        if (slice.len != expected_len) {
            return PyError.BufferError;
        }

        return slice;
    }

    fn computeBufferLength(view: *const c.Py_buffer) usize {
        if (view.ndim == 0) return 0;

        var len: usize = 1;
        var i: usize = 0;
        while (i < view.ndim) : (i += 1) {
            len *= @intCast(view.shape[i]);
        }
        return len;
    }
};
