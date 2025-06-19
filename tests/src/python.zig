///! A minimal Python C API wrapper for verification testing
const std = @import("std");

const c = switch (@import("builtin").target.os.tag) {
    // .macos => @cImport(@cInclude("/opt/homebrew/Cellar/python@3.11/3.11.10/Frameworks/Python.framework/Versions/3.11/include/python3.11/Python.h")),
    inline else => @cImport(@cInclude("Python.h")),
};

pub const PyError = error{PythonError};
var mutex = std.Thread.Mutex{};
var initialized = false;
globals: c.PyObject,

/// Initialize the Python interpreter, only do this once.
pub fn init() void {
    var mtx = &mutex;
    mtx.lock();
    defer mtx.unlock();
    if (initialized) return;
    _ = c.Py_InitializeEx(0);
    initialized = true;
}

/// Finalize the Python interpreter, only do this once.
pub fn finalize() void {
    var mtx = &mutex;
    mtx.lock();
    defer mtx.unlock();
    if (!initialized) return;
    _ = c.Py_FinalizeEx();
    initialized = false;
}

/// Get the "__main__" module's dictionary
pub fn import_main() !PyModule {
    const result = try check(c.PyImport_AddModule("__main__"));
    return PyModule{ .obj = result };
}

/// Create a new Python module with a clean namespace
pub fn create_module(name: [:0]const u8) !PyModule {
    init();
    const module = try check(c.PyModule_New(name));
    const dict = c.PyModule_GetDict(module);
    const builtins = try check(c.PyImport_ImportModule("builtins"));
    _ = c.PyDict_SetItemString(dict, "__builtins__", builtins);
    c.Py_DECREF(builtins);
    return PyModule{ .obj = module };
}

/// Run a Python script in the given globals dictionary
pub fn run(script: [:0]const u8, globals: *c.PyObject) !void {
    const res = try check(c.PyRun_String(script, c.Py_file_input, globals, globals));
    c.Py_DECREF(res); // Clean up the new reference immediately
}

/// Get a variable from the Python globals dictionary by name
/// Returns a new reference to the variable
pub fn get_var(globals: *c.PyObject, name: [:0]const u8) !PyObject {
    const obj = try check(c.PyDict_GetItemString(globals, name));
    c.Py_INCREF(obj); // GetItemString returns borrowed reference
    return PyObject{ .obj = obj };
}

/// Convenience method to get a variable as a buffer directly
pub fn get_var_buffer(globals: *c.PyObject, name: [:0]const u8) !PyBuffer {
    var py_obj = try get_var(globals, name);
    defer py_obj.deinit();
    return py_obj.get_buffer();
}

/// Convenience method to get a variable as a slice directly
pub fn get_var_slice(globals: *c.PyObject, T: type, name: [:0]const u8) ![]const T {
    var py_obj = try get_var(globals, name);
    defer py_obj.deinit();
    return try py_obj.get_slice(T);
}

/// Check if a Python object is not null and return it
/// If the object is null, prints the Python error and returns an error
pub fn check(obj: ?*c.PyObject) !*c.PyObject {
    if (obj == null) {
        c.PyErr_Print();
        return PyError.PythonError;
    }
    return obj.?;
}

/// Evaluate a Python expression and return the resulting PyObject wrapper.
pub fn eval(expr: [:0]const u8, globals: *c.PyObject) !PyObject {
    const obj = try check(c.PyRun_String(expr, c.Py_eval_input, globals, globals));
    return PyObject{ .obj = obj };
}

pub fn eval_f64(expr: [:0]const u8, globals: *c.PyObject) !f64 {
    const obj = try eval(expr, globals);
    defer obj.deinit();
    return try obj.double();
}

pub fn eval_slice(comptime T: type, expr: [:0]const u8, globals: *c.PyObject) ![]const T {
    const obj = try eval(expr, globals);
    defer obj.deinit();
    return try obj.get_slice(T);
}

/// Wrapper for a Python module so tests can create a clean namespace
pub const PyModule = struct {
    const Self = @This();
    obj: *c.PyObject,

    /// Clean up
    pub fn deinit(self: *PyModule) void {
        c.Py_DECREF(self.obj);
    }

    /// Run Python code in this module's namespace
    pub fn run(self: Self, script: [:0]const u8) !void {
        const dict = c.PyModule_GetDict(self.obj); // Borrowed reference
        const res = try check(c.PyRun_String(script, c.Py_file_input, dict, dict));
        c.Py_DECREF(res);
    }

    /// Evaluate a Python expression in this module's namespace
    pub fn eval(self: Self, expr: [:0]const u8) !PyObject {
        const dict = c.PyModule_GetDict(self.obj); // Borrowed reference
        const obj = try check(c.PyRun_String(expr, c.Py_eval_input, dict, dict));
        return PyObject{ .obj = obj };
    }

    /// Evaluate a Python expression evaluating to a slice-coerceable in this module's namespace
    pub fn eval_slice(self: Self, T: type, expr: [:0]const u8) ![]const T {
        const pyobj = try self.eval(expr);
        return try pyobj.get_slice(T);
    }

    /// Evaluate a Python expression evaluating to a float-coerceable in this module's namespace
    pub fn eval_float(self: Self, expr: [:0]const u8) !f64 {
        const pyobj = try self.eval(expr);
        return try pyobj.double();
    }

    /// Get a variable from this module by name
    pub fn get_var(self: Self, name: [:0]const u8) !PyObject {
        const dict = self._dict();
        const obj = try check(c.PyDict_GetItemString(dict, name)); // borrowed reference
        c.Py_INCREF(obj);
        return PyObject{ .obj = obj };
    }

    /// Internal use only for temporary access
    fn _dict(self: Self) *c.PyObject {
        return c.PyModule_GetDict(self.obj); // borrowed
    }

    /// Get a reference to the dict.
    pub fn get_dict(self: Self) PyObject {
        const dict = c.PyModule_GetDict(self.obj); // borrowed
        c.Py_IncRef(dict); // User API
        return PyObject{ .obj = dict };
    }

    /// Get a variable from this module as a slice
    pub fn get_slice(self: PyModule, T: type, name: [:0]const u8) ![]const T {
        return try get_var_slice(self._dict(), T, name);
    }
};

/// Wrapper for a Python object with automatic reference counting
pub const PyObject = struct {
    obj: *c.PyObject,

    /// Clean up
    pub fn deinit(self: *const PyObject) void {
        c.Py_DECREF(self.obj);
    }

    /// Get an attribute of the Python object by name
    /// Returns a new reference to the attribute
    pub fn get_attr(self: PyObject, name: [:0]const u8) !PyObject {
        const obj = try check(c.PyObject_GetAttrString(self.obj, name));
        return PyObject{ .obj = obj };
    }

    /// Call a method on the Python object with no arguments
    /// Returns a new reference to the result
    pub fn call_method(self: PyObject, name: [:0]const u8) !PyObject {
        const obj = try check(c.PyObject_CallMethod(self.obj, name, null));
        return PyObject{ .obj = obj };
    }

    /// Get a buffer view of the Python object
    /// This is useful for accessing numpy arrays or other buffer protocol objects
    pub fn get_buffer(self: PyObject) !PyBuffer {
        var buffer = PyBuffer{};
        // Use standard flags for numerically typed data
        const flags = c.PyBUF_FORMAT | c.PyBUF_ANY_CONTIGUOUS;
        const result = c.PyObject_GetBuffer(self.obj, &buffer.view, flags);
        if (result != 0) {
            c.PyErr_Print();
            return PyError.PythonError;
        }
        return buffer;
    }

    /// Get a buffer view of the Python object as a slice
    pub fn get_slice(self: PyObject, T: type) ![]const T {
        var buffer: c.Py_buffer = .{};
        const flags = c.PyBUF_FORMAT | c.PyBUF_ANY_CONTIGUOUS;
        const result = c.PyObject_GetBuffer(self.obj, &buffer, flags);
        if (result != 0) {
            c.PyErr_Print();
            return PyError.PythonError;
        }

        const data_ptr: [*]const T = @ptrCast(@alignCast(buffer.buf orelse return PyError.PythonError));
        return data_ptr[0..buffer_size(buffer)];
    }

    // pub const Dtype = union(enum) {
    //     F32 = f32,
    //     F64 = f64,
    // };

    /// Convert the Python object to a double-precision float
    /// Useful for getting scalar numeric values from Python
    // pub fn to_scalar(self: PyObject, dtype: Dtype) !dtype {
    // const result = switch (dtype) {
    //     .F32 => c.PyFloat_AsDouble(self.obj),
    //     .F64 => c.PyLong_AsDouble(self.obj),
    // };
    pub fn double(self: PyObject) !f64 {
        const result = c.PyFloat_AsDouble(self.obj);
        if (result == -1.0 and c.PyErr_Occurred() != null) {
            c.PyErr_Print();
            return PyError.PythonError;
        }
        return result;
    }

    pub fn float(self: PyObject, T: type) !T {
        return @floatCast(try self.double());
    }
};

/// Wrapper for a Python buffer view
/// This is used to access the data of buffer protocol objects like numpy arrays
pub const PyBuffer = struct {
    view: c.Py_buffer = .{},

    /// Release the backing Python buffer object
    pub fn deinit(self: *PyBuffer) void {
        c.PyBuffer_Release(&self.view);
    }

    /// Get the total number of elements in the buffer with a sanity check
    pub fn len(self: PyBuffer) usize {
        const calc_len = buffer_size(self.view);
        std.debug.assert(self.view.len, calc_len);
        return calc_len;
    }

    /// Get a slice of the buffer data as a specific type
    /// The type must match the size of the buffer items
    pub fn slice(self: PyBuffer, comptime T: type) ![]const T {
        if (self.view.ndim < 1 or self.view.itemsize != @sizeOf(T)) {
            return PyError.PythonError;
        }

        const data_ptr: [*]const T = @ptrCast(@alignCast(self.view.buf orelse return PyError.PythonError));
        return data_ptr[0..buffer_size(self.view)];
    }
};

fn buffer_size(view: c.Py_buffer) usize {
    if (view.ndim == 0) return 0;

    var _len: usize = 1;
    var i: usize = 0;
    while (i < view.ndim) : (i += 1) {
        _len *= @intCast(view.shape[i]);
    }
    std.debug.assert(@as(usize, @intCast(view.len)) == _len * @as(usize, @intCast(view.itemsize)));
    std.debug.assert(@as(usize, @intCast(@divTrunc(view.len, view.itemsize))) == _len);
    return _len;
}
