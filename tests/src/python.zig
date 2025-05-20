///! A minimal Python C API wrapper for verification testing
const std = @import("std");

const c = @cImport({
    @cInclude("Python.h");
});

pub const PyError = error{PythonError};

/// Initialize the Python interpreter, only do this once.
pub fn init() void {
    _ = c.Py_InitializeEx(0);
}

/// Finalize the Python interpreter, only do this once.
pub fn finalize() void {
    _ = c.Py_FinalizeEx();
}

/// Get the "__main__" module's dictionary
/// This is typically used as the globals dictionary for executing Python code
pub fn import_main() !*c.PyObject {
    return check(c.PyImport_AddModule("__main__"));
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
pub fn get_var_slice(globals: *c.PyObject, name: [:0]const u8, T: type) ![]const T {
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

pub fn eval_float(expr: [:0]const u8, globals: *c.PyObject) !f64 {
    const obj = try eval(expr, globals);
    defer obj.deinit();
    return try obj.to_float();
}

pub fn eval_slice(comptime T: type, expr: [:0]const u8, globals: *c.PyObject) ![]const T {
    const obj = try eval(expr, globals);
    defer obj.deinit();
    return try obj.get_slice(T);
}

/// Wrapper for a Python object with automatic reference counting
pub const PyObject = struct {
    obj: *c.PyObject,

    /// Decrement the reference count of the object
    /// Call this when you're done with the object
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

    /// Convert the Python object to a double-precision float
    /// Useful for getting scalar numeric values from Python
    pub fn to_float(self: PyObject) !f64 {
        const result = c.PyFloat_AsDouble(self.obj);
        if (result == -1.0 and c.PyErr_Occurred() != null) {
            c.PyErr_Print();
            return PyError.PythonError;
        }
        return result;
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
    return _len;
}
