//! TVM runtime module and tensor wrappers.
//!
//! Covers `tvm/runtime/module.h` (RuntimeModule) and `tvm/runtime/ndarray.h`
//! (Tensor via DLPack bridge).
const std = @import("std");
const api = @import("api.zig");
const c = @import("c.zig");
const dlpack = @import("../dlpack.zig");
const Value = api.Value;
const ObjectHandle = api.ObjectHandle;
const TvmError = api.TvmError;

const helpers = api.helpers;
const log = std.log.scoped(.@"zg/tvm_runtime");

// RuntimeModule

pub const RuntimeModule = struct {
    handle: ObjectHandle,

    pub const deinit = helpers.deinit(RuntimeModule);
    pub const as_value = helpers.as_value_fixed(RuntimeModule, c.kTVMFFIModule);

    /// Load a compiled module (.so) from disk.
    pub fn load_from_file(allocator: std.mem.Allocator, path: [:0]const u8) !RuntimeModule {
        const result = try api.call_global(allocator, "ffi.ModuleLoadFromFile", &.{
            Value.str(path),
        });
        return .{ .handle = .{ .ptr = result.as_object() orelse return error.TvmCallFailed } };
    }

    /// Get a packed function from this module by name.
    pub fn get_function(self: RuntimeModule, allocator: std.mem.Allocator, name: [:0]const u8, query_imports: bool) !Value {
        return api.call_global(allocator, "ffi.ModuleGetFunction", &.{
            self.as_value(), Value.str(name), Value.boolean(query_imports),
        });
    }

    /// Write the module to a file in the given format ("o", "so", "ptx", etc.).
    pub fn write_to_file(self: RuntimeModule, allocator: std.mem.Allocator, path: [:0]const u8, format: [:0]const u8) !void {
        _ = try api.call_global(allocator, "ffi.ModuleWriteToFile", &.{
            self.as_value(),
            Value.str(path),
            Value.str(format),
        });
        log.debug("wrote module to {s} (format={s})", .{ path, format });
    }

    /// Pack device module imports into an LLVM blob (for CUDA .so linking).
    pub fn pack_imports_to_llvm(self: RuntimeModule, allocator: std.mem.Allocator) !RuntimeModule {
        const result = try api.call_global(allocator, "runtime.ModulePackImportsToLLVM", &.{
            self.as_value(),
            Value.boolean(false), // system_lib
            Value.str("llvm"),
            Value.str(""),
        });
        const obj = result.as_object() orelse return error.TvmCallFailed;
        return .{ .handle = .{ .ptr = obj } };
    }
};

// Tensor - DLPack <-> TVM tensor bridge

pub const Tensor = struct {
    handle: ObjectHandle,

    pub const deinit = helpers.deinit(Tensor);

    /// Create a TVM tensor from a DLPack ManagedTensor.
    pub fn from_dlpack(managed: *dlpack.ManagedTensor) TvmError!Tensor {
        var out: c.TVMFFIObjectHandle = null;
        if (c.TVMFFITensorFromDLPack(@ptrCast(managed), 0, 0, &out) != 0 or out == null) {
            return error.TvmCallFailed;
        }
        return .{ .handle = .{ .ptr = out } };
    }

    /// Allocate a TVM tensor on the given device and copy data from host.
    ///
    /// `shape` must remain valid for the lifetime of the returned tensor
    /// (TVM stores the shape pointer internally).
    pub fn allocate(
        allocator: std.mem.Allocator,
        data: []f32,
        shape: []i64,
        device_type: dlpack.DeviceType,
        device_ordinal: i32,
    ) TvmError!Tensor {
        // 1. Create Shape object
        var shape_vals: [4]Value = undefined;
        for (shape, 0..) |dim, i| {
            shape_vals[i] = Value.int(dim);
        }
        const shape_obj = try api.call_global(allocator, "ffi.Shape", shape_vals[0..shape.len]);
        defer shape_obj.decref();

        // 2. Allocate empty tensor on device
        const tensor_val = try api.call_global(allocator, "runtime.TVMTensorAllocWithScope", &.{
            shape_obj,
            dtype_value(.float, 32),
            device_value(device_type, device_ordinal),
            Value.none(),
        });

        // 3. Copy host data to device tensor
        const nbytes = data.len * @sizeOf(f32);
        _ = try api.call_global(allocator, "runtime.TVMTensorCopyFromBytes", &.{
            tensor_val,
            ptr_value(@ptrCast(@constCast(data.ptr))),
            Value.int(@intCast(nbytes)),
        });

        const obj = tensor_val.as_object() orelse return error.TvmCallFailed;
        return .{ .handle = .{ .ptr = obj } };
    }

    /// Copy tensor data back to host memory.
    pub fn copy_to_host(self: Tensor, allocator: std.mem.Allocator, dest: []f32) TvmError!void {
        const nbytes = dest.len * @sizeOf(f32);
        _ = try api.call_global(allocator, "runtime.TVMTensorCopyToBytes", &.{
            self.as_value(),
            ptr_value(@ptrCast(dest.ptr)),
            Value.int(@intCast(nbytes)),
        });
    }

    pub const as_value = helpers.as_value_fixed(Tensor, c.kTVMFFITensor);
};

/// Wait for work on the device's current TVM stream to finish.
pub fn synchronize(
    allocator: std.mem.Allocator,
    device_type: dlpack.DeviceType,
    device_ordinal: i32,
) TvmError!void {
    var result = try api.call_global(allocator, "runtime.Device_StreamSync", &.{
        device_value(device_type, device_ordinal),
        Value.int(0),
    });
    defer result.decref();
}

// Private helpers for constructing special Value types needed by Tensor methods.

fn device_value(device_type: dlpack.DeviceType, device_id: i32) Value {
    var v = std.mem.zeroes(c.TVMFFIAny);
    v.type_index = c.kTVMFFIDevice;
    v.unnamed_1.v_device = .{ .device_type = @intCast(@intFromEnum(device_type)), .device_id = device_id };
    return .{ .raw = v };
}

fn dtype_value(code: dlpack.DataTypeCode, bits: u8) Value {
    var v = std.mem.zeroes(c.TVMFFIAny);
    v.type_index = c.kTVMFFIDataType;
    v.unnamed_1.v_dtype = .{ .code = @intCast(@intFromEnum(code)), .bits = bits, .lanes = 1 };
    return .{ .raw = v };
}

fn ptr_value(ptr: *anyopaque) Value {
    var v = std.mem.zeroes(c.TVMFFIAny);
    v.type_index = c.kTVMFFIOpaquePtr;
    v.unnamed_1.v_int64 = @bitCast(@intFromPtr(ptr));
    return .{ .raw = v };
}
