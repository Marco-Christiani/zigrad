//! DLPack v1.2 type definitions — standalone, no TVM dependency.
//!
//! Pure Zig `extern struct` definitions ABI-compatible with the DLPack
//! interchange format used by TVM, PyTorch, JAX, IREE, etc.

const std = @import("std");

pub const DeviceType = enum(i32) {
    cpu = 1,
    cuda = 2,
    cuda_host = 3,
    opencl = 4,
    vulkan = 7,
    metal = 8,
    rocm = 10,
    rocm_host = 11,
    _,
};

pub const DataTypeCode = enum(u8) {
    int = 0,
    uint = 1,
    float = 2,
    opaque_handle = 3,
    bfloat = 4,
    complex = 5,
    bool_ = 6,
    _,
};

pub const Device = extern struct {
    device_type: DeviceType,
    device_id: i32,
};

pub const DataType = extern struct {
    code: DataTypeCode,
    bits: u8,
    lanes: u16,

    pub const f32_ = DataType{ .code = .float, .bits = 32, .lanes = 1 };
    pub const f16_ = DataType{ .code = .float, .bits = 16, .lanes = 1 };
    pub const i64_ = DataType{ .code = .int, .bits = 64, .lanes = 1 };
    pub const i32_ = DataType{ .code = .int, .bits = 32, .lanes = 1 };
};

pub const Tensor = extern struct {
    data: ?*anyopaque,
    device: Device,
    ndim: i32,
    dtype: DataType,
    shape: [*]i64,
    strides: ?[*]i64,
    byte_offset: u64,

    /// Create a contiguous DLPack tensor borrowing existing host memory.
    ///
    /// The returned tensor references `data` and `shape` by pointer —
    /// both must outlive the tensor.
    pub fn init_contiguous(comptime T: type, data: []T, shape: []i64) Tensor {
        return .{
            .data = @ptrCast(data.ptr),
            .device = .{ .device_type = .cpu, .device_id = 0 },
            .ndim = @intCast(shape.len),
            .dtype = comptime dtype_of(T),
            .shape = shape.ptr,
            .strides = null,
            .byte_offset = 0,
        };
    }
};

pub const ManagedTensor = extern struct {
    dl_tensor: Tensor,
    manager_ctx: ?*anyopaque,
    deleter: ?*const fn (?*ManagedTensor) callconv(.c) void,

    /// Wrap a Tensor as a ManagedTensor with a no-op deleter (borrowed memory).
    pub fn borrowing(dl_tensor: Tensor) ManagedTensor {
        return .{
            .dl_tensor = dl_tensor,
            .manager_ctx = null,
            .deleter = &noop_deleter,
        };
    }
};

pub fn noop_deleter(_: ?*ManagedTensor) callconv(.c) void {}

fn dtype_of(comptime T: type) DataType {
    return switch (T) {
        f32 => DataType.f32_,
        f16 => DataType.f16_,
        i64 => DataType.i64_,
        i32 => DataType.i32_,
        else => @compileError("unsupported DLPack element type"),
    };
}

comptime {
    std.debug.assert(@sizeOf(Device) == 8);
    std.debug.assert(@sizeOf(DataType) == 4);
    // Tensor: ptr(8) + Device(8) + ndim(4) + DataType(4) + shape(8) + strides(8) + byte_offset(8)
    std.debug.assert(@sizeOf(Tensor) == 48);
    // ManagedTensor: Tensor(48) + manager_ctx(8) + deleter(8)
    std.debug.assert(@sizeOf(ManagedTensor) == 64);
}
