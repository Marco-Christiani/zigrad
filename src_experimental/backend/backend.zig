/// Backend Abstraction Interface
///
/// Generic backend interface using VTables.
/// PJRT is ONE implementation, not THE implementation.
///
/// Design principle: No PJRT types leak into this interface.

const std = @import("std");
const runtime = @import("../runtime/buffer.zig");

pub const DType = runtime.DType;
pub const Shape = runtime.Shape;


pub const Error = error{
    BackendInitFailed,
    DeviceNotFound,
    CompileFailed,
    ExecuteFailed,
    BufferTransferFailed,
    OutOfMemory,
    InvalidArgument,
    Unimplemented,
    Internal,
    PluginLoadFailed,
};


pub const Backend = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        deinit: *const fn (*anyopaque) void,
        getDevices: *const fn (*anyopaque, std.mem.Allocator) Error![]Device,
        compile: *const fn (*anyopaque, *const Device, CompileOptions) Error!Executable,
        bufferFromHost: *const fn (*anyopaque, *const Device, []const u8, DType, Shape) Error!Buffer,
    };

    pub fn deinit(self: Backend) void {
        self.vtable.deinit(self.ptr);
    }

    pub fn getDevices(self: Backend, allocator: std.mem.Allocator) Error![]Device {
        return self.vtable.getDevices(self.ptr, allocator);
    }

    pub fn compile(self: Backend, device: *const Device, options: CompileOptions) Error!Executable {
        return self.vtable.compile(self.ptr, device, options);
    }

    pub fn bufferFromHost(self: Backend, device: *const Device, data: []const u8, dtype: DType, shape: Shape) Error!Buffer {
        return self.vtable.bufferFromHost(self.ptr, device, data, dtype, shape);
    }
};


pub const Device = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        deinit: *const fn (*anyopaque) void,
        getId: *const fn (*anyopaque) Error!u32,
        getName: *const fn (*anyopaque, std.mem.Allocator) Error![]const u8,
        getKind: *const fn (*anyopaque) DeviceKind,
    };

    pub const DeviceKind = enum {
        cpu,
        cuda,
        rocm,
        tpu,
        custom,
    };

    pub fn deinit(self: Device) void {
        self.vtable.deinit(self.ptr);
    }

    pub fn getId(self: *const Device) Error!u32 {
        return self.vtable.getId(self.ptr);
    }

    pub fn getName(self: *const Device, allocator: std.mem.Allocator) Error![]const u8 {
        return self.vtable.getName(self.ptr, allocator);
    }

    pub fn getKind(self: *const Device) DeviceKind {
        return self.vtable.getKind(self.ptr);
    }
};


pub const Executable = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        deinit: *const fn (*anyopaque) void,
        execute: *const fn (*anyopaque, []const Buffer, std.mem.Allocator) Error!ExecuteResult,
        serialize: ?*const fn (*anyopaque, std.mem.Allocator) Error![]const u8,
    };

    pub fn deinit(self: Executable) void {
        self.vtable.deinit(self.ptr);
    }

    pub fn execute(self: Executable, inputs: []const Buffer, allocator: std.mem.Allocator) Error!ExecuteResult {
        return self.vtable.execute(self.ptr, inputs, allocator);
    }

    pub fn serialize(self: Executable, allocator: std.mem.Allocator) Error![]const u8 {
        if (self.vtable.serialize) |serialize_fn| {
            return serialize_fn(self.ptr, allocator);
        }
        return error.Unimplemented;
    }
};

pub const ExecuteResult = struct {
    outputs: []Buffer,
    event: ?Event,

    pub fn deinit(self: *ExecuteResult, allocator: std.mem.Allocator) void {
        for (self.outputs) |*buf| {
            buf.deinit();
        }
        allocator.free(self.outputs);
        if (self.event) |*evt| {
            evt.deinit();
        }
    }
};


pub const Buffer = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        deinit: *const fn (*anyopaque) void,
        getShape: *const fn (*anyopaque) Shape,
        getDtype: *const fn (*anyopaque) DType,
        toHost: *const fn (*anyopaque, []u8) Error!Event,
    };

    pub fn deinit(self: Buffer) void {
        self.vtable.deinit(self.ptr);
    }

    pub fn getShape(self: Buffer) Shape {
        return self.vtable.getShape(self.ptr);
    }

    pub fn getDtype(self: Buffer) DType {
        return self.vtable.getDtype(self.ptr);
    }

    pub fn toHost(self: Buffer, dst: []u8) Error!Event {
        return self.vtable.toHost(self.ptr, dst);
    }
};


pub const Event = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        deinit: *const fn (*anyopaque) void,
        await_: *const fn (*anyopaque) Error!void,
        isReady: *const fn (*anyopaque) Error!bool,
    };

    pub fn deinit(self: Event) void {
        self.vtable.deinit(self.ptr);
    }

    pub fn await_(self: Event) Error!void {
        return self.vtable.await_(self.ptr);
    }

    pub fn isReady(self: Event) Error!bool {
        return self.vtable.isReady(self.ptr);
    }
};


pub const CompileOptions = struct {
    /// IR format
    format: Format,

    /// Program bytecode (MLIR text, bytecode, or StableHLO)
    bytecode: []const u8,

    /// Optimization level (0-3)
    optimization_level: u8 = 3,

    /// Directory for debug dumps (null = no dumps)
    dump_dir: ?[]const u8 = null,

    /// Backend-specific options (opaque string, e.g., serialized proto or JSON)
    backend_options: ?[]const u8 = null,

    pub const Format = enum {
        stablehlo_portable,
        stablehlo_mlir_text,
        stablehlo_mlir_bytecode,
        mlir_text,
        mlir_bytecode,
    };
};
