//! Type-erased backend interface.
//!
//! Minimal contract for zigrad core code: compile, execute, transfer, sync,
//!  lifecycle, and device enumeration. Buffer introspection, unsafe access,
//!  and backend-specific diagnostics live on the concrete backend -- callers
//!  that need those capabilities hold a direct reference to the concrete type.
//!
//! Follows the `std.Io` interface pattern. Concrete backends embed a
//!  `Backend` field, wire up a static `VTable`, and should expose an
//!  `interface` field. Similar relationship between `std.Io.Reader` and
//!  implementations like `std.Io.File.Reader`.
const std = @import("std");
const pr = @import("../pr/pr.zig");
const kernel = @import("../kernel.zig");

const Backend = @This();

vtable: *const VTable,

pub const VTable = struct {
    // Compilation
    compile: *const fn (b: *Backend, device: Device, mlir: []const u8, is_bytecode: bool, opts: CompileOptions) Error!Executable,

    // Buffer management
    buffer_from_host: *const fn (b: *Backend, device: Device, data: []const u8, dtype: pr.DType, shape: []const i64) Error!Buffer,
    buffer_to_host: *const fn (b: *Backend, buf: Buffer, dst: []u8) Error!Event,

    // Execution
    execute: *const fn (b: *Backend, exe: Executable, allocator: std.mem.Allocator, inputs: []const Buffer, opts: ExecuteOptions) Error!ExecuteResult,
    execute_into: *const fn (b: *Backend, exe: Executable, inputs: []const RawBuffer, outputs: []?RawBuffer, non_donatable: ?[]const i64, opts: ExecuteOptions) Error!?Event,

    // Synchronization
    await_event: *const fn (b: *Backend, ev: Event) Error!void,

    // Lifecycle
    deinit_buffer: *const fn (b: *Backend, buf: Buffer) void,
    deinit_event: *const fn (b: *Backend, ev: Event) void,
    deinit_executable: *const fn (b: *Backend, exe: Executable) void,

    // Device enumeration
    get_devices: *const fn (b: *Backend, allocator: std.mem.Allocator) Error![]Device,
};

/// Opaque handle to a buffer as defined by the backend implementation
pub const Buffer = struct { handle: *anyopaque };
pub const RawBuffer = *anyopaque;

/// Opaque handle to a device as defined by the backend implementation
pub const Device = struct { handle: *anyopaque };

/// Opaque handle to an executable as defined by the backend implementation
pub const Executable = struct { handle: *anyopaque };

/// Opaque handle to an event as defined by the backend implementation
pub const Event = struct { handle: *anyopaque };

pub const ExecuteResult = struct {
    outputs: []Buffer,
    event: ?Event,
};

// TODO: looks like boundary leakage violation, need to revisit CompileOptions.
pub const CompileOptions = struct {
    num_replicas: u32 = 1,
    num_partitions: u32 = 1,
};

pub const ExecuteOptions = struct {
    store: ?*const kernel.KernelStore = null,
    dispatch_registry: ?*const kernel.DispatchRegistry = null,
};

pub const Error = error{
    PjrtError,
    OutOfMemory,
    CompilationFailed,
    ExecutionFailed,
    BufferTransferFailed,
    DeviceError,
    TypedFfiUnavailable,
    TypedFfiRegistrationFailed,
    BackendError,
    Unexpected,
};

// TODO: this signature is leaking across abs boundary
pub fn compile(self: *Backend, device: Device, mlir: []const u8, is_bytecode: bool, opts: CompileOptions) Error!Executable {
    return self.vtable.compile(self, device, mlir, is_bytecode, opts);
}

pub fn buffer_from_host(self: *Backend, device: Device, data: []const u8, dtype: pr.DType, shape: []const i64) Error!Buffer {
    return self.vtable.buffer_from_host(self, device, data, dtype, shape);
}

pub fn buffer_to_host(self: *Backend, buf: Buffer, dst: []u8) Error!Event {
    return self.vtable.buffer_to_host(self, buf, dst);
}

pub fn execute(self: *Backend, exe: Executable, allocator: std.mem.Allocator, inputs: []const Buffer, opts: ExecuteOptions) Error!ExecuteResult {
    return self.vtable.execute(self, exe, allocator, inputs, opts);
}

pub fn execute_into(self: *Backend, exe: Executable, inputs: []const RawBuffer, outputs: []?RawBuffer, non_donatable: ?[]const i64, opts: ExecuteOptions) Error!?Event {
    return self.vtable.execute_into(self, exe, inputs, outputs, non_donatable, opts);
}

pub fn await_event(self: *Backend, ev: Event) Error!void {
    return self.vtable.await_event(self, ev);
}

pub fn deinit_buffer(self: *Backend, buf: Buffer) void {
    return self.vtable.deinit_buffer(self, buf);
}

pub fn deinit_event(self: *Backend, ev: Event) void {
    return self.vtable.deinit_event(self, ev);
}

pub fn deinit_executable(self: *Backend, exe: Executable) void {
    return self.vtable.deinit_executable(self, exe);
}

pub fn get_devices(self: *Backend, allocator: std.mem.Allocator) Error![]Device {
    return self.vtable.get_devices(self, allocator);
}
