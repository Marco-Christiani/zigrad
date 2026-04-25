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
//!
//! All handle types are opaque wrappers around `*anyopaque`, erasing the
//!  concrete C pointer types at the interface boundary. Details about this
//!  can be nuanced as they depend on compiler internals, specifically
//!  zig translate-c. We try to document this when it comes up.
//!
//! Provides concrete implementations only for ubiquitous patterns,
//!  currently only `.transfer()`.
const std = @import("std");
const pr = @import("pr/pr.zig");
const kernel = @import("kernel.zig");
const utils = @import("utils.zig");
const host_buffer = utils.host_buffer;
const Tree = utils.Tree;

const Backend = @This();

vtable: *const VTable,

pub const VTable = struct {
    // Compilation
    // TODO: this signature is leaking across abs boundary
    compile: *const fn (b: *Backend, device: Device, mlir: []const u8, is_bytecode: bool, opts: CompileOptions) Error!Executable,

    // Buffer management
    buffer_from_host: *const fn (b: *Backend, device: Device, data: []const u8, dtype: pr.DType, shape: []const i64) Error!Buffer,
    buffer_to_host: *const fn (b: *Backend, buf: Buffer, dst: []u8) Error!?Event,

    // Execution
    execute: *const fn (b: *Backend, exe: Executable, allocator: std.mem.Allocator, inputs: []const Buffer, opts: ExecuteOptions) Error!ExecuteResult,
    execute_into: *const fn (b: *Backend, exe: Executable, inputs: []const Buffer, outputs: []Buffer, non_donatable: ?[]const i64, opts: ExecuteOptions) Error!?Event,

    // Synchronization
    await_event: *const fn (b: *Backend, ev: Event) Error!void,

    // Serialization
    serialize_executable: *const fn (b: *Backend, exe: Executable, allocator: std.mem.Allocator) Error![]u8,
    load_serialized: *const fn (b: *Backend, data: []const u8) Error!Executable,

    /// Retrieve the backend's compiled program.
    ///
    /// Representation and semantics are backend-specific and may not
    ///  be supported by all backends.
    ///
    /// E.g., IREE's current interface, or a PJRT plugin that does not
    ///  implement the `PJRT_Executable_OptimizedProgram` API. The
    ///  returned `code` is backend-defined (XLA: serialized
    ///  `HloModuleProtoWithConfig`).
    ///
    /// Returns `null` if the backend does not expose this.
    /// Caller owns memory.
    get_optimized_program: *const fn (b: *Backend, exe: Executable, allocator: std.mem.Allocator) Error!?OptimizedProgram,

    // Lifecycle
    deinit_buffer: *const fn (b: *Backend, buf: Buffer) void,
    deinit_event: *const fn (b: *Backend, ev: Event) void,
    deinit_executable: *const fn (b: *Backend, exe: Executable) void,

    // Device enumeration
    get_devices: *const fn (b: *Backend, allocator: std.mem.Allocator) Error![]Device,
};

/// Post-compilation program bytes returned by `get_optimized_program`.
pub const OptimizedProgram = struct {
    /// Backend-specific: for XLA/PJRT it is a serialized `HloModuleProtoWithConfig`,
    ///  decodable via `xla.HloModuleProto`.
    code: []u8,
    ///  A short identifier like `"hlo"`, `"mlir_bytecode"`, etc.
    format: []const u8,

    pub fn deinit(self: *OptimizedProgram, allocator: std.mem.Allocator) void {
        allocator.free(self.code);
        allocator.free(self.format);
        self.* = undefined;
    }
};

/// Opaque handle to a buffer as defined by the backend implementation
pub const Buffer = struct { handle: *anyopaque };

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

/// Copy buffer contents to host memory.
///
/// Returns null when the buffer is already host-resident and the transfer
///  completed synchronously (no event to await or deinit). Callers must
///  handle the null case -- identical to the `execute_into` event contract.
pub fn buffer_to_host(self: *Backend, buf: Buffer, dst: []u8) Error!?Event {
    return self.vtable.buffer_to_host(self, buf, dst);
}

pub fn execute(self: *Backend, exe: Executable, allocator: std.mem.Allocator, inputs: []const Buffer, opts: ExecuteOptions) Error!ExecuteResult {
    return self.vtable.execute(self, exe, allocator, inputs, opts);
}

/// Execute into caller-provided output slots.
///
/// All output slots are populated on success. The output slice must have
///  exactly the right arity.
///
/// Slot contents before the call are undefined (the backend overwrites
///  them unconditionally).
/// NOTE: the above statement has questionable phrasing we should improve
///  this docstring as things stabilize
pub fn execute_into(self: *Backend, exe: Executable, inputs: []const Buffer, outputs: []Buffer, non_donatable: ?[]const i64, opts: ExecuteOptions) Error!?Event {
    return self.vtable.execute_into(self, exe, inputs, outputs, non_donatable, opts);
}

pub fn await_event(self: *Backend, ev: Event) Error!void {
    return self.vtable.await_event(self, ev);
}

pub fn serialize_executable(self: *Backend, exe: Executable, allocator: std.mem.Allocator) Error![]u8 {
    return self.vtable.serialize_executable(self, exe, allocator);
}

pub fn load_serialized(self: *Backend, data: []const u8) Error!Executable {
    return self.vtable.load_serialized(self, data);
}

/// Retrieve the backend-optimized program for an executable (if supported).
///
/// Returns `null` when the backend does not support introspection.
/// The caller owns the returned buffer.
/// See also: `VTable.get_optimized_program` and `OptimizedProgram`.
pub fn get_optimized_program(self: *Backend, exe: Executable, allocator: std.mem.Allocator) Error!?OptimizedProgram {
    return self.vtable.get_optimized_program(self, exe, allocator);
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

/// Direction to transfer data (e.g., H2D, D2H)
pub const TransferDirection = enum {
    /// Transfer host data to device memory. Returns owned `Buffer`(s).
    to_device,
    /// Transfer device data to host memory. TODO: Not yet implemented.
    to_host,
};

/// Transfer data between host and device.
///
/// For `.to_device`, accepts a single `HostBuffer` (returns `Buffer`) or
///  `*[const] Tree(HostBuffer)` (returns `Tree(Buffer)`). Host data is
///  copied -- caller retains ownership of the source. The returned device
///  buffer(s) are owned by the caller.
///
/// For the `Tree` overload, the returned tree's internal allocations use
///  the source tree's allocator. Caller must `deinit` the returned tree.
///
/// NOTE: `.to_host` is not yet implemented -- produces a compile error.
/// TODO: with new tensor methods this now seems dead, were we going somewhere with this? I think the tree support
///  was the selling point so we should probably keep it and decide about relocation or just use it
///  as a reference for when we implemet tree transfer support (to avoid the verbose map pattern).
pub fn transfer(
    self: *Backend,
    device: Device,
    source: anytype,
    comptime direction: TransferDirection,
) TransferError(@TypeOf(source), direction) {
    const S = @TypeOf(source);
    switch (direction) {
        .to_device => {
            if (S == host_buffer.HostBuffer) {
                // single buffer transfer
                return self.buffer_from_host(device, source.data(), source.dtype, source.shape.const_slice());
            } else if (S == *const Tree(host_buffer.HostBuffer) or S == *Tree(host_buffer.HostBuffer)) {
                // tree of buffers
                const Ctx = struct { b: *Backend, d: Device };
                return source.map(Buffer, Ctx{ .b = self, .d = device }, struct {
                    fn f(ctx: Ctx, buf: host_buffer.HostBuffer) !Buffer {
                        return try ctx.b.buffer_from_host(ctx.d, buf.data(), buf.dtype, buf.shape.const_slice());
                    }
                }.f);
            } else {
                @compileError("transfer: source must be HostBuffer or *[const] Tree(HostBuffer), got " ++ @typeName(S));
            }
        },
        .to_host => @compileError("transfer(.to_host) not yet implemented"),
    }
}

fn TransferError(comptime S: type, comptime direction: TransferDirection) type {
    const HostBuffer = host_buffer.HostBuffer;
    const TreeHB = Tree(HostBuffer);
    return switch (direction) {
        .to_device => if (S == HostBuffer) Error!Buffer else if (S == *const TreeHB or S == *TreeHB) anyerror!Tree(Buffer) else @compileError("unsupported source type for to_device"),
        .to_host => @compileError("to_host not yet implemented"),
    };
}
