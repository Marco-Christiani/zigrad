/// Zigrad PJRT/XLA Backend Prototype
///
/// This is an orphan branch prototype evaluating PJRT/XLA as a potential backend.
const std = @import("std");

// Core backend abstraction
pub const backend = @import("backend/backend.zig");
pub const pjrt_backend = @import("backend/pjrt.zig");

// PJRT bindings
pub const pjrt = struct {
    pub const api = @import("pjrt/api.zig");
    pub const plugin = @import("pjrt/plugin.zig");
    pub const types = @import("pjrt/types.zig");
    pub const c = @import("pjrt/c.zig");
};

// Runtime utilities
pub const runtime = struct {
    pub const buffer = @import("runtime/buffer.zig");
    pub const DType = buffer.DType;
    pub const Shape = buffer.Shape;
    pub const HostBuffer = buffer.HostBuffer;
};

pub const diagnostics = @import("diagnostics.zig");

// MLIR/StableHLO
pub const mlir = struct {
    pub const program = @import("mlir/program.zig");
    pub const Program = program.Program;
};

// Re-export commonly used types
pub const Backend = backend.Backend;
pub const Device = backend.Device;
pub const Executable = backend.Executable;
pub const Buffer = backend.Buffer;
pub const Event = backend.Event;
pub const CompileOptions = backend.CompileOptions;

pub const DType = runtime.DType;
pub const Shape = runtime.Shape;
pub const HostBuffer = runtime.HostBuffer;

pub const Program = mlir.Program;

test "all tests" {
    std.testing.refAllDecls(@This());
}
