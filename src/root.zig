/// Zigrad Root Module
///
/// Public API for the Zigrad compiler.
///
/// Module organization:
/// - pr: Program Representation (Zigrad-owned, toolchain-neutral)
/// - lower: Lowering passes (PR -> MLIR)
/// - backend: Unified backend (compile + execute)
/// - pipeline: Pass-based pipeline infrastructure
/// - frontend: User-facing program construction
/// - utils: Utility types (HostBuffer, etc.)
///
/// See KB: "Pass-Based Pipeline Direction (Design Update)"
const std = @import("std");

// Core modules
pub const pr = @import("pr/root.zig");
pub const frontend = @import("frontend/frontend.zig");
pub const pipeline = @import("pipeline/root.zig");
pub const backend = @import("backend/root.zig");
pub const lower = @import("lower/root.zig");
pub const kernel = @import("kernel.zig");

// Runtime
pub const tvm_runtime = @import("tvm_runtime.zig");

// Kernels
pub const kernels = struct {
    pub const gemm = @import("kernels/gemm.zig");
};

// Benchmark infrastructure
pub const benchmark = struct {
    pub const Harness = @import("benchmark/harness.zig").Harness;
    pub const BenchmarkConfig = @import("benchmark/config.zig").BenchmarkConfig;
    pub const Shape = @import("benchmark/config.zig").Shape;
    pub const Implementation = @import("benchmark/config.zig").Implementation;
};

// Utility types
pub const utils = struct {
    pub const HostBuffer = @import("utils/host_buffer.zig").HostBuffer;
    pub const DType = @import("utils/host_buffer.zig").DType;
    pub const Shape = @import("utils/host_buffer.zig").Shape;
};

test {
    std.testing.refAllDecls(@This());
}
