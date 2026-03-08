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

pub const build_options = @import("build_options");

// Core modules
pub const pr = @import("pr/root.zig");
pub const frontend = @import("frontend/frontend.zig");
pub const pipeline = @import("pipeline/root.zig");
pub const backend = @import("backend/root.zig");
pub const lower = @import("lower/root.zig");
pub const kernel = @import("kernel.zig");
pub const utils = @import("utils/root.zig");
pub const kernels = @import("kernels/root.zig");
pub const benchmark = @import("benchmark/root.zig");

// TVM subsystem (gated by SDK header availability)
pub const tvm = if (build_options.has_tvm) @import("tvm/root.zig") else struct {};

// Mirage subsystem (gated by SDK header availability)
pub const mirage = if (build_options.has_mirage) @import("mirage/root.zig") else struct {};

// Tier 1: commonly used types at top level
pub const Backend = backend.PjrtBackend;
pub const HostBuffer = utils.HostBuffer;
pub const DType = utils.DType;
pub const Shape = utils.Shape;

test {
    std.testing.refAllDecls(@This());
}
