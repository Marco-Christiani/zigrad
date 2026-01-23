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
/// See: .internal/2026-01-16-03_PASS_BASED_PIPELINE.md
const std = @import("std");

// Core modules
pub const pr = @import("pr/root.zig");
pub const frontend = @import("frontend/frontend.zig");
pub const pipeline = @import("pipeline/root.zig");
pub const backend = @import("backend/root.zig");
pub const lower = @import("lower/root.zig");

// Utility types
pub const utils = struct {
    pub const HostBuffer = @import("utils/host_buffer.zig").HostBuffer;
    pub const DType = @import("utils/host_buffer.zig").DType;
    pub const Shape = @import("utils/host_buffer.zig").Shape;
};

test {
    std.testing.refAllDecls(@This());
}
