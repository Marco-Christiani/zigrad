//! Zigrad Root Module
//!
//! Public API for the Zigrad compiler.
//!
//! Module organization:
//! - `pr`: Program Representation (Zigrad's in memory IR).
//!    Includes `pr.ad` for core AD transforms (VJP, JVP).
//! - `lower`: Lowering passes (PR -> MLIR).
//! - `backend`: Unified backend interface (compile + execute).
//! - `pipeline`: Infrastructure for compiler passes.
//! - `frontend`: Higher level user-facing APIs.
//!   - `frontend.compile`: AOT compilation of traced functions.
//!   - `frontend.transforms`: Trace-time function transforms (e.g. `value_and_grad`).
//!   - `frontend.optim`: Traced-mode optimizer building blocks.
//!   - `frontend.train`: Training loop state management (`TrainState`).
//! - `Tensor`: Unified tensor type (traced, device, host, or abstract backing).
//! - `utils`: Utility types (`HostBuffer`, `Tree`, etc.).
const std = @import("std");

pub const build_options = @import("build_options");

// Core modules
pub const pr = @import("pr/root.zig");
pub const frontend = @import("frontend/frontend.zig");
pub const pipeline = @import("pipeline/root.zig");
pub const backend = @import("backend/root.zig");
pub const lower = @import("lower.zig");
pub const kernel = @import("kernel.zig");
pub const tune = @import("tune.zig");
pub const Cache = @import("cache.zig").Cache;
pub const utils = @import("utils/root.zig");

// TVM subsystem (gated by SDK header availability)
pub const tvm = if (build_options.has_tvm) @import("tvm/root.zig") else struct {};

// Mirage subsystem (gated by SDK header availability)
pub const mirage = if (build_options.has_mirage) @import("mirage/root.zig") else struct {};

// Tier 1: commonly used types at top level
pub const Backend = backend.Backend;
pub const Tensor = @import("tensor.zig");
pub const HostBuffer = utils.HostBuffer;
pub const DType = pr.DType;
pub const Shape = pr.Shape;
pub const BoundedShape = pr.BoundedShape;

test {
    std.testing.refAllDecls(@This());
}
