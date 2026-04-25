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
//!   - `frontend.trace` / `zg.trace`: Trace a comptime function against abstract specs -> `pr.Program`.
//!   - `frontend.compile_program`: Run pipeline on a traced program -> backend executable.
//!   - `frontend.transforms` / `zg.value_and_grad`: Trace-time function transforms.
//!   - `frontend.optim`: Traced-mode optimizer building blocks.
//!   - `frontend.train`: Training loop state management (`TrainState`).
//! - `Tensor`: Unified tensor type (traced, device, host, or abstract backing).
//! - `utils`: Utility types (`HostBuffer`, `Tree`, etc.).
const std = @import("std");

pub const build_options = @import("build_options");

// Core modules
pub const pr = @import("pr.zig");
pub const frontend = @import("frontend/frontend.zig");
pub const pipeline = @import("pipeline.zig");
pub const lower = @import("lower.zig");
pub const kernel = @import("kernel.zig");
pub const tune = @import("tune.zig");
pub const Cache = @import("cache.zig").Cache;
pub const utils = @import("utils.zig");

// TVM subsystem (gated by SDK header availability)
pub const tvm = if (build_options.has_tvm) @import("tvm.zig") else struct {};

// Mirage subsystem (gated by SDK header availability)
pub const mirage = if (build_options.has_mirage) @import("mirage.zig") else struct {};

// Tier 1: commonly used types at top level
pub const Tensor = @import("tensor.zig");
pub const HostBuffer = utils.HostBuffer;
pub const DType = pr.DType;
pub const Shape = pr.Shape;
pub const BoundedShape = pr.BoundedShape;

// Backends
pub const Backend = @import("Backend.zig");
pub const pjrt = @import("backend/pjrt.zig");
/// IREE backend module.  Only compiled when `-Diree-backend=true`.
pub const iree = if (build_options.has_iree) @import("backend/iree.zig") else struct {};

// Tier 1: commonly used functions at top level
pub const jit = frontend.jit;
pub const trace = frontend.trace;
pub const grad = frontend.transforms.make_grad;
pub const value_and_grad = frontend.transforms.make_value_and_grad;

pub const from_safetensors = utils.safetensors.from_safetensors;
pub const FromSafetensorsOpts = utils.safetensors.Opts;

test {
    @setEvalBranchQuota(10000);
    std.testing.refAllDecls(@This());
}
