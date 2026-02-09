//! Common types and enums for TVM integration.
//!
//! This module defines shared types used across the TVM subsystem,
//! Options are centralized here.

const std = @import("std");
const c = @import("../ffi/tvm.zig");

/// Target kind for TVM compilation.
pub const TargetKind = enum {
    cpu,
    cuda,
};

/// Options for TVM MetaSchedule autotuning.
pub const TuneOpts = struct {
    /// Directory to store tuning database and artifacts.
    work_dir: []const u8 = "artifacts/tvm_cache",

    /// Maximum number of tuning trials.
    max_trials: u32 = 64,

    /// Number of trials per iteration (batch size for parallel builds).
    trials_per_iter: u32 = 16,
};

/// Matrix multiplication shape for tuning.
pub const MatmulShape = struct {
    M: usize,
    N: usize,
    K: usize,
};

/// Context for tuning callbacks.
///
/// Stored in a global to bridge the C callback interface.
/// Contains allocator, target, and shape information needed by builder/runner.
pub const TuneContext = struct {
    allocator: std.mem.Allocator,
    target: c.TVMFFIAny,
    target_kind: TargetKind,
    work_dir: []const u8,
    shape: MatmulShape,
    build_counter: u32 = 0,

    pub fn init(
        allocator: std.mem.Allocator,
        target: c.TVMFFIAny,
        target_kind: TargetKind,
        work_dir: []const u8,
        shape: MatmulShape,
    ) TuneContext {
        return .{
            .allocator = allocator,
            .target = target,
            .target_kind = target_kind,
            .work_dir = work_dir,
            .shape = shape,
            .build_counter = 0,
        };
    }

    pub fn deinit(self: *TuneContext) void { // NOTE: is this used anywhere?
        _ = self;
        // .so files are left in work_dir for potential reuse
    }
};
