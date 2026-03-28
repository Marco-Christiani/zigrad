//! Mirage MLIR Kernel Types
//!
//! Provider-specific MLIR descriptor types for Mirage's MLIR-level
//! kernelization path. These types are internal to the Mirage integration
//! and not part of the core kernel provider interface.
const std = @import("std");
const pr = @import("../pr/pr.zig");

/// Ranked tensor descriptor extracted from an MLIR operation signature.
pub const MlirTensorDesc = struct {
    dtype: pr.DType,
    dims: []const usize,
};

/// Stable operation-pattern identity selected by MLIR kernel passes.
pub const MlirKernelPattern = enum {
    dot,
    dot_general,
    dot_add,
    dot_add_mul,
    dot_log,
    dot_exp,
    rms_norm,
    rms_norm_matmul,
    softmax_matmul,
    attention,
};

/// Provider-neutral descriptor for one selected MLIR kernel call.
///
/// This descriptor is intentionally small and stable: it captures only the
/// information needed to compile known selected carrier patterns without
/// depending on PR region descriptors.
pub const MlirKernelDescriptor = struct {
    name: []const u8,
    provider_name: []const u8,
    pattern: MlirKernelPattern,
    inputs: []const MlirTensorDesc,
    outputs: []const MlirTensorDesc,
    /// rms_norm: size of the last dimension (normalized axis).
    normalized_size: i32 = 0,
    /// softmax_matmul, attention: dimension index for the reduce-sum.
    reduction_dim: i32 = 0,
    /// softmax_matmul, attention: size of the reduction dimension.
    reduction_factor: i32 = 0,
    /// attention: scale factor applied to raw scores (1/sqrt(d)).
    scale: f32 = 0.0,
};
