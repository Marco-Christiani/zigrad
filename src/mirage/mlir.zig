//! MLIR carrier types used by the optional Mirage prototype.
//!
//! These types are internal to the Mirage integration and do not participate
//!  in the kernel-provider contract.
const std = @import("std");
const pr = @import("../pr/pr.zig");

/// Ranked tensor descriptor extracted from an MLIR operation signature.
pub const MlirTensorDesc = struct {
    dtype: pr.DType,
    dims: []const usize,
};

/// Operation pattern recognized by the Mirage MLIR prototype.
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

/// Descriptor for one Mirage candidate selected through MLIR.
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
