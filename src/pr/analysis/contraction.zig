//! Structural recognition for PR contraction operations.

const std = @import("std");
const pr = @import("../pr.zig");

/// Return whether a dot-general parameter set describes a rank-two matrix multiply.
pub fn is_matrix_matmul(dg: pr.DotGeneralParams) bool {
    return is_canonical_batched_matmul(dg, 2, 2);
}

/// Return whether dot-general describes a canonical batched matrix multiply.
///
/// Canonical form places batch dimensions first, the lhs contracting dimension
///  last, and the rhs contracting dimension immediately after its batch dimensions.
pub fn is_canonical_batched_matmul(
    dg: pr.DotGeneralParams,
    lhs_rank: usize,
    rhs_rank: usize,
) bool {
    const batch_len = dg.lhs_batch_dims.len;

    if (batch_len != dg.rhs_batch_dims.len) return false;
    if (dg.lhs_contracting_dims.len != 1 or dg.rhs_contracting_dims.len != 1) return false;
    if (lhs_rank != rhs_rank) return false;
    if (lhs_rank != batch_len + 2) return false;
    if (!dims_are_prefix(dg.lhs_batch_dims) or !dims_are_prefix(dg.rhs_batch_dims)) return false;

    const lhs_contract_expected: i64 = @intCast(lhs_rank - 1);
    const rhs_contract_expected: i64 = @intCast(rhs_rank - 2);
    return dg.lhs_contracting_dims[0] == lhs_contract_expected and
        dg.rhs_contracting_dims[0] == rhs_contract_expected;
}

fn dims_are_prefix(dims: []const i64) bool {
    for (dims, 0..) |dim, idx| {
        if (dim != @as(i64, @intCast(idx))) return false;
    }
    return true;
}

test is_matrix_matmul {
    const dg: pr.DotGeneralParams = .{
        .lhs_batch_dims = &.{},
        .rhs_batch_dims = &.{},
        .lhs_contracting_dims = &.{1},
        .rhs_contracting_dims = &.{0},
    };
    try std.testing.expect(is_matrix_matmul(dg));
}

test "is_matrix_matmul rejects non-canonical contractions" {
    const with_batch: pr.DotGeneralParams = .{
        .lhs_batch_dims = &.{0},
        .rhs_batch_dims = &.{0},
        .lhs_contracting_dims = &.{1},
        .rhs_contracting_dims = &.{0},
    };
    try std.testing.expect(!is_matrix_matmul(with_batch));

    const wrong_contract: pr.DotGeneralParams = .{
        .lhs_batch_dims = &.{},
        .rhs_batch_dims = &.{},
        .lhs_contracting_dims = &.{0},
        .rhs_contracting_dims = &.{1},
    };
    try std.testing.expect(!is_matrix_matmul(wrong_contract));
}

test is_canonical_batched_matmul {
    const dg: pr.DotGeneralParams = .{
        .lhs_batch_dims = &.{ 0, 1 },
        .rhs_batch_dims = &.{ 0, 1 },
        .lhs_contracting_dims = &.{3},
        .rhs_contracting_dims = &.{2},
    };
    try std.testing.expect(is_canonical_batched_matmul(dg, 4, 4));
}

test "is_canonical_batched_matmul rejects non-prefix batches" {
    const dg: pr.DotGeneralParams = .{
        .lhs_batch_dims = &.{1},
        .rhs_batch_dims = &.{1},
        .lhs_contracting_dims = &.{2},
        .rhs_contracting_dims = &.{1},
    };
    try std.testing.expect(!is_canonical_batched_matmul(dg, 3, 3));
}

test "is_canonical_batched_matmul rejects rank mismatch" {
    const dg: pr.DotGeneralParams = .{
        .lhs_batch_dims = &.{0},
        .rhs_batch_dims = &.{0},
        .lhs_contracting_dims = &.{2},
        .rhs_contracting_dims = &.{1},
    };
    try std.testing.expect(!is_canonical_batched_matmul(dg, 3, 4));
}
