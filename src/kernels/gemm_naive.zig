//! Naive matmul implementation (baseline, no optimizations).
//! TODO: again, outdated scratch, see benchmark/root.zig, same thing.
const std = @import("std");

/// Naive strawman triple-loop matmul: C += A @ B.
pub fn gemm_f32(
    m: usize,
    n: usize,
    k: usize,
    /// Input matrix A (MxK).
    a: []const f32,
    /// Leading dimension of A (stride).
    lda: usize,
    /// Input matrix B (KxN).
    b: []const f32,
    /// Leading dimension of B (stride).
    ldb: usize,
    /// Output matrix C (MxN), assumed pre-zeroed.
    c: []f32,
    /// Leading dimension of C (stride).
    ldc: usize,
) void {
    // C[i,j] += A[i,k_idx] * B[k_idx,j]
    for (0..m) |i| {
        for (0..k) |k_idx| {
            const a_val = a[i * lda + k_idx];
            for (0..n) |j| {
                c[i * ldc + j] += a_val * b[k_idx * ldb + j];
            }
        }
    }
}

test "gemm_naive: 2x2 matmul" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 }; // [[1, 2], [3, 4]]
    const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 }; // [[5, 6], [7, 8]]
    var c = [_]f32{0.0} ** 4;

    gemm_f32(2, 2, 2, &a, 2, &b, 2, &c, 2);

    // Expected: [[19, 22], [43, 50]]
    const expected = [_]f32{ 19.0, 22.0, 43.0, 50.0 };
    try std.testing.expectEqualSlices(f32, &expected, &c);
}
