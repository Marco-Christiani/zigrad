/// Public API for hand-rolled naive GEMM kernel and BLAS baselines.
const std = @import("std");
const build_options = @import("build_options");

const gemm_naive = @import("gemm_naive.zig");

/// Single-precision matmul: C = A @ B.
/// A is M×K, B is K×N, C is M×N (row-major layout).
///
/// Simple triple-loop baseline implementation (i-k-j order).
/// Caller must zero C before calling if accumulation is not desired.
pub fn gemm_f32(
    m: usize,
    n: usize,
    k: usize,
    /// Input matrix A (M×K).
    a: []const f32,
    /// Leading dimension of A (stride).
    lda: usize,
    /// Input matrix B (K×N).
    b: []const f32,
    /// Leading dimension of B (stride).
    ldb: usize,
    /// Output matrix C (M×N), accumulated into.
    c: []f32,
    /// Leading dimension of C (stride).
    ldc: usize,
) void {
    gemm_naive.gemm_f32(m, n, k, a, lda, b, ldb, c, ldc);
}

test "gemm: naive 2x2 matmul" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var c = [_]f32{0.0} ** 4;

    const expected = [_]f32{ 19.0, 22.0, 43.0, 50.0 };

    gemm_f32(2, 2, 2, &a, 2, &b, 2, &c, 2);

    for (expected, c) |exp, act| {
        try std.testing.expectApproxEqAbs(exp, act, 1e-5);
    }
}

const blas = if (build_options.enable_mkl) @cImport({
    @cInclude("mkl_cblas.h");
}) else @compileError("MKL not available -- rebuild with MKL in SDK");

pub fn blas_gemm_f32(
    m: usize,
    n: usize,
    k: usize,
    /// Input matrix A (M×K).
    a: []const f32,
    /// Leading dimension of A (stride).
    lda: usize,
    /// Input matrix B (K×N).
    b: []const f32,
    /// Leading dimension of B (stride).
    ldb: usize,
    /// Output matrix C (M×N), accumulated into.
    c: []f32,
    /// Leading dimension of C (stride).
    ldc: usize,
) void {
    blas.cblas_sgemm(
        blas.CblasRowMajor,
        blas.CblasNoTrans,
        blas.CblasNoTrans,
        @intCast(m),
        @intCast(n),
        @intCast(k),
        1.0,
        a.ptr,
        @intCast(lda),
        b.ptr,
        @intCast(ldb),
        0,
        c.ptr,
        @intCast(ldc),
    );
}
