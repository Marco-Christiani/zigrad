/// Correctness verification utilities for matmul benchmarks.
const std = @import("std");

/// Naive reference matmul implementation (not optimized, for verification only).
/// C = A @ B where A is MxK, B is KxN, C is MxN (row-major layout).
pub fn reference_matmul_f32(
    m: usize,
    n: usize,
    k: usize,
    a: []const f32,
    lda: usize,
    b: []const f32,
    ldb: usize,
    c: []f32,
    ldc: usize,
) void {
    // Zero output
    for (0..m) |i| {
        for (0..n) |j| {
            c[i * ldc + j] = 0.0;
        }
    }

    // Triple loop: C[i,j] += A[i,k] * B[k,j]
    for (0..m) |i| {
        for (0..k) |k_idx| {
            const a_val = a[i * lda + k_idx];
            for (0..n) |j| {
                c[i * ldc + j] += a_val * b[k_idx * ldb + j];
            }
        }
    }
}

/// Computes the maximum absolute error between two f32 arrays.
pub fn max_abs_error(expected: []const f32, actual: []const f32) f64 {
    std.debug.assert(expected.len == actual.len);

    var max_err: f64 = 0.0;
    for (expected, actual) |e, a| {
        const err = @abs(@as(f64, @floatCast(e)) - @as(f64, @floatCast(a)));
        max_err = @max(max_err, err);
    }
    return max_err;
}

/// Verifies that the actual result matches the reference within tolerance.
/// Returns the maximum absolute error.
pub fn verify(
    expected: []const f32,
    actual: []const f32,
    tolerance: f64,
) !f64 {
    const err = max_abs_error(expected, actual);
    if (err > tolerance) {
        return error.CorrectnessCheckFailed;
    }
    return err;
}

test "correctness: reference matmul 2x2" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 }; // [[1, 2], [3, 4]]
    const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 }; // [[5, 6], [7, 8]]
    var c = [_]f32{0.0} ** 4;

    reference_matmul_f32(2, 2, 2, &a, 2, &b, 2, &c, 2);

    // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    const expected = [_]f32{ 19.0, 22.0, 43.0, 50.0 };
    try std.testing.expectEqualSlices(f32, &expected, &c);
}

test "correctness: max abs error" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 1.1, 2.05, 2.95 };
    const err = max_abs_error(&a, &b);
    try std.testing.expectApproxEqAbs(0.1, err, 1e-6);
}

test "correctness: verify pass" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 1.0001, 2.0001, 3.0001 };
    const err = try verify(&a, &b, 1e-3);
    try std.testing.expect(err < 1e-3);
}

test "correctness: verify fail" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 1.1, 2.0, 3.0 };
    const result = verify(&a, &b, 1e-4);
    try std.testing.expectError(error.CorrectnessCheckFailed, result);
}
