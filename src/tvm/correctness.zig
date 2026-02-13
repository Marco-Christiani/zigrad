//! Reference implementations and correctness verification for TVM modules.
//!
//! This module provides reference implementations for verifying TVM-compiled
//! kernels against known-correct CPU implementations.

const std = @import("std");

const blas = @cImport({
    @cInclude("mkl_cblas.h");
});

/// Compute attention with MKL loop (one GEMM per batch).
pub fn compute_attention_mkl_loop(
    batch: usize,
    seq: usize,
    head_dim: usize,
    q_data: []const f32,
    k_data: []const f32,
    v_data: []const f32,
    scores: []f32,
    output: []f32,
) void {
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    for (0..batch) |b| {
        const q_batch = q_data[b * seq * head_dim ..][0 .. seq * head_dim];
        const k_batch = k_data[b * seq * head_dim ..][0 .. seq * head_dim];
        const v_batch = v_data[b * seq * head_dim ..][0 .. seq * head_dim];
        const scores_batch = scores[b * seq * seq ..][0 .. seq * seq];
        const output_batch = output[b * seq * head_dim ..][0 .. seq * head_dim];

        blas.cblas_sgemm(blas.CblasRowMajor, blas.CblasNoTrans, blas.CblasTrans, @intCast(seq), @intCast(seq), @intCast(head_dim), scale, q_batch.ptr, @intCast(head_dim), k_batch.ptr, @intCast(head_dim), 0.0, scores_batch.ptr, @intCast(seq));

        for (0..seq) |i| {
            const row = scores_batch[i * seq ..][0..seq];
            softmax_inplace(row);
        }

        blas.cblas_sgemm(blas.CblasRowMajor, blas.CblasNoTrans, blas.CblasNoTrans, @intCast(seq), @intCast(head_dim), @intCast(seq), 1.0, scores_batch.ptr, @intCast(seq), v_batch.ptr, @intCast(head_dim), 0.0, output_batch.ptr, @intCast(head_dim));
    }
}

/// Compute attention output using MKL batched GEMM: output = softmax(Q @ K^T / sqrt(head_dim)) @ V.
///
/// Caller must provide pre-allocated output and scores buffers.
pub fn compute_attention_mkl(
    batch: usize,
    seq: usize,
    head_dim: usize,
    q_data: []const f32,
    k_data: []const f32,
    v_data: []const f32,
    scores: []f32, // Pre-allocated: batch * seq * seq
    output: []f32, // Pre-allocated: batch * seq * head_dim
) void {
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    // 1. Q @ K^T using mkl batched gemm (scores[B,S,S] = Q[B,S,D] @ K^T[B,D,S])
    blas.cblas_sgemm_batch_strided(
        blas.CblasRowMajor,
        blas.CblasNoTrans, // Q is S×D
        blas.CblasTrans, // K^T is D×S
        @intCast(seq), // M
        @intCast(seq), // N
        @intCast(head_dim), // K
        scale, // alpha (includes scaling)
        q_data.ptr,
        @intCast(head_dim), // lda
        @intCast(seq * head_dim), // stride_a
        k_data.ptr,
        @intCast(head_dim), // ldb
        @intCast(seq * head_dim), // stride_b
        0.0, // beta
        scores.ptr,
        @intCast(seq), // ldc
        @intCast(seq * seq), // stride_c
        @intCast(batch), // batch_size
    );

    // 2. batched softmax over last dim (row-wise)
    for (0..batch) |b| {
        for (0..seq) |i| {
            const row_start = b * seq * seq + i * seq;
            const row = scores[row_start .. row_start + seq];
            softmax_inplace(row);
        }
    }

    // 3. scores @ V using mkl batched gemm (output[B,S,D] = scores[B,S,S] @ V[B,S,D])
    blas.cblas_sgemm_batch_strided(
        blas.CblasRowMajor,
        blas.CblasNoTrans, // scores is S×S
        blas.CblasNoTrans, // V is S×D
        @intCast(seq), // M
        @intCast(head_dim), // N
        @intCast(seq), // K
        1.0, // alpha
        scores.ptr,
        @intCast(seq), // lda
        @intCast(seq * seq), // stride_a
        v_data.ptr,
        @intCast(head_dim), // ldb
        @intCast(seq * head_dim), // stride_b
        0.0, // beta
        output.ptr,
        @intCast(head_dim), // ldc
        @intCast(seq * head_dim), // stride_c
        @intCast(batch), // batch_size
    );
}

/// Verify attention output against a reference.
///
/// Computes: output = softmax(Q @ K^T / sqrt(head_dim)) @ V
///
/// Returns max absolute error between actual and expected outputs.
pub fn verify_attention(
    allocator: std.mem.Allocator,
    batch: usize,
    seq: usize,
    head_dim: usize,
    q_data: []const f32,
    k_data: []const f32,
    v_data: []const f32,
    actual_output: []const f32,
) !f32 {
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    std.log.debug("Verifying attention: batch={d}, seq={d}, head_dim={d}, scale={d:.4}", .{ batch, seq, head_dim, scale });
    std.log.debug("Sample Q[0,0,0:3]: [{d:.4}, {d:.4}, {d:.4}]", .{ q_data[0], q_data[1], q_data[2] });
    std.log.debug("Sample actual_output[0,0,0:3]: [{d:.4}, {d:.4}, {d:.4}]", .{ actual_output[0], actual_output[1], actual_output[2] });

    // allocate intermediates
    const scores = try allocator.alloc(f32, batch * seq * seq);
    defer allocator.free(scores);
    const expected = try allocator.alloc(f32, batch * seq * head_dim);
    defer allocator.free(expected);

    // 1. Q @ K^T -> scores[B,S,S]
    for (0..batch) |b| {
        for (0..seq) |i| {
            for (0..seq) |j| {
                var sum: f32 = 0;
                for (0..head_dim) |d| {
                    const q_val = q_data[b * seq * head_dim + i * head_dim + d];
                    const k_val = k_data[b * seq * head_dim + j * head_dim + d];
                    sum += q_val * k_val;
                }
                scores[b * seq * seq + i * seq + j] = sum * scale;
            }
        }
    }

    // 2. softmax over last dim
    for (0..batch) |b| {
        for (0..seq) |i| {
            const row_start = b * seq * seq + i * seq;
            const row = scores[row_start .. row_start + seq];
            softmax_inplace(row);
        }
    }

    // 3. scores @ V -> output[B,S,D]
    for (0..batch) |b| {
        for (0..seq) |i| {
            for (0..head_dim) |d| {
                var sum: f32 = 0;
                for (0..seq) |j| {
                    const score_val = scores[b * seq * seq + i * seq + j];
                    const v_val = v_data[b * seq * head_dim + j * head_dim + d];
                    sum += score_val * v_val;
                }
                expected[b * seq * head_dim + i * head_dim + d] = sum;
            }
        }
    }

    // compute max abs err and check for nans
    var max_error: f32 = 0;
    var has_nan = false;
    for (expected, actual_output, 0..) |exp, act, i| {
        if (std.math.isNan(exp) or std.math.isNan(act)) {
            has_nan = true;
            std.log.err("NaN at index {d}: expected={}, actual={}", .{ i, exp, act });
            if (i < 5) continue; // Only log first few
        }
        const err = @abs(act - exp);
        if (err > max_error) {
            max_error = err;
            if (max_error > 0.1) {
                std.log.debug("Large error at idx {d}: expected={d:.4}, actual={d:.4}, diff={d:.4}", .{ i, exp, act, err });
            }
        }
    }
    if (has_nan) return error.NaNDetected;

    return max_error;
}

/// In-place stable softmax
fn softmax_inplace(slice: []f32) void {
    if (slice.len == 0) return;

    // find max for stability
    var max_val = slice[0];
    for (slice[1..]) |v| max_val = @max(max_val, v);

    // sum-exp
    var sum: f32 = 0;
    for (slice) |*v| {
        v.* = @exp(v.* - max_val);
        sum += v.*;
    }

    // normalize
    for (slice) |*v| v.* /= sum;
}

test "softmax basic" {
    var data = [_]f32{ 1.0, 2.0, 3.0 };
    softmax_inplace(&data);

    // Should sum to 1.0
    var sum: f32 = 0;
    for (data) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-6);

    // Largest input should have largest output
    try std.testing.expect(data[2] > data[1]);
    try std.testing.expect(data[1] > data[0]);
}
