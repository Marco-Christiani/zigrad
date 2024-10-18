const std = @import("std");
const builtin = @import("builtin");
const using_mkl = blk: {
    const decls = @typeInfo(c).Struct.decls;
    for (decls) |decl| {
        if (std.mem.startsWith(u8, decl.name, "mkl_") or std.mem.startsWith(u8, decl.name, "MKL_")) {
            break :blk true;
        }
    }
    break :blk false;
};
const c = switch (builtin.target.os.tag) {
    .linux => @cImport(@cInclude("cblas.h")),
    .macos => @cImport(@cInclude("Accelerate/Accelerate.h")),
    else => @compileError("Unsupported os"),
};

/// Computes dot product assuming a stride of 1 and row-major. (N,) x (N,) = (1,)
pub fn blas_dot(T: type, A: []T, B: []T) T {
    switch (T) {
        f32 => return c.cblas_sdot(@intCast(A.len), A.ptr, 1, B.ptr, 1),
        f64 => return c.cblas_ddot(@intCast(A.len), A.ptr, 1, B.ptr, 1),
        else => std.debug.panic("Unsupported type {}\n", .{@typeName(T)}),
    }
}

/// Computes mat-vec assuming a stride of 1 for the vec and row-major.
/// a * (M, N) x (N,) + b * (N,) = (M,)
/// Y = aAX + bY
pub fn blas_matvec(T: type, A: []T, X: []T, Y: []T, M: usize, N: usize, alpha: T, beta: T, trans_a: bool) void {
    const lda = N;
    const ta = if (trans_a) c.CblasTrans else c.CblasNoTrans;
    switch (T) {
        f32 => c.cblas_sgemv(c.CblasRowMajor, @intCast(ta), @intCast(M), @intCast(N), alpha, A.ptr, @intCast(lda), X.ptr, 1, beta, Y.ptr, 1),
        f64 => c.cblas_dgemv(c.CblasRowMajor, @intCast(ta), @intCast(M), @intCast(N), alpha, A.ptr, @intCast(lda), X.ptr, 1, beta, Y.ptr, 1),
        else => std.debug.panic("Unsupported type {}\n", .{@typeName(T)}),
    }
}

// pub const blas_matmul = vanilla_blas_matmul;
// pub const blas_matmul = mkl_direct_matmul;
pub const blas_matmul = mkl_jit_matmul;

///  Assumes row-major.
///  (M, K) x (K, N) = (M, N)
pub fn vanilla_blas_matmul(T: type, A: []T, B: []T, C: []T, M: usize, N: usize, K: usize, trans_a: bool, trans_b: bool, lda: usize, ldb: usize, ldc: usize, alpha: T, beta: T) void {
    const ta = if (trans_a) c.CblasTrans else c.CblasNoTrans;
    const tb = if (trans_b) c.CblasTrans else c.CblasNoTrans;
    switch (T) {
        f32 => c.cblas_sgemm(c.CblasRowMajor, @intCast(ta), @intCast(tb), @intCast(M), @intCast(N), @intCast(K), alpha, A.ptr, @intCast(lda), B.ptr, @intCast(ldb), beta, C.ptr, @intCast(ldc)),
        f64 => c.cblas_dgemm(c.CblasRowMajor, @intCast(ta), @intCast(tb), @intCast(M), @intCast(N), @intCast(K), alpha, A.ptr, @intCast(lda), B.ptr, @intCast(ldb), beta, C.ptr, @intCast(ldc)),
        else => std.debug.panic("Unsupported type {}\n", .{@typeName(T)}),
    }
}

// pub fn mkl_direct_matmul(T: type, A: []T, B: []T, C: []T, M: usize, N: usize, K: usize, trans_a: bool, trans_b: bool, lda: usize, ldb: usize, ldc: usize, alpha: T, beta: T) void {
//     const ta = if (trans_a) c.MKL_TRANS else c.MKL_NOTRANS;
//     const tb = if (trans_b) c.MKL_TRANS else c.MKL_NOTRANS;
//     switch (T) {
//         f32 => c.gemm(c.MKL_ROW_MAJOR, ta, tb, @intCast(M), @intCast(N), @intCast(K), alpha, A.ptr, @intCast(lda), B.ptr, @intCast(ldb), beta, C.ptr, @intCast(ldc)),
//         f64 => c.MKL_Dgemm(c.MKL_ROW_MAJOR, ta, tb, @intCast(M), @intCast(N), @intCast(K), alpha, A.ptr, @intCast(lda), B.ptr, @intCast(ldb), beta, C.ptr, @intCast(ldc)),
//         else => std.debug.panic("Unsupported type {}\n", .{@typeName(T)}),
//     }
// }

const mkl = @cImport(@cInclude("mkl/mkl_blas.h"));
// pub fn mkl_jit_matmul(T: type, A: []T, B: []T, C: []T, M: usize, N: usize, K: usize, trans_a: bool, trans_b: bool, lda: usize, ldb: usize, ldc: usize, alpha: T, beta: T) void {
//     const ta = if (trans_a) c.MKL_TRANS else c.MKL_NOTRANS;
//     const tb = if (trans_b) c.MKL_TRANS else c.MKL_NOTRANS;
//
//     switch (T) {
//         f32 => {
//             var jitter: ?*anyopaque = null;
//             _ = c.mkl_jit_create_sgemm(&jitter, c.MKL_ROW_MAJOR, @intCast(ta), @intCast(tb), @intCast(M), @intCast(N), @intCast(K), alpha, @intCast(lda), @intCast(ldb), beta, @intCast(ldc));
//             const sgemm_jit = c.mkl_jit_get_sgemm_ptr(jitter).?;
//             sgemm_jit(jitter, A.ptr, B.ptr, C.ptr);
//             _ = c.mkl_jit_destroy(jitter);
//         },
//         f64 => {
//             var jitter: ?*anyopaque = null;
//             _ = c.mkl_jit_create_dgemm(&jitter, c.MKL_ROW_MAJOR, ta, tb, @intCast(M), @intCast(N), @intCast(K), alpha, @intCast(lda), @intCast(ldb), beta, @intCast(ldc));
//             const dgemm_jit = c.mkl_jit_get_dgemm_ptr(jitter).?;
//             dgemm_jit(jitter, A.ptr, B.ptr, C.ptr);
//             _ = c.mkl_jit_destroy(jitter);
//         },
//         else => std.debug.panic("Unsupported type {}\n", .{@typeName(T)}),
//     }
// }

pub fn mkl_jit_matmul(T: type, A: []T, B: []T, C: []T, M: usize, N: usize, K: usize, trans_a: bool, trans_b: bool, lda: usize, ldb: usize, ldc: usize, alpha: T, beta: T) void {
    _ = alpha;
    _ = beta;
    const jitter = get_or_create_jit(T, M, N, K, trans_a, trans_b, lda, ldb, ldc) orelse @panic("Failed to create JIT function");

    switch (T) {
        f32 => {
            const sgemm_jit = c.mkl_jit_get_sgemm_ptr(jitter).?;
            sgemm_jit(jitter, A.ptr, B.ptr, C.ptr);
        },
        f64 => {
            const dgemm_jit = c.mkl_jit_get_dgemm_ptr(jitter).?;
            dgemm_jit(jitter, A.ptr, B.ptr, C.ptr);
        },
        else => @compileError("Unsupported type for JIT"),
    }
}

const JitCache = struct {
    jitter: ?*anyopaque,
    m: usize,
    n: usize,
    k: usize,
    trans_a: bool,
    trans_b: bool,
    lda: usize,
    ldb: usize,
    ldc: usize,
};

var jit_cache: [10]JitCache = .{.{ .jitter = null, .m = 0, .n = 0, .k = 0, .trans_a = false, .trans_b = false, .lda = 0, .ldb = 0, .ldc = 0 }} ** 10;
var jit_cache_index: usize = 0;

pub fn get_or_create_jit(
    comptime T: type,
    m: usize,
    n: usize,
    k: usize,
    trans_a: bool,
    trans_b: bool,
    lda: usize,
    ldb: usize,
    ldc: usize,
) ?*anyopaque {
    for (jit_cache) |cache| {
        if (cache.jitter != null and cache.m == m and cache.n == n and cache.k == k and cache.trans_a == trans_a and cache.trans_b == trans_b and cache.lda == lda and cache.ldb == ldb and cache.ldc == ldc) {
            // std.log.info("CACHED" ** 10, .{});
            return cache.jitter;
        }
    }

    const ta = if (trans_a) c.MKL_TRANS else c.MKL_NOTRANS;
    const tb = if (trans_b) c.MKL_TRANS else c.MKL_NOTRANS;

    var new_jitter: ?*anyopaque = null;
    switch (T) {
        f32 => _ = c.mkl_jit_create_sgemm(&new_jitter, c.MKL_ROW_MAJOR, @intCast(ta), @intCast(tb), @intCast(m), @intCast(n), @intCast(k), 1.0, @intCast(lda), @intCast(ldb), 0.0, @intCast(ldc)),
        f64 => _ = c.mkl_jit_create_dgemm(&new_jitter, c.MKL_ROW_MAJOR, @intCast(ta), @intCast(tb), @intCast(m), @intCast(n), @intCast(k), 1.0, @intCast(lda), @intCast(ldb), 0.0, @intCast(ldc)),
        else => @compileError("Unsupported type for JIT"),
    }

    jit_cache[jit_cache_index] = .{ .jitter = new_jitter, .m = m, .n = n, .k = k, .trans_a = trans_a, .trans_b = trans_b, .lda = lda, .ldb = ldb, .ldc = ldc };
    jit_cache_index = (jit_cache_index + 1) % jit_cache.len;

    return new_jitter;
}

pub fn cleanup_jit() void {
    for (&jit_cache) |*cache| {
        if (cache.jitter) |jitter| {
            _ = c.mkl_jit_destroy(jitter);
            cache.jitter = null;
        }
    }
}

/// Outer product: A = alpha(xy') + A
/// A: (M, N)
pub fn blas_outer(T: type, x: []T, y: []T, A: []T, alpha: T) void {
    switch (T) {
        f32 => c.cblas_sger(c.CblasRowMajor, @intCast(x.len), @intCast(y.len), alpha, x.ptr, 1, y.ptr, 1, A.ptr, @intCast(y.len)),
        f64 => c.cblas_dger(c.CblasRowMajor, @intCast(x.len), @intCast(y.len), alpha, x.ptr, 1, y.ptr, 1, A.ptr, @intCast(y.len)),
        else => std.debug.panic("Unsupported type {}\n", .{@typeName(T)}),
    }
}

/// Outer product: A = alpha(xy') + A
/// A: (M, N)
pub fn blas_nrm2(T: type, x: []T, stride: usize) T {
    return switch (T) {
        f32 => c.cblas_snrm2(@intCast(x.len), x.ptr, @intCast(stride)),
        f64 => c.cblas_dnrm2(@intCast(x.len), x.ptr, @intCast(stride)),
        else => @compileError("Unsupported type" ++ @typeName(T)),
    };
}

pub fn blas_max(T: type, x: []const T) T {
    switch (T) {
        f32 => return c.cblas_isamax(@intCast(x.len), x.ptr, 1),
        f64 => return c.cblas_idamax(@intCast(x.len), x.ptr, 1),
        else => @compileError("Unsupported type for BLAS max"),
    }
}

pub fn blas_sum(T: type, x: []const T) T {
    switch (T) {
        f32 => return c.cblas_sasum(@intCast(x.len), x.ptr, 1),
        f64 => return c.cblas_dasum(@intCast(x.len), x.ptr, 1),
        else => @compileError("Unsupported type for BLAS sum"),
    }
}

pub fn blas_scale(T: type, alpha: T, x: []T) void {
    switch (T) {
        f32 => c.cblas_sscal(@intCast(x.len), alpha, x.ptr, 1),
        f64 => c.cblas_dscal(@intCast(x.len), alpha, x.ptr, 1),
        else => @compileError("Unsupported type for BLAS scale"),
    }
}

pub fn blas_axpy(comptime T: type, n: usize, alpha: T, x: []const T, incx: usize, y: []T, incy: usize) void {
    switch (T) {
        f32 => c.cblas_saxpy(@intCast(n), alpha, x.ptr, @intCast(incx), y.ptr, @intCast(incy)),
        f64 => c.cblas_daxpy(@intCast(n), alpha, x.ptr, @intCast(incx), y.ptr, @intCast(incy)),
        else => @compileError("Unsupported type for blas_axpy: " ++ @typeName(T)),
    }
}
