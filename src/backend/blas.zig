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

///  Assumes row-major.
///  (M, K) x (K, N) = (M, N)
/// C := alpha*op(A)*op(B) + beta*C
pub fn blas_matmul(T: type, A: []T, B: []T, C: []T, M: usize, N: usize, K: usize, trans_a: bool, trans_b: bool, lda: usize, ldb: usize, ldc: usize, alpha: T, beta: T) void {
    const ta = if (trans_a) c.CblasTrans else c.CblasNoTrans;
    const tb = if (trans_b) c.CblasTrans else c.CblasNoTrans;
    switch (T) {
        f32 => c.cblas_sgemm(c.CblasRowMajor, @intCast(ta), @intCast(tb), @intCast(M), @intCast(N), @intCast(K), alpha, A.ptr, @intCast(lda), B.ptr, @intCast(ldb), beta, C.ptr, @intCast(ldc)),
        f64 => c.cblas_dgemm(c.CblasRowMajor, @intCast(ta), @intCast(tb), @intCast(M), @intCast(N), @intCast(K), alpha, A.ptr, @intCast(lda), B.ptr, @intCast(ldb), beta, C.ptr, @intCast(ldc)),
        else => std.debug.panic("Unsupported type {}\n", .{@typeName(T)}),
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
