const std = @import("std");
const opspec = @import("../opspec.zig");
const HostDevice = @import("../host_device.zig");
const builtin = @import("builtin");
const build_options = @import("build_options");

pub const using_mkl_blas: bool = blk: {
    const decls = @typeInfo(c).Struct.decls;
    for (decls) |decl| {
        if (std.mem.startsWith(u8, decl.name, "mkl_") or std.mem.startsWith(u8, decl.name, "MKL_")) {
            break :blk true;
        }
    }
    break :blk false;
};

pub const c = switch (builtin.target.os.tag) {
    .linux => @cImport({
        if (build_options.enable_mkl) {
            @cInclude("mkl_vml_functions.h");
            @cInclude("mkl_cblas.h");
        } else {
            @cInclude("cblas.h");
        }
    }),
    .macos => @cImport(@cInclude("Accelerate/Accelerate.h")),
    else => @compileError("Unsupported os"),
};

pub fn dot(_: *const HostDevice, T: type, p: opspec.dot(T)) void {
    switch (T) {
        f32 => p.z[0] = c.cblas_sdot(@intCast(p.x.len), p.x.ptr, 1, p.y.ptr, 1),
        f64 => p.z[0] = c.cblas_ddot(@intCast(p.x.len), p.x.ptr, 1, p.y.ptr, 1),
        else => @compileError("Unsupported type for BLAS dot" ++ @typeName(T) ++ " for platform " ++ @tagName(builtin.target.os.tag)),
    }
}

pub fn matvec(_: *const HostDevice, T: type, p: opspec.matvec(T)) void {
    const lda = p.n;
    const ta = if (p.trans_a) c.CblasTrans else c.CblasNoTrans;
    switch (T) {
        f32 => c.cblas_sgemv(c.CblasRowMajor, @intCast(ta), @intCast(p.m), @intCast(p.n), p.alpha, p.A.ptr, @intCast(lda), p.x.ptr, 1, p.beta, p.y.ptr, 1),
        f64 => c.cblas_dgemv(c.CblasRowMajor, @intCast(ta), @intCast(p.m), @intCast(p.n), p.alpha, p.A.ptr, @intCast(lda), p.x.ptr, 1, p.beta, p.y.ptr, 1),
        else => @compileError("Unsupported type for BLAS demv" ++ @typeName(T) ++ " for platform " ++ @tagName(builtin.target.os.tag)),
    }
}

pub fn matmul(_: *const HostDevice, T: type, p: opspec.matmul(T)) void {
    const ta = if (p.trans_a) c.CblasTrans else c.CblasNoTrans;
    const tb = if (p.trans_b) c.CblasTrans else c.CblasNoTrans;
    switch (T) {
        f32 => c.cblas_sgemm(c.CblasRowMajor, @intCast(ta), @intCast(tb), @intCast(p.m), @intCast(p.n), @intCast(p.k), p.alpha, p.A.ptr, @intCast(p.lda), p.B.ptr, @intCast(p.ldb), p.beta, p.C.ptr, @intCast(p.ldc)),
        f64 => c.cblas_dgemm(c.CblasRowMajor, @intCast(ta), @intCast(tb), @intCast(p.m), @intCast(p.n), @intCast(p.k), p.alpha, p.A.ptr, @intCast(p.lda), p.B.ptr, @intCast(p.ldb), p.beta, p.C.ptr, @intCast(p.ldc)),
        else => @compileError("Unsupported type for BLAS gemm" ++ @typeName(T) ++ " for platform " ++ @tagName(builtin.target.os.tag)),
    }
}

pub fn outer(_: *const HostDevice, T: type, p: opspec.outer(T)) void {
    switch (T) {
        f32 => c.cblas_sger(c.CblasRowMajor, @intCast(p.x.len), @intCast(p.y.len), p.alpha, p.x.ptr, 1, p.y.ptr, 1, p.A.ptr, @intCast(p.y.len)),
        f64 => c.cblas_dger(c.CblasRowMajor, @intCast(p.x.len), @intCast(p.y.len), p.alpha, p.x.ptr, 1, p.y.ptr, 1, p.A.ptr, @intCast(p.y.len)),
        else => @compileError("Unsupported type for BLAS ger" ++ @typeName(T) ++ " for platform " ++ @tagName(builtin.target.os.tag)),
    }
}

// TODO: extend to greater than 2D and optimize this
pub fn transpose(_: *const HostDevice, T: type, p: opspec.transpose(T)) void {
    for (0..p.m) |i| {
        for (0..p.n) |j| {
            p.B[j * p.m + i] = p.A[i * p.n + j] + p.alpha * p.B[j * p.m + i];
        }
    }
}

pub fn axpy(_: *const HostDevice, T: type, p: opspec.axpy(T)) void {
    switch (T) {
        f32 => c.cblas_saxpy(@intCast(p.x.len), p.alpha.*, p.x.ptr, 1, p.y.ptr, 1),
        f64 => c.cblas_daxpy(@intCast(p.x.len), p.alpha.*, p.x.ptr, 1, p.y.ptr, 1),
        else => @compileError("Unsupported type for BLAS axpy" ++ @typeName(T) ++ " for platform " ++ @tagName(builtin.target.os.tag)),
    }
}

pub fn bmm_acc(self: *const HostDevice, T: type, p: opspec.bmm_acc(T)) void {
    const n_batches_a = p.A_shape[0];
    const n_batches_b = p.B_shape[0];
    const n_batches_c = p.C_shape[0];
    const A_chunk = p.A_shape[1] * p.A_shape[2];
    const B_chunk = p.B_shape[1] * p.B_shape[2];
    const C_chunk = p.C_shape[1] * p.C_shape[2];

    for (0..n_batches_c) |i| {
        const a_index = i % n_batches_a;
        const b_index = i % n_batches_b;

        const a_start = a_index * A_chunk;
        const b_start = b_index * B_chunk;
        const c_start = i * C_chunk;

        const a_slice = p.A[a_start .. a_start + A_chunk];
        const b_slice = p.B[b_start .. b_start + B_chunk];
        const c_slice = p.C[c_start .. c_start + C_chunk];

        self.matmul(T, .{
            .A = a_slice,
            .B = b_slice,
            .C = c_slice,
            .m = p.C_shape[1],
            .n = p.C_shape[2],
            .k = p.A_shape[2],
            .trans_a = p.trans_a,
            .trans_b = p.trans_b,
            .lda = p.lda,
            .ldb = p.ldb,
            .ldc = p.ldc,
            .alpha = p.alpha,
            .beta = p.beta,
        });
    }
}

pub fn sum(_: *const HostDevice, T: type, p: opspec.sum(T)) void {
    switch (T) {
        f32 => p.y[0] = c.cblas_sasum(@intCast(p.x.len), p.x.ptr, 1),
        f64 => p.y[0] = c.cblas_dasum(@intCast(p.x.len), p.x.ptr, 1),
        else => @compileError("Unsupported type for BLAS asum" ++ @typeName(T) ++ " for platform " ++ @tagName(builtin.target.os.tag)),
    }
}

pub fn scale(_: *const HostDevice, T: type, p: opspec.scale(T)) void {
    switch (T) {
        f32 => c.cblas_sscal(@intCast(p.x.len), p.alpha, p.x.ptr, 1),
        f64 => c.cblas_dscal(@intCast(p.x.len), p.alpha, p.x.ptr, 1),
        else => @compileError("Unsupported type for BLAS scal" ++ @typeName(T) ++ " for platform " ++ @tagName(builtin.target.os.tag)),
    }
}

pub fn nrm2(_: *const HostDevice, T: type, p: opspec.nrm2(T)) void {
    switch (T) {
        f32 => p.y[0] = c.cblas_snrm2(@intCast(p.x.len), p.x.ptr, 1),
        f64 => p.y[0] = c.cblas_dnrm2(@intCast(p.x.len), p.x.ptr, 1),
        else => @compileError("Unsupported type for BLAS nrm2" ++ @typeName(T) ++ " for platform " ++ @tagName(builtin.target.os.tag)),
    }
}

pub fn clip_nrm2(self: *const HostDevice, T: type, p: opspec.clip_nrm2(T)) void {
    var scratch: [1]T = undefined;
    self.nrm2(T, .{ .x = p.x, .y = scratch[0..] });
    const norm = scratch[0];
    if (norm > p.max_norm) {
        self.scale(T, .{ .x = p.x, .alpha = p.max_norm / (norm + p.delta) });
    }
}
