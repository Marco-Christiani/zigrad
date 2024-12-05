const std = @import("std");
const backend = @import("root.zig").backend;
const builtin = @import("builtin");

fn host_reference(self: *HostDevice) DeviceReference {
    return self;
}

pub const DeviceReference = *HostDevice;

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

pub const Blas = struct {
    /// Computes dot product assuming a stride of 1 and row-major. (N,) x (N,) = (1,)
    pub fn dot(
        _: Blas,
        T: type,
        x: []const T,
        y: []const T,
        z: []T,
    ) T {
        switch (T) {
            f32 => z[0] = c.cblas_sdot(@intCast(x.len), x.ptr, 1, y.ptr, 1),
            f64 => z[0] = c.cblas_ddot(@intCast(x.len), x.ptr, 1, y.ptr, 1),
            else => std.debug.panic("Unsupported type {}\n", .{@typeName(T)}),
        }
    }

    /// Computes mat-vec assuming a stride of 1 for the vec and row-major.
    /// a * (M, N) x (N,) + b * (N,) = (M,)
    /// Y = aAX + bY
    pub fn matvec(
        _: Blas,
        T: type,
        A: []const T,
        x: []const T,
        y: []T,
        M: usize,
        N: usize,
        trans_a: bool,
        alpha: T,
        beta: T,
    ) void {
        const lda = N;
        const ta = if (trans_a) c.CblasTrans else c.CblasNoTrans;
        switch (T) {
            f32 => c.cblas_sgemv(c.CblasRowMajor, @intCast(ta), @intCast(M), @intCast(N), alpha, A.ptr, @intCast(lda), x.ptr, 1, beta, y.ptr, 1),
            f64 => c.cblas_dgemv(c.CblasRowMajor, @intCast(ta), @intCast(M), @intCast(N), alpha, A.ptr, @intCast(lda), x.ptr, 1, beta, y.ptr, 1),
            else => std.debug.panic("Unsupported type {}\n", .{@typeName(T)}),
        }
    }

    /// Assumes row-major.
    /// (M, K) x (K, N) = (M, N)
    /// C := alpha*op(A)*op(B) + beta*C
    pub fn matmul(
        _: Blas,
        T: type,
        A: []const T,
        B: []const T,
        C: []T,
        M: usize,
        N: usize,
        K: usize,
        trans_a: bool,
        trans_b: bool,
        lda: usize,
        ldb: usize,
        ldc: usize,
        alpha: T,
        beta: T,
    ) void {
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
    pub fn outer(
        _: Blas,
        T: type,
        x: []const T,
        y: []const T,
        A: []T,
        alpha: T,
    ) void {
        switch (T) {
            f32 => c.cblas_sger(c.CblasRowMajor, @intCast(x.len), @intCast(y.len), alpha, x.ptr, 1, y.ptr, 1, A.ptr, @intCast(y.len)),
            f64 => c.cblas_dger(c.CblasRowMajor, @intCast(x.len), @intCast(y.len), alpha, x.ptr, 1, y.ptr, 1, A.ptr, @intCast(y.len)),
            else => std.debug.panic("Unsupported type {}\n", .{@typeName(T)}),
        }
    }

    /// Outer product: A = alpha(xy') + A
    /// A: (M, N)
    pub fn nrm2(
        _: Blas,
        T: type,
        x: []T,
    ) T {
        return switch (T) {
            f32 => c.cblas_snrm2(@intCast(x.len), x.ptr, 1),
            f64 => c.cblas_dnrm2(@intCast(x.len), x.ptr, 1),
            else => @compileError("Unsupported type" ++ @typeName(T)),
        };
    }

    pub fn maxForward(
        _: Blas,
        T: type,
        src: []const T,
        dst: []T,
        idx: *i32,
    ) void {
        const _idx = switch (T) {
            f32 => c.cblas_isamax(@intCast(src.len), src.ptr, 1),
            f64 => c.cblas_idamax(@intCast(src.len), src.ptr, 1),
            else => @compileError("Unsupported type for BLAS max"),
        };
        idx.* = @intCast(_idx);
        dst[0] = src[_idx];
    }

    pub fn maxReverse(
        _: Blas,
        T: type,
        y_grd: []const T,
        x_grd: []T,
        idx: *i32,
    ) void {
        const _idx: usize = @intCast(idx.*);
        x_grd[0] += y_grd[_idx];
    }

    pub fn sum(
        _: Blas,
        T: type,
        x: []const T,
        z: []T,
    ) T {
        switch (T) {
            f32 => z[0] = c.cblas_sasum(@intCast(x.len), x.ptr, 1),
            f64 => z[0] = c.cblas_dasum(@intCast(x.len), x.ptr, 1),
            else => @compileError("Unsupported type for BLAS sum"),
        }
    }

    pub fn scale(
        _: Blas,
        T: type,
        x: []T,
        alpha: T,
    ) void {
        switch (T) {
            f32 => c.cblas_sscal(@intCast(x.len), alpha, x.ptr, 1),
            f64 => c.cblas_dscal(@intCast(x.len), alpha, x.ptr, 1),
            else => @compileError("Unsupported type for BLAS scale"),
        }
    }

    pub fn axpy(
        _: Blas,
        comptime T: type,
        x: []const T,
        y: []T,
        alpha: *const T,
    ) void {
        switch (T) {
            f32 => c.cblas_saxpy(@intCast(x.len), alpha.*, x.ptr, 1, y.ptr, 1),
            f64 => c.cblas_daxpy(@intCast(x.len), alpha.*, x.ptr, 1, y.ptr, 1),
            else => @compileError("Unsupported type for blas_axpy: " ++ @typeName(T)),
        }
    }
};

pub const HostDevice = struct {
    const Error = std.mem.Allocator.Error;

    blas: Blas,
    allocator: std.mem.Allocator,

    pub fn init(backing_allocator: std.mem.Allocator) HostDevice {
        return .{ .allocator = backing_allocator, .blas = .{} };
    }

    pub fn deinit(self: *HostDevice) void {
        // more in the future when we bring in caching
        self.* = undefined;
    }

    pub fn memAlloc(self: HostDevice, comptime T: type, n: usize) Error![]T {
        return self.allocator.alloc(T, n);
    }

    pub fn memFree(self: HostDevice, slice: anytype) void {
        return self.allocator.free(slice);
    }

    pub fn memCreate(self: HostDevice, comptime T: type) Error!*T {
        return self.allocator.create(T);
    }

    pub fn memDestroy(self: HostDevice, slice: anytype) void {
        return self.allocator.destroy(slice);
    }

    pub fn memDupe(self: HostDevice, comptime T: type, src: []const T) Error![]T {
        return self.allocator.dupe(T, src);
    }

    pub fn memFill(_: HostDevice, comptime T: type, slice: []T, value: T) void {
        @memset(slice, value);
    }

    // remove data dependencies on this to speed it up
    pub fn memSequence(_: HostDevice, comptime T: type, slice: []T, initial: T, step: T) void {
        var current = initial; // move from register memory
        for (slice) |*x| {
            x.* = current;
            current += step;
        }
    }

    pub fn sync(_: HostDevice) void {}

    // since host is it's own reference when compiling for HOST only,
    // this is always trivially true. Only for debug compatibility.
    pub inline fn isCompatible(_: *const HostDevice, _: *const HostDevice) bool {
        return true;
    }

    pub fn isHost(_: DeviceReference) bool {
        return true;
    }

    pub const reference: fn (self: *HostDevice) DeviceReference = switch (backend) {
        .HOST => host_reference,
        .CUDA => @import("cuda_device.zig").host_reference,
    };
};
