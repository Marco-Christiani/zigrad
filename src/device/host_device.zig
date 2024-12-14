const std = @import("std");
pub const backend = @import("root.zig").backend;
const builtin = @import("builtin");
const DimensionMap = @import("dimension_map.zig");
const RandType = @import("device_common.zig").RandType;

fn host_reference(self: *HostDevice) DeviceReference {
    return self;
}

pub const DeviceReference = *HostDevice;

pub const using_mkl: bool = blk: {
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
    ) void {
        switch (T) {
            f32 => z[0] = c.cblas_sdot(@intCast(x.len), x.ptr, 1, y.ptr, 1),
            f64 => z[0] = c.cblas_ddot(@intCast(x.len), x.ptr, 1, y.ptr, 1),
            else => std.debug.panic("Unsupported type {}\n", .{@typeName(T)}),
        }
    }

    pub fn dot(
        _: Blas,
        T: type,
        x: []const T,
        y: []const T,
        z: []T,
    ) void {
        switch (T) {
            f32 => z[0] = c.cblas_sdot(@intCast(x.len), x.ptr, 1, y.ptr, 1),
            f64 => z[0] = c.cblas_ddot(@intCast(x.len), x.ptr, 1, y.ptr, 1),
            else => std.debug.panic("Unsupported type {}\n", .{@typeName(T)}),
        }
    }

    pub fn add(
        _: Blas,
        T: type,
        x: []const T,
        y: []const T,
        z: []T,
    ) void {
        for (0..x.len) |i| z[i] = x[i] + y[i];
    }

    pub fn sub(
        _: Blas,
        T: type,
        x: []const T,
        y: []const T,
        z: []T,
    ) void {
        for (0..x.len) |i| z[i] = x[i] - y[i];
    }

    pub fn dot(
        _: Blas,
        T: type,
        x: []const T,
        y: []const T,
        z: []T,
    ) void {
        switch (T) {
            f32 => z[0] = c.cblas_sdot(@intCast(x.len), x.ptr, 1, y.ptr, 1),
            f64 => z[0] = c.cblas_ddot(@intCast(x.len), x.ptr, 1, y.ptr, 1),
            else => std.debug.panic("Unsupported type {}\n", .{@typeName(T)}),
        }
    }

    pub fn dot(
        _: Blas,
        T: type,
        x: []const T,
        y: []const T,
        z: []T,
    ) void {
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

    /// L2 Norm: A = alpha(xy') + A
    /// A: (M, N)
    pub fn nrm2(
        _: Blas,
        T: type,
        x: []const T,
        y: []T,
    ) void {
        switch (T) {
            f32 => y[0] = c.cblas_snrm2(@intCast(x.len), x.ptr, 1),
            f64 => y[0] = c.cblas_dnrm2(@intCast(x.len), x.ptr, 1),
            else => @compileError("Unsupported type" ++ @typeName(T)),
        }
    }

    pub fn max_forward(
        _: Blas,
        T: type,
        src: []const T,
        dst: []T,
        idx: *i32,
        incX: usize,
    ) void {
        const _idx = switch (T) {
            f32 => c.cblas_isamax(@intCast(src.len), src.ptr, incX),
            f64 => c.cblas_idamax(@intCast(src.len), src.ptr, incX),
            else => @compileError("Unsupported type for BLAS max"),
        };
        idx.* = @intCast(_idx);
        dst[0] = src[_idx];
    }

    pub fn max_reverse(
        _: Blas,
        T: type,
        y_grd: []const T,
        x_grd: []T,
        idx: *i32,
    ) void {
        const _idx: usize = @intCast(idx.*);
        x_grd[_idx] += y_grd[0];
    }

    pub fn sum(
        _: Blas,
        T: type,
        x: []const T,
        y: []T,
    ) void {
        switch (T) {
            f32 => y[0] = c.cblas_sasum(@intCast(x.len), x.ptr, 1),
            f64 => y[0] = c.cblas_dasum(@intCast(x.len), x.ptr, 1),
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

    const bmm = if (using_mkl) bmm_impl else void;

    fn bmm_impl(comptime _: type) void {}
};

pub const ScratchMemory = struct {
    head: usize = 0,
    tail: usize = 0,
    // We're using malloc for memory alignment. After one pass
    // through a network, this should never need to be resized.
    pub fn deinit(self: *ScratchMemory) void {
        if (self.head != 0) {
            std.c.free(@ptrFromInt(self.head));
        }
        self.* = undefined;
    }
    pub fn get(self: *ScratchMemory, comptime T: type, n: usize) []T {
        const total: usize = @sizeOf(T) * n;
        // check if we have enough scratch to provide a payload
        if (self.tail < (self.head + total)) {
            if (self.head != 0) {
                std.c.free(@ptrFromInt(self.head));
            }
            const ptr = std.c.malloc(total) orelse @panic("Cannot allocate scratch memory");
            self.head = @intFromPtr(ptr);
            self.tail = self.head + total;
        }
        const ptr: [*]T = @ptrFromInt(self.head);
        return ptr[0..n];
    }
};

pub const NN = struct {
    pub fn clip_norm(self: *const NN, comptime T: type, x_val: []T, max_norm: T, delta: T) f64 {
        var nrm: [1]T = undefined;
        self.parent().blas.nrm2(T, x_val, nrm[0..]);

        const norm = nrm[0];
        if (norm > max_norm) {
            const scale = max_norm / (norm + delta);
            for (self.data) |*value| {
                value.* *= scale;
            }
        }
    }

    pub fn relu_forward(_: NN, comptime T: type, x: []const T, y: []T) void {
        for (x, y) |x_v, *y_v| {
            y_v.* = if (x_v > 0) x_v else 0;
        }
    }

    pub fn relu_backward(_: NN, comptime T: type, x_val: []const T, y_grd: []const T, x_grd: []T) void {
        for (x_val, y_grd, x_grd) |x_v, y_g, *x_g| {
            x_g.* += if (x_v > 0) y_g else 0;
        }
    }

    fn parent(self: *const NN) *const HostDevice {
        return @alignCast(@fieldParentPtr("nn", self));
    }
};

pub const HostDevice = struct {
    const Error = std.mem.Allocator.Error;

    nn: NN,
    blas: Blas,
    scratch: ScratchMemory,
    cache: DimensionMap,
    allocator: std.mem.Allocator,

    pub fn init(backing_allocator: std.mem.Allocator) HostDevice {
        return .{
            .nn = .{},
            .blas = .{},
            .scratch = .{},
            .cache = .{ .allocator = backing_allocator },
            .allocator = backing_allocator,
        };
    }

    pub fn deinit(self: *HostDevice) void {
        self.scratch.deinit();
        self.cache.deinit();
        self.* = undefined;
    }

    pub fn mem_alloc(self: HostDevice, comptime T: type, n: usize) Error![]T {
        return self.allocator.alloc(T, n);
    }

    pub fn mem_free(self: HostDevice, slice: anytype) void {
        return self.allocator.free(slice);
    }

    pub fn mem_create(self: HostDevice, comptime T: type) Error!*T {
        return self.allocator.create(T);
    }

    pub fn mem_destroy(self: HostDevice, slice: anytype) void {
        return self.allocator.destroy(slice);
    }

    pub fn mem_dupe(self: HostDevice, comptime T: type, src: []const T) Error![]T {
        return self.allocator.dupe(T, src);
    }

    pub fn mem_copy(_: HostDevice, comptime T: type, src: []const T, dst: []T) void {
        @memcpy(dst, src);
    }

    pub fn mem_fill(_: HostDevice, comptime T: type, slice: []T, value: T) void {
        @memset(slice, value);
    }

    pub fn mem_random(_: HostDevice, comptime T: type, slice: []T, op: RandType, seed: u64) void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rand = prng.random();
        if (op == .uniform) {
            for (slice) |*e| e.* = rand.float(T);
        } else {
            for (slice) |*e| e.* = rand.floatNorm(T);
        }
    }

    // remove data dependencies on this to speed it up
    pub fn mem_sequence(_: HostDevice, comptime T: type, slice: []T, initial: T, step: T) void {
        var current = initial; // move from register memory
        for (slice) |*x| {
            x.* = current;
            current += step;
        }
    }

    pub fn sync(_: HostDevice) void {}

    // since host is it's own reference when compiling for HOST only,
    // this is always trivially true. Only for debug compatibility.
    pub inline fn is_compatible(_: *const HostDevice, _: *const HostDevice) bool {
        return true;
    }

    pub fn is_host(_: DeviceReference) bool {
        return true;
    }

    pub const reference = switch (backend) {
        .HOST => host_reference,
        .CUDA => @import("cuda_device.zig").host_reference,
    };
};
