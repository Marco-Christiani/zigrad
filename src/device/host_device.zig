//! BLAS ops for host device, CPU or Apple Silicon.
//! Important: strides are assumed to be 1 for many ops now.
//! This assumption is fine until slicing support comes along.
const std = @import("std");
pub const backend = @import("root.zig").backend;
const builtin = @import("builtin");
const BinaryOp = @import("device_common.zig").BinaryOp;
const RandType = @import("device_common.zig").RandType;
const CachingAllocator = @import("caching_allocator.zig").CachingAllocator(HostMalloc);

pub const HostMalloc = struct {
    pub fn raw_alloc(n: usize, _: *anyopaque) ?[*]u8 {
        return @ptrCast(@alignCast(std.c.malloc(n) orelse return null));
    }
    pub fn raw_free(ptr: ?*anyopaque, _: *anyopaque) void {
        return std.c.free(ptr);
    }
};

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
            else => @compileError("Unsupported type for BLAS dot" ++ @typeName(T)),
        }
    }

    pub fn add(
        _: Blas,
        T: type,
        x: []const T,
        y: []const T,
        z: []T,
    ) void {
        for (0..z.len) |i| z[i] = x[i % x.len] + y[i % y.len];
    }

    pub fn sub(
        _: Blas,
        T: type,
        x: []const T,
        y: []const T,
        z: []T,
    ) void {
        for (0..z.len) |i| z[i] = x[i % x.len] - y[i % y.len];
    }

    pub fn mul(
        _: Blas,
        T: type,
        x: []const T,
        y: []const T,
        z: []T,
    ) void {
        for (0..z.len) |i| z[i] = x[i % x.len] * y[i % y.len];
    }

    pub fn div(
        _: Blas,
        T: type,
        x: []const T,
        y: []const T,
        z: []T,
    ) void {
        for (0..z.len) |i| z[i] = x[i % x.len] / y[i % y.len];
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
            else => @compileError("Unsupported type for BLAS matvec" ++ @typeName(T)),
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
            else => @compileError("Unsupported type for BLAS matmul" ++ @typeName(T)),
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
            else => @compileError("Unsupported type for BLAS outer" ++ @typeName(T)),
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
            else => @compileError("Unsupported type for BLAS nrm2" ++ @typeName(T)),
        }
    }

    pub fn clip_norm(
        self: Blas,
        T: type,
        x_val: []T,
        max_norm: T,
        delta: T,
    ) void {
        var scratch: [1]T = undefined;
        self.nrm2(T, x_val, scratch[0..]);
        const norm = scratch[0];
        if (norm > max_norm) {
            const _scale = max_norm / (norm + delta);
            for (x_val) |*value| {
                value.* *= _scale;
            }
        }
    }

    pub fn max_forward(
        _: Blas,
        T: type,
        src: []const T,
        dst: []T,
    ) void {
        const _idx = switch (T) {
            f32 => c.cblas_isamax(@intCast(src.len), src.ptr, 1),
            f64 => c.cblas_idamax(@intCast(src.len), src.ptr, 1),
            else => @compileError("Unsupported type for BLAS max" ++ @typeName(T)),
        };
        dst[0] = src[_idx];
    }

    pub fn max_reverse(
        _: Blas,
        T: type,
        x_val: []const T,
        x_grd: []T,
        y_val: []const T,
        y_grd: []const T,
    ) void {
        const val = y_val[0];
        const grd = y_grd[0];
        for (x_val, x_grd) |x, *g| {
            if (val == x) g.* += grd;
        }
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
            else => @compileError("Unsupported type for BLAS sum" ++ @typeName(T)),
        }
    }

    pub fn sum_along(
        _: Blas,
        T: type,
        src_vals: []const T,
        src_sizes: []const usize,
        dst_vals: []T,
        rdx_idx: usize,
    ) void {
        const sum_dim_size = src_sizes[rdx_idx];

        var slice_size: usize = 1;
        for (rdx_idx + 1..src_sizes.len) |i| {
            slice_size *= src_sizes[i];
        }

        var num_slices: usize = 1;
        for (0..rdx_idx) |i| {
            num_slices *= src_sizes[i];
        }

        for (0..dst_vals.len) |i| {
            var total: T = 0;
            const base_idx = (i / slice_size) * (slice_size * sum_dim_size) + (i % slice_size);
            for (0..sum_dim_size) |j| {
                const curr_idx = base_idx + j * slice_size;
                total += src_vals[curr_idx];
            }
            dst_vals[i] = total;
        }
    }

    pub fn max_along(
        _: Blas,
        T: type,
        src_vals: []const T,
        src_sizes: []const usize,
        dst_vals: []T,
        rdx_idx: usize,
    ) void {
        const max_dim_size = src_sizes[rdx_idx];
        var slice_size: usize = 1;

        for (rdx_idx + 1..src_sizes.len) |i| {
            slice_size *= src_sizes[i];
        }

        for (0..dst_vals.len) |i| {
            var max_val: T = -std.math.inf(T);
            const base_offs = (i / slice_size) * (slice_size * max_dim_size) + (i % slice_size);
            for (0..max_dim_size) |j| { // can be optimized if the view along this dim is contiguous (just check dim stride)
                const curr_offs = base_offs + j * slice_size;
                const curr_val = src_vals[curr_offs];
                if (curr_val > max_val) {
                    max_val = curr_val;
                }
            }
            dst_vals[i] = max_val;
        }
    }

    // copy pasta
    pub fn prod(dims: []const usize) usize {
        if (dims.len == 0) return 0;
        var s: usize = 1;
        for (dims) |f| s *= f;
        return s;
    }

    pub fn bmm_acc(
        self: Blas,
        T: type,
        A: []const T,
        A_sizes: []const usize,
        B: []const T,
        B_sizes: []const usize,
        C: []T,
        C_sizes: []const usize,
        trans_a: bool,
        trans_b: bool,
        lda: usize,
        ldb: usize,
        ldc: usize,
        alpha: T,
        beta: T,
    ) void {
        const n_batches_a = A_sizes[0];
        const n_batches_b = B_sizes[0];
        const n_batches_c = C_sizes[0];
        const A_chunk = A_sizes[1] * A_sizes[2];
        const B_chunk = B_sizes[1] * B_sizes[2];
        const C_chunk = C_sizes[1] * C_sizes[2];

        for (0..n_batches_c) |i| {
            const a_index = i % n_batches_a;
            const b_index = i % n_batches_b;

            const a_start = a_index * A_chunk;
            const b_start = b_index * B_chunk;
            const c_start = i * C_chunk;

            const a_slice = A[a_start .. a_start + A_chunk];
            const b_slice = B[b_start .. b_start + B_chunk];
            const c_slice = C[c_start .. c_start + C_chunk];

            self.matmul(
                T,
                a_slice,
                b_slice,
                c_slice,
                C_sizes[1],
                C_sizes[2],
                A_sizes[2],
                trans_a,
                trans_b,
                lda,
                ldb,
                ldc,
                alpha,
                beta,
            );
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
            else => @compileError("Unsupported type for BLAS scale" ++ @typeName(T)),
        }
    }

    pub fn axpy(
        _: Blas,
        T: type,
        alpha: T,
        x: []const T,
        y: []T,
    ) void {
        switch (T) {
            f32 => c.cblas_saxpy(@intCast(x.len), alpha, x.ptr, 1, y.ptr, 1),
            f64 => c.cblas_daxpy(@intCast(x.len), alpha, x.ptr, 1, y.ptr, 1),
            else => @compileError("Unsupported type for BLAS axpy: " ++ @typeName(T)),
        }
    }

    pub fn reduce(
        self: Blas,
        T: type,
        x_vals: []const T,
        x_dims: []const usize,
        y_vals: []T,
        y_dims: []const usize,
        dim_idxs: []const usize,
        alpha: T,
        beta: T,
        comptime op: BinaryOp,
    ) void {
        _ = self;
        _ = x_vals;
        _ = x_dims;
        _ = y_vals;
        _ = y_dims;
        _ = dim_idxs;
        _ = alpha;
        _ = beta;
        _ = op;
        // TODO
    }
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
    pub fn get(self: *ScratchMemory, T: type, n: usize) []T {
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
    pub fn exp(_: NN, T: type, x: []const T, y: []T) void {
        for (0..x.len) |i| y[i] = @exp(x[i]);
    }

    pub fn relu_forward(_: NN, T: type, x: []const T, y: []T) void {
        for (x, y) |x_v, *y_v| {
            y_v.* = if (x_v > 0) x_v else 0;
        }
    }

    pub fn relu_backward(_: NN, T: type, x_val: []const T, y_grd: []const T, x_grd: []T) void {
        for (x_val, y_grd, x_grd) |x_v, y_g, *x_g| {
            x_g.* += if (x_v > 0) y_g else 0;
        }
    }

    fn parent(self: *const NN) *const HostDevice {
        return @alignCast(@fieldParentPtr("nn", self));
    }
};

pub const HostDevice = struct {
    const Error = CachingAllocator.Error;

    nn: NN,
    blas: Blas,
    scratch: ScratchMemory,
    cache: CachingAllocator,
    allocator: std.mem.Allocator,

    pub fn init(backing_allocator: std.mem.Allocator) HostDevice {
        return .{
            .nn = .{},
            .blas = .{},
            .scratch = .{},
            .cache = CachingAllocator.init(.{}),
            .allocator = backing_allocator,
        };
    }

    pub fn deinit(self: *HostDevice) void {
        self.cache.deinit(undefined);
        self.scratch.deinit();
        self.* = undefined;
    }

    pub fn mem_alloc(self: *HostDevice, T: type, n: usize) ![]T {
        return self.cache.alloc(T, n, undefined);
    }

    pub fn mem_free(self: *HostDevice, slice: anytype) void {
        return self.cache.free(slice, undefined);
    }

    pub fn mem_dupe(self: *HostDevice, T: type, src: []const T) ![]T {
        const dst = try self.cache.alloc(T, src.len, undefined);
        self.mem_copy(T, src, dst);
        return dst;
    }

    pub fn mem_copy(_: HostDevice, T: type, src: []const T, dst: []T) void {
        @memcpy(dst, src);
    }

    pub fn mem_fill(_: HostDevice, T: type, slice: []T, value: T) void {
        @memset(slice, value);
    }

    pub fn mem_random(_: HostDevice, T: type, slice: []T, op: RandType, seed: u64) void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rand = prng.random();
        if (op == .uniform) {
            for (slice) |*e| e.* = rand.float(T);
        } else {
            for (slice) |*e| e.* = rand.floatNorm(T);
        }
    }

    // remove data dependencies on this to speed it up
    pub fn mem_sequence(_: HostDevice, T: type, slice: []T, initial: T, step: T) void {
        var current = initial; // move from register memory
        for (slice) |*x| {
            x.* = current;
            current += step;
        }
    }
    pub fn mem_take(_: HostDevice, T: type, src: []const T, idxs: []const usize, dst: []T) void {
        std.debug.assert(dst.len >= idxs.len);
        for (idxs, 0..) |i, j| dst[j] = src[i];
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
