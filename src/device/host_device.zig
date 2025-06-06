//! BLAS ops for host device, CPU or Apple Silicon.
//! Important: strides are assumed to be 1 for many ops now.
//! This assumption is fine until slicing support comes along.
const std = @import("std");
pub const backend = @import("root.zig").backend;
const builtin = @import("builtin");
const BinaryOp = @import("device_common.zig").BinaryOp;
const RandType = @import("device_common.zig").RandType;
const TransferDirection = @import("device_common.zig").TransferDirection;
const ByteMask = std.bit_set.IntegerBitSet(8);
const CachingAllocator = @import("caching_allocator.zig").CachingAllocator(HostMalloc);
const opspec = @import("opspec.zig");

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

pub const HostMalloc = struct {
    pub fn raw_alloc(n: usize, _: *anyopaque) ?[*]u8 {
        return @ptrCast(@alignCast(std.c.malloc(n) orelse return null));
    }
    pub fn raw_free(ptr: ?*anyopaque, _: *anyopaque) void {
        return std.c.free(ptr);
    }
};

/////////////////////////////
// Host Device Implementation

const Self = @This();
const Error = CachingAllocator.Error;
const DeviceReference = @import("device_reference.zig");

cache: CachingAllocator,

pub fn init() Self {
    return .{ .cache = CachingAllocator.init(.{}) };
}

pub fn deinit(self: *Self) void {
    self.cache.deinit(undefined);
    self.* = undefined;
}

// callback to replace host reference to union
pub fn reference(self: *Self) DeviceReference {
    return .{ .ptrs = .{ .host = self } };
}

///////////////////
// element wise ops
pub fn add(_: *const Self, T: type, p: opspec.add(T)) void {
    return elwise(T, p.x, p.y, p.z, add_op);
}

pub fn sub(_: *const Self, T: type, p: opspec.sub(T)) void {
    return elwise(T, p.x, p.y, p.z, sub_op);
}

pub fn mul(_: *const Self, T: type, p: opspec.mul(T)) void {
    return elwise(T, p.x, p.y, p.z, mul_op);
}

pub fn div(_: *const Self, T: type, p: opspec.div(T)) void {
    return elwise(T, p.x, p.y, p.z, div_op);
}

/////////////////////////////////
// linear algebra ops

pub fn dot(_: *const Self, T: type, p: opspec.dot(T)) void {
    switch (T) {
        f32 => p.z[0] = c.cblas_sdot(@intCast(p.x.len), p.x.ptr, 1, p.y.ptr, 1),
        f64 => p.z[0] = c.cblas_ddot(@intCast(p.x.len), p.x.ptr, 1, p.y.ptr, 1),
        else => @compileError("Unsupported type for BLAS dot" ++ @typeName(T)),
    }
}

pub fn matvec(_: *const Self, T: type, p: opspec.matvec(T)) void {
    const lda = p.n;
    const ta = if (p.trans_a) c.CblasTrans else c.CblasNoTrans;
    switch (T) {
        f32 => c.cblas_sgemv(c.CblasRowMajor, @intCast(ta), @intCast(p.m), @intCast(p.n), p.alpha, p.A.ptr, @intCast(lda), p.x.ptr, 1, p.beta, p.y.ptr, 1),
        f64 => c.cblas_dgemv(c.CblasRowMajor, @intCast(ta), @intCast(p.m), @intCast(p.n), p.alpha, p.A.ptr, @intCast(lda), p.x.ptr, 1, p.beta, p.y.ptr, 1),
        else => @compileError("Unsupported type for BLAS matvec" ++ @typeName(T)),
    }
}

pub fn matmul(_: *const Self, T: type, p: opspec.matmul(T)) void {
    const ta = if (p.trans_a) c.CblasTrans else c.CblasNoTrans;
    const tb = if (p.trans_b) c.CblasTrans else c.CblasNoTrans;
    switch (T) {
        f32 => c.cblas_sgemm(c.CblasRowMajor, @intCast(ta), @intCast(tb), @intCast(p.m), @intCast(p.n), @intCast(p.k), p.alpha, p.A.ptr, @intCast(p.lda), p.B.ptr, @intCast(p.ldb), p.beta, p.C.ptr, @intCast(p.ldc)),
        f64 => c.cblas_dgemm(c.CblasRowMajor, @intCast(ta), @intCast(tb), @intCast(p.m), @intCast(p.n), @intCast(p.k), p.alpha, p.A.ptr, @intCast(p.lda), p.B.ptr, @intCast(p.ldb), p.beta, p.C.ptr, @intCast(p.ldc)),
        else => @compileError("Unsupported type for BLAS matmul" ++ @typeName(T)),
    }
}

pub fn outer(_: *const Self, T: type, p: opspec.outer(T)) void {
    switch (T) {
        f32 => c.cblas_sger(c.CblasRowMajor, @intCast(p.x.len), @intCast(p.y.len), p.alpha, p.x.ptr, 1, p.y.ptr, 1, p.A.ptr, @intCast(p.y.len)),
        f64 => c.cblas_dger(c.CblasRowMajor, @intCast(p.x.len), @intCast(p.y.len), p.alpha, p.x.ptr, 1, p.y.ptr, 1, p.A.ptr, @intCast(p.y.len)),
        else => @compileError("Unsupported type for BLAS outer" ++ @typeName(T)),
    }
}

// TODO: extend to greater than 2D and optimize this
pub fn transpose(_: *const Self, T: type, p: opspec.transpose(T)) void {
    for (0..p.m) |i| {
        for (0..p.n) |j| {
            p.B[j * p.m + i] = p.A[i * p.n + j] + p.alpha * p.B[j * p.m + i];
        }
    }
}

pub fn axpy(_: *const Self, T: type, p: opspec.axpy(T)) void {
    switch (T) {
        f32 => c.cblas_saxpy(@intCast(p.x.len), p.alpha.*, p.x.ptr, 1, p.y.ptr, 1),
        f64 => c.cblas_daxpy(@intCast(p.x.len), p.alpha.*, p.x.ptr, 1, p.y.ptr, 1),
        else => @compileError("Unsupported type for BLAS axpy: " ++ @typeName(T)),
    }
}

pub fn bmm_acc(self: *const Self, T: type, p: opspec.bmm_acc(T)) void {
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

pub fn sum(_: *const Self, T: type, p: opspec.sum(T)) void {
    switch (T) {
        f32 => p.y[0] = c.cblas_sasum(@intCast(p.x.len), p.x.ptr, 1),
        f64 => p.y[0] = c.cblas_dasum(@intCast(p.x.len), p.x.ptr, 1),
        else => @compileError("Unsupported type for BLAS sum" ++ @typeName(T)),
    }
}

pub fn scale(_: *const Self, T: type, p: opspec.scale(T)) void {
    switch (T) {
        f32 => c.cblas_sscal(@intCast(p.x.len), p.alpha, p.x.ptr, 1),
        f64 => c.cblas_dscal(@intCast(p.x.len), p.alpha, p.x.ptr, 1),
        else => @compileError("Unsupported type for BLAS scale" ++ @typeName(T)),
    }
}

pub fn nrm2(_: *const Self, T: type, p: opspec.nrm2(T)) void {
    switch (T) {
        f32 => p.y[0] = c.cblas_snrm2(@intCast(p.x.len), p.x.ptr, 1),
        f64 => p.y[0] = c.cblas_dnrm2(@intCast(p.x.len), p.x.ptr, 1),
        else => @compileError("Unsupported type for BLAS nrm2" ++ @typeName(T)),
    }
}

pub fn clip_nrm2(self: *const Self, T: type, p: opspec.clip_nrm2(T)) void {
    var scratch: [1]T = undefined;
    self.nrm2(T, .{ .x = p.x, .y = scratch[0..] });
    const norm = scratch[0];
    if (norm > p.max_norm) {
        self.scale(T, .{ .x = p.x, .alpha = p.max_norm / (norm + p.delta) });
    }
}

/////////////////////////////////
// non-linear ops

pub fn exp_fwd(_: *const Self, T: type, p: opspec.exp_fwd(T)) void {
    var i: usize = 0;
    if (comptime std.simd.suggestVectorLength(T)) |N| {
        const V = @Vector(N, T);
        while ((i + N) <= p.x.len) : (i += N) {
            const u: V = p.x[i..][0..N].*;
            p.y[i..][0..N].* = @exp(u);
        }
    }
    while (i < p.x.len) : (i += 1) {
        p.y[i] = @exp(p.x[i]);
    }
}

pub fn exp_bwd(_: *const Self, T: type, p: opspec.exp_bwd(T)) void {
    // TODO: vectorize this...
    for (p.x_g, p.y, p.y_g) |*x_g, y, y_g| x_g.* += y * y_g;
}

pub fn relu_fwd(_: *const Self, T: type, p: opspec.relu_fwd(T)) void {
    var i: usize = 0;
    if (comptime std.simd.suggestVectorLength(T)) |N| {
        const V = @Vector(N, T);
        const zero: V = @splat(0);
        while ((i + N) <= p.x.len) : (i += N) {
            const u: V = p.x[i..][0..N].*;
            p.y[i..][0..N].* = @max(zero, u);
        }
    }
    while (i < p.x.len) : (i += 1) {
        p.y[i] = @max(0, p.x[i]);
    }
}

pub fn relu_bwd(_: *const Self, T: type, p: opspec.relu_bwd(T)) void {
    for (p.x, p.x_g, p.y_g) |x, *x_g, y_g| x_g.* += if (x > 0) y_g else 0;
}

pub fn relu_inplace_bwd(_: *const Self, T: type, p: opspec.relu_inplace_bwd(T)) void {
    for (p.x, p.x_g) |x, *x_g| {
        if (x <= 0) x_g.* = 0;
    }
}

pub fn tanh_fwd(_: *const Self, T: type, p: opspec.tanh_fwd(T)) void {
    for (p.x, p.y) |x, *y| y.* = std.math.tanh(x);
}

pub fn tanh_bwd(_: *const Self, T: type, p: opspec.tanh_bwd(T)) void {
    for (p.x_g, p.y, p.y_g) |*x_g, y, y_g| x_g.* += (1 - (y * y)) * y_g;
}

pub fn tanh_inplace_bwd(_: *const Self, T: type, p: opspec.tanh_inplace_bwd(T)) void {
    for (p.x, p.x_g) |x, *x_g| x_g.* *= (1 - (x * x));
}

pub fn sigm_fwd(_: *const Self, T: type, p: opspec.sigm_fwd(T)) void {
    for (p.x, p.y) |x, *y| y.* = 1 / (1 + @exp(-x));
}

pub fn sigm_bwd(_: *const Self, T: type, p: opspec.sigm_bwd(T)) void {
    for (p.x_g, p.y, p.y_g) |*x_g, y, y_g| x_g.* += y * (1 - y) * y_g;
}

pub fn sigm_inplace_bwd(_: *const Self, T: type, p: opspec.sigm_inplace_bwd(T)) void {
    for (p.x, p.x_g) |x, *x_g| x_g.* *= (x * (1 - x));
}

pub fn max_fwd(_: *const Self, T: type, p: opspec.max_fwd(T)) void {
    const idx = switch (T) {
        f32 => c.cblas_isamax(@intCast(p.x.len), p.x.ptr, 1),
        f64 => c.cblas_idamax(@intCast(p.x.len), p.x.ptr, 1),
        else => @compileError("Unsupported type for BLAS max" ++ @typeName(T)),
    };
    p.y[0] = p.x[idx];
}

pub fn max_bwd(_: *const Self, T: type, p: opspec.max_bwd(T)) void {
    const val = p.y[0];
    const grd = p.y_g[0];
    for (p.x, p.x_g) |x, *x_g| {
        if (val == x) x_g.* += grd;
    }
}

pub fn clamp_fwd(_: *const Self, T: type, p: opspec.clamp_fwd(T)) void {
    for (p.x, p.y) |x, *y| y.* = @min(p.max, @max(p.min, x));
}

pub fn clamp_bwd(_: *const Self, T: type, p: opspec.clamp_bwd(T)) void {
    for (p.x, p.x_g, p.y_g) |x, *x_g, y_g| {
        if (p.min < x and x < p.max) x_g.* += y_g;
    }
}

pub fn clamp_mask_fwd(_: *const Self, T: type, p: opspec.clamp_mask_fwd(T)) void {
    var vals_idx: usize = 0;
    var byte_idx: usize = 0;
    const chunks: usize = @divFloor(p.x.len, 8);

    for (0..chunks) |_| {
        var bits: ByteMask = .{ .mask = 0 };
        inline for (0..8) |b| {
            const x = p.x[vals_idx];
            const clamped = @min(p.max, @max(p.min, x));
            bits.setValue(b, x != clamped);
            p.y[vals_idx] = x;
            vals_idx += 1;
        }
        p.mask[byte_idx] = @bitCast(bits.mask);
        byte_idx += 1;
    }

    if (vals_idx == p.x.len) return;

    var bits: ByteMask = .{ .mask = 0 };
    for (vals_idx..p.x.len, 0..p.x_g.len - vals_idx) |i, b| {
        const x = p.x[i];
        const clamped = @min(p.max, @max(p.min, x));
        bits.setValue(b, x != clamped);
        p.y[vals_idx] = x;
    }
    p.mask[byte_idx] = @bitCast(bits.mask);
}

pub fn clamp_mask_bwd(_: *const Self, T: type, p: opspec.clamp_mask_bwd(T)) void {
    var vals_idx: usize = 0;
    var byte_idx: usize = 0;
    const chunks: usize = @divFloor(p.x.len, 8);

    for (0..chunks) |_| {
        var bits: ByteMask = .{ .mask = p.mask[byte_idx] };
        inline for (0..8) |b| {
            if (bits.isSet(b)) p.x_g[vals_idx] += p.y_g[vals_idx];
            vals_idx += 1;
        }
        byte_idx += 1;
    }

    if (vals_idx == p.x.len) return;

    var bits: ByteMask = .{ .mask = p.mask[byte_idx] };
    for (vals_idx..p.x.len, 0..p.x_g.len - vals_idx) |i, b| {
        if (bits.isSet(b)) p.x_g[i] += p.y_g[i];
    }
}

pub fn relu_mask_fwd(_: *const Self, T: type, p: opspec.relu_mask_fwd(T)) void {
    var vals_idx: usize = 0;
    var byte_idx: usize = 0;
    const chunks: usize = @divFloor(p.x.len, 8);

    for (0..chunks) |_| {
        var bits: ByteMask = .{ .mask = 0 };
        inline for (0..8) |b| {
            if (p.x[vals_idx] <= 0) {
                p.x[vals_idx] = 0;
                bits.setValue(b, true);
            }
            vals_idx += 1;
        }
        p.mask[byte_idx] = @bitCast(bits.mask);
        byte_idx += 1;
    }
    if (vals_idx == p.x.len) return;

    var bits: ByteMask = .{ .mask = 0 };
    for (vals_idx..p.x.len, 0..p.x.len - vals_idx) |i, b| {
        if (p.x[i] <= 0) {
            p.x[i] = 0;
            bits.setValue(b, true);
        }
    }
    p.mask[byte_idx] = @bitCast(bits.mask);
}

pub fn relu_mask_bwd(_: *const Self, T: type, p: opspec.relu_mask_bwd(T)) void {
    var vals_idx: usize = 0;
    var byte_idx: usize = 0;
    const chunks: usize = @divFloor(p.x_g.len, 8);

    for (0..chunks) |_| {
        var bits: ByteMask = .{ .mask = p.mask[byte_idx] };
        inline for (0..8) |b| {
            if (bits.isSet(b)) p.x_g[vals_idx] = 0;
            vals_idx += 1;
        }
        byte_idx += 1;
    }

    if (vals_idx == p.x_g.len) return;

    var bits: ByteMask = .{ .mask = p.mask[byte_idx] };
    for (vals_idx..p.x_g.len, 0..p.x_g.len - vals_idx) |i, b| {
        if (bits.isSet(b)) p.x_g[i] = 0;
    }
}

fn _prod(sizes: []const usize) usize {
    var n: usize = 1;
    for (sizes) |m| n *= m;
    return n;
}

pub fn unbroadcast(self: *Self, T: type, p: opspec.unbroadcast(T)) void {
    const local = @import("reduce.zig");
    const Array = std.BoundedArray(usize, 8);

    if (p.x.len == p.y.len) {
        return local.scaled_copy(T, .{
            .x = p.x,
            .y = p.y,
            .alpha = p.alpha,
            .beta = p.beta,
        });
    }

    // remove any leading ones because they contribute nothing to the reduction
    var x_shape = blk: {
        const trimmed = std.mem.trimLeft(usize, p.x_shape, &.{1});
        break :blk Array.fromSlice(trimmed) catch unreachable;
    };

    var y_shape = blk: {
        const trimmed = std.mem.trim(usize, p.y_shape, &.{1});
        break :blk Array.fromSlice(trimmed) catch unreachable;
    };

    // check if we have to deal with any body reductions
    var ones = std.mem.count(usize, y_shape.slice(), &.{1});

    if (x_shape.len > y_shape.len) {
        const dif = x_shape.len - y_shape.len;

        local.fold_rows(T, .{
            .x = p.x,
            .y = if (ones == 0) p.y else p.scratch,
            .row = _prod(x_shape.slice()[0..dif]),
            .col = _prod(x_shape.slice()[dif..]),
            .alpha = p.alpha,
            .beta = p.beta,
        });

        if (ones == 0) return;

        // remove the indices we just reduced for next round
        x_shape = Array.fromSlice(x_shape.slice()[dif..]) catch unreachable;
    }

    var i: usize = 0;
    while (ones > 0 and i < x_shape.len) {
        if (y_shape.get(i) == 1 and x_shape.get(i) != 1) {
            // TODO: optimize this by detecting streams of 1's in the y_shape
            // and reduce all of those together in one move.

            const x_data = if (x_shape.len == p.x_shape.len) p.x else p.scratch;
            const y_data = if (ones == 1) p.y else p.scratch;

            self.sum_along(T, .{
                .x = x_data,
                .x_shape = x_shape.slice(),
                .y = y_data,
                .dim = i,
                .alpha = p.alpha,
                .beta = p.beta,
            });

            _ = x_shape.orderedRemove(i);
            _ = y_shape.orderedRemove(i);
            ones -= 1;
            continue;
        }
        i += 1; // only increment if didn't remove an index
    }
}

pub fn broadcast(_: *const Self, T: type, p: opspec.broadcast(T)) void {
    const local = @import("reduce.zig");
    if (p.x.len == p.y.len) {
        local.scaled_copy(T, .{
            .x = p.x,
            .y = p.y,
            .alpha = p.alpha,
            .beta = p.beta,
        });
    }

    // TODO: make a better broadcast
    for (0..p.y.len) |i| p.y[i] = p.alpha * p.x[i % p.x.len] + p.beta * p.y[i % p.y.len];
}

// TODO: Replace this with general reduce
pub fn sum_along(_: *const Self, T: type, p: opspec.sum_along(T)) void {
    std.debug.assert(0 < p.x_shape.len);
    std.debug.assert(p.dim < p.x_shape.len);
    const local = @import("reduce.zig");

    // flat, head, and tail reduce base cases
    if (p.y.len == 1) {
        return local.flat_reduce(T, .{
            .x = p.x,
            .y = p.y,
            .alpha = p.alpha,
            .beta = p.beta,
        });
    } else if (p.dim == 0) {
        return local.fold_rows(T, .{
            .x = p.x,
            .y = p.y,
            .row = p.x_shape[0],
            .col = _prod(p.x_shape[1..]),
            .alpha = p.alpha,
            .beta = p.beta,
        });
    } else if (p.dim + 1 == p.x_shape.len) {
        return local.fold_cols(T, .{
            .x = p.x,
            .y = p.y,
            .row = _prod(p.x_shape[0..p.dim]),
            .col = p.x_shape[p.dim],
            .alpha = p.alpha,
            .beta = p.beta,
        });
    }
    // body reduction - we can always imagine that we have an ijk
    // tensor where j is the value we want to reduce. This works
    // because we require flat and symmetric memory layout.
    const n_chunks = _prod(p.x_shape[0..p.dim]);
    const y_chunk_size = _prod(p.x_shape[p.dim + 1 ..]);
    const x_chunk_size = p.x_shape[p.dim] * y_chunk_size;

    for (0..n_chunks) |n| {
        local.fold_rows(T, .{
            .x = p.x[x_chunk_size * n ..][0..x_chunk_size],
            .y = p.y[y_chunk_size * n ..][0..y_chunk_size],
            .row = p.x_shape[p.dim],
            .col = y_chunk_size,
            .alpha = p.alpha,
            .beta = p.beta,
        });
    }
}

// TODO: Should this ever be mixed with reduce? Seems like a bad idea
// because certain optimizations actually have extra data that general
// reduce (using addiction) doesn't have.
pub fn max_along(_: *const Self, T: type, p: opspec.max_along(T)) void {
    std.debug.assert(p.dim < p.x_shape.len);

    const max_dim_size = p.x_shape[p.dim];

    var slice_size: usize = 1;
    for (p.dim + 1..p.x_shape.len) |i| {
        slice_size *= p.x_shape[i];
    }

    for (0..p.y.len) |i| {
        var max_val: T = -std.math.inf(T);
        const base_offs = (i / slice_size) * (slice_size * max_dim_size) + (i % slice_size);
        for (0..max_dim_size) |j| { // can be optimized if the view along this dim is contiguous (just check dim stride)
            const curr_offs = base_offs + j * slice_size;
            const curr_val = p.x[curr_offs];
            if (curr_val > max_val) {
                max_val = curr_val;
            }
        }
        p.y[i] = max_val;
    }
}

pub fn mem_alloc(self: *Self, T: type, n: usize) ![]T {
    if (n == 0) return &.{};
    return self.cache.alloc(T, n, undefined);
}

pub fn mem_alloc_byte_mask(self: *Self, n: usize) ![]u8 {
    return self.mem_alloc(u8, @divFloor(n - 1, 8) + 1);
}

pub fn mem_free(self: *Self, slice: anytype) void {
    if (slice.len == 0) return;
    return self.cache.free(slice, undefined);
}

pub fn mem_dupe(self: *Self, T: type, src: []const T) ![]T {
    const dst = try self.cache.alloc(T, src.len, undefined);
    self.mem_copy(T, src, dst);
    return dst;
}

pub fn mem_scratch(self: *Self, T: type, n: usize) ![]T {
    return self.cache.get_scratch(T, n, undefined);
}

pub fn mem_copy(_: *const Self, T: type, src: []const T, dst: []T) void {
    @memcpy(dst, src);
}

pub fn mem_transfer(_: *const Self, T: type, src: []const T, dst: []T, _: TransferDirection) void {
    @memcpy(dst, src);
}

pub fn mem_fill(_: *const Self, T: type, slice: []T, value: T) void {
    @memset(slice, value);
}

pub fn mem_random(_: *const Self, T: type, slice: []T, op: RandType, rand: std.Random) void {
    switch (op) {
        .uniform => {
            for (slice) |*e| e.* = rand.float(T);
        },
        .normal => {
            for (slice) |*e| e.* = rand.floatNorm(T);
        },
        .kaiming => |fan_mode| {
            const fan_in: T = @floatFromInt(fan_mode);
            const std_dev = @sqrt(2.0 / fan_in);
            for (slice) |*e| e.* = rand.floatNorm(T) * std_dev;
        },
    }
}

// remove data dependencies on this to speed it up
pub fn mem_sequence(_: *const Self, T: type, slice: []T, initial: T, step: T) void {
    var current = initial; // move from register memory
    for (slice) |*x| {
        x.* = current;
        current += step;
    }
}
pub fn mem_take(_: *const Self, T: type, src: []const T, idxs: []const usize, dst: []T) void {
    std.debug.assert(dst.len >= idxs.len);
    for (idxs, 0..) |i, j| dst[j] = src[i];
}

pub fn clear_cache(self: *Self) void {
    self.cache.clear(undefined);
}

pub fn sync(_: *const Self) void {}

inline fn add_op(x: anytype, y: anytype) @TypeOf(x, y) {
    return x + y;
}
inline fn sub_op(x: anytype, y: anytype) @TypeOf(x, y) {
    return x - y;
}
inline fn mul_op(x: anytype, y: anytype) @TypeOf(x, y) {
    return x * y;
}
inline fn div_op(x: anytype, y: anytype) @TypeOf(x, y) {
    return x / y;
}

pub fn elwise(T: type, x: []const T, y: []const T, z: []T, comptime op: anytype) void {
    if (x.len == 1 or y.len == 1) {
        return _elwise_scalar_dispatch(T, x, y, z, op);
    } else if (x.len == y.len) {
        return _elwise_equal_len(T, x, y, z, op);
    }

    const min_size = @min(x.len, y.len);
    const x_step = if (x.len == z.len) min_size else 0;
    const y_step = if (y.len == z.len) min_size else 0;
    const z_step = min_size;
    var _x, var _y, var _z = .{ x, y, z };
    while (0 < _z.len) {
        _elwise_equal_len(T, _x[0..min_size], _y[0..min_size], _z[0..min_size], op);
        _x = _x[x_step..];
        _y = _y[y_step..];
        _z = _z[z_step..];
    }
}

fn _elwise_scalar_dispatch(T: type, x: []const T, y: []const T, z: []T, comptime op: anytype) void {
    std.debug.assert(x.len == 1 or y.len == 1);
    if (z.len == 1) {
        z[0] = op(x[0], y[0]);
    } else if (x.len == 1) {
        _elwise_scalar_lhs(T, x[0], y, z, op);
    } else {
        _elwise_scalar_rhs(T, x, y[0], z, op);
    }
}

fn _elwise_equal_len(T: type, x: []const T, y: []const T, z: []T, comptime op: anytype) void {
    std.debug.assert(x.len == y.len and y.len == z.len);
    for (0..z.len) |j| z[j] = op(x[j], y[j]);
}

fn _elwise_scalar_lhs(T: type, x: T, y: []const T, z: []T, comptime op: anytype) void {
    std.debug.assert(y.len == z.len);
    for (0..z.len) |j| z[j] = op(x, y[j]);
}

fn _elwise_scalar_rhs(T: type, x: []const T, y: T, z: []T, comptime op: anytype) void {
    std.debug.assert(x.len == z.len);
    for (0..z.len) |j| z[j] = op(x[j], y);
}
