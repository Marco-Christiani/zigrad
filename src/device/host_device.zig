//! BLAS ops for host device, CPU or Apple Silicon.
//! Important: strides are assumed to be 1 for many ops now.
//! This assumption is fine until slicing support comes along.
//! Some elementwise ops are not numerically stable, check the code.
//! Open an issue/PR if you need stable variants.
const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
pub const using_mkl_rt = build_options.enable_mkl;

const BinaryOp = @import("device_common.zig").BinaryOp;
const DeviceReference = @import("device_reference.zig");
const RandType = @import("device_common.zig").RandType;
const TransferDirection = @import("device_common.zig").TransferDirection;

const ByteMask = std.bit_set.IntegerBitSet(8);
const round_to_next_page = @import("../allocators.zig").round_to_next_page;
const round_to_prev_page = @import("../allocators.zig").round_to_prev_page;
const adjust_map_size = @import("../allocators.zig").adjust_map_size;
const CachingAllocator = @import("../allocators.zig").CachingAllocator(DataHandler);
const DeviceData = @import("../allocators.zig").DeviceData;
const Error = @import("../allocators.zig").Error;
const opspec = @import("opspec.zig");
pub const Options = CachingAllocator.Options;
const zg = @import("../zigrad.zig");

pub const using_mkl_blas: bool = blk: {
    const decls = @typeInfo(c).Struct.decls;
    for (decls) |decl| {
        if (std.mem.startsWith(u8, decl.name, "mkl_") or std.mem.startsWith(u8, decl.name, "MKL_")) {
            break :blk true;
        }
    }
    break :blk false;
};

const c = switch (builtin.target.os.tag) {
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

const DataHandler = struct {
    pub const min_split_size = 64; // bytes

    // zig fmt: off
    pub fn map(self: DataHandler, size: ?usize) ![]u8 {
        // This only promises pages, but does not assign to physical memory.
        // We over subscribe this mapping and only use what is required.
        const adjusted_size = adjust_map_size(size, self.page_size(), host_total_memory());
        
        return @ptrCast(@alignCast(try std.posix.mmap(
            null, adjusted_size,
            std.posix.PROT.READ | std.posix.PROT.WRITE,
            .{ .ANONYMOUS = true, .TYPE = .PRIVATE },
            -1, 0
        )));
    }

    pub fn unmap(_: DataHandler, buf: []u8) void {
        std.posix.munmap(@ptrCast(@alignCast(buf)));
    }

    pub fn alloc(_: DataHandler, n: usize) ?[*]u8 {
        if (comptime builtin.is_test) {
            return std.testing.allocator.rawAlloc(n, @enumFromInt(@alignOf(usize)), @returnAddress());
        }
        return @ptrCast(@alignCast(std.c.malloc(n) orelse return null));
    }

    pub fn free(_: DataHandler, buf: []u8) void {
        if (comptime builtin.is_test) {
            return std.testing.allocator.rawFree(buf, @enumFromInt(@alignOf(usize)), @returnAddress());
        }
        return std.c.free(buf.ptr);
    }

    pub fn page_size(_: DataHandler) usize {
        return std.heap.pageSize();
    }

    pub inline fn deinit(_: DataHandler) void {}
    pub inline fn reset(_: DataHandler) void {}
    // zig fmt: on
};

// TODO Remove this once stdlib support is available
pub const Sysinfo = switch (builtin.abi) {
    .gnux32, .muslx32 => extern struct {
        /// Seconds since boot
        uptime: i64,
        /// 1, 5, and 15 minute load averages
        loads: [3]u64,
        /// Total usable main memory size
        totalram: u64,
        /// Available memory size
        freeram: u64,
        /// Amount of shared memory
        sharedram: u64,
        /// Memory used by buffers
        bufferram: u64,
        /// Total swap space size
        totalswap: u64,
        /// swap space still available
        freeswap: u64,
        /// Number of current processes
        procs: u16,
        /// Explicit padding for m68k
        pad: u16,
        /// Total high memory size
        totalhigh: u64,
        /// Available high memory size
        freehigh: u64,
        /// Memory unit size in bytes
        mem_unit: u32,
    },
    else => extern struct {
        /// Seconds since boot
        uptime: isize,
        /// 1, 5, and 15 minute load averages
        loads: [3]usize,
        /// Total usable main memory size
        totalram: usize,
        /// Available memory size
        freeram: usize,
        /// Amount of shared memory
        sharedram: usize,
        /// Memory used by buffers
        bufferram: usize,
        /// Total swap space size
        totalswap: usize,
        /// swap space still available
        freeswap: usize,
        /// Number of current processes
        procs: u16,
        /// Explicit padding for m68k
        pad: u16,
        /// Total high memory size
        totalhigh: usize,
        /// Available high memory size
        freehigh: usize,
        /// Memory unit size in bytes
        mem_unit: u32,
        /// Pad
        _f: [20 - 2 * @sizeOf(usize) - @sizeOf(u32)]u8,
    },
};

pub fn host_total_memory() usize {
    switch (comptime builtin.target.os.tag) {
        .linux => {
            var si: Sysinfo = undefined;
            if (std.os.linux.syscall1(.sysinfo, @intFromPtr(&si)) != 0)
                @panic("Failed to query sysinfo");
            return si.totalram * si.mem_unit;
        },
        .macos => {
            var total: usize = 0;
            var len: usize = @sizeOf(usize);
            std.posix.sysctlbynameZ("hw.memsize", &total, &len, null, 0) catch |err| {
                std.debug.panic("Failed to get hw.memsize: {}", .{err});
            };
            return total;
        },
        else => @compileError("host_total_memory not implemented for OS"),
    }
}

/////////////////////////////
// Host Device Implementation

const Self = @This();

cache: CachingAllocator,

pub fn init() Self {
    return init_advanced(.{}); // system defaults
}

// TODO: There is probably more to configure than the
// caching allocator - make a unified optoins struct?
pub fn init_advanced(opts: CachingAllocator.Options) Self {
    return .{ .cache = CachingAllocator.init(.{}, opts) };
}

pub fn deinit(self: *Self) void {
    self.cache.deinit();
    self.* = undefined;
}

// callback to replace host reference to union
pub fn reference(self: *Self) DeviceReference {
    return .{ .ptrs = .{ .host = self } };
}

///////////////////
// element wise ops
pub fn add(_: *const Self, T: type, p: opspec.add(T)) void {
    return elwise_binop(T, p.x, p.y, p.z, add_op);
}

pub fn sub(_: *const Self, T: type, p: opspec.sub(T)) void {
    return elwise_binop(T, p.x, p.y, p.z, sub_op);
}

pub fn mul(_: *const Self, T: type, p: opspec.mul(T)) void {
    return elwise_binop(T, p.x, p.y, p.z, mul_op);
}

pub fn div(_: *const Self, T: type, p: opspec.div(T)) void {
    return elwise_binop(T, p.x, p.y, p.z, div_op);
}

/////////////////////////////////
// linear algebra ops

pub fn dot(_: *const Self, T: type, p: opspec.dot(T)) void {
    switch (T) {
        f32 => p.z[0] = c.cblas_sdot(@intCast(p.x.len), p.x.ptr, 1, p.y.ptr, 1),
        f64 => p.z[0] = c.cblas_ddot(@intCast(p.x.len), p.x.ptr, 1, p.y.ptr, 1),
        else => @compileError("Unsupported type for BLAS dot" ++ @typeName(T) ++ " for platform " ++ @tagName(builtin.target.os.tag)),
    }
}

pub fn matvec(_: *const Self, T: type, p: opspec.matvec(T)) void {
    const lda = p.n;
    const ta = if (p.trans_a) c.CblasTrans else c.CblasNoTrans;
    switch (T) {
        f32 => c.cblas_sgemv(c.CblasRowMajor, @intCast(ta), @intCast(p.m), @intCast(p.n), p.alpha, p.A.ptr, @intCast(lda), p.x.ptr, 1, p.beta, p.y.ptr, 1),
        f64 => c.cblas_dgemv(c.CblasRowMajor, @intCast(ta), @intCast(p.m), @intCast(p.n), p.alpha, p.A.ptr, @intCast(lda), p.x.ptr, 1, p.beta, p.y.ptr, 1),
        else => @compileError("Unsupported type for BLAS demv" ++ @typeName(T) ++ " for platform " ++ @tagName(builtin.target.os.tag)),
    }
}

pub fn matmul(_: *const Self, T: type, p: opspec.matmul(T)) void {
    const ta = if (p.trans_a) c.CblasTrans else c.CblasNoTrans;
    const tb = if (p.trans_b) c.CblasTrans else c.CblasNoTrans;
    switch (T) {
        f32 => c.cblas_sgemm(c.CblasRowMajor, @intCast(ta), @intCast(tb), @intCast(p.m), @intCast(p.n), @intCast(p.k), p.alpha, p.A.ptr, @intCast(p.lda), p.B.ptr, @intCast(p.ldb), p.beta, p.C.ptr, @intCast(p.ldc)),
        f64 => c.cblas_dgemm(c.CblasRowMajor, @intCast(ta), @intCast(tb), @intCast(p.m), @intCast(p.n), @intCast(p.k), p.alpha, p.A.ptr, @intCast(p.lda), p.B.ptr, @intCast(p.ldb), p.beta, p.C.ptr, @intCast(p.ldc)),
        else => @compileError("Unsupported type for BLAS gemm" ++ @typeName(T) ++ " for platform " ++ @tagName(builtin.target.os.tag)),
    }
}

pub fn outer(_: *const Self, T: type, p: opspec.outer(T)) void {
    switch (T) {
        f32 => c.cblas_sger(c.CblasRowMajor, @intCast(p.x.len), @intCast(p.y.len), p.alpha, p.x.ptr, 1, p.y.ptr, 1, p.A.ptr, @intCast(p.y.len)),
        f64 => c.cblas_dger(c.CblasRowMajor, @intCast(p.x.len), @intCast(p.y.len), p.alpha, p.x.ptr, 1, p.y.ptr, 1, p.A.ptr, @intCast(p.y.len)),
        else => @compileError("Unsupported type for BLAS ger" ++ @typeName(T) ++ " for platform " ++ @tagName(builtin.target.os.tag)),
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
        else => @compileError("Unsupported type for BLAS axpy" ++ @typeName(T) ++ " for platform " ++ @tagName(builtin.target.os.tag)),
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
        else => @compileError("Unsupported type for BLAS asum" ++ @typeName(T) ++ " for platform " ++ @tagName(builtin.target.os.tag)),
    }
}

pub fn scale(_: *const Self, T: type, p: opspec.scale(T)) void {
    switch (T) {
        f32 => c.cblas_sscal(@intCast(p.x.len), p.alpha, p.x.ptr, 1),
        f64 => c.cblas_dscal(@intCast(p.x.len), p.alpha, p.x.ptr, 1),
        else => @compileError("Unsupported type for BLAS scal" ++ @typeName(T) ++ " for platform " ++ @tagName(builtin.target.os.tag)),
    }
}

pub fn nrm2(_: *const Self, T: type, p: opspec.nrm2(T)) void {
    switch (T) {
        f32 => p.y[0] = c.cblas_snrm2(@intCast(p.x.len), p.x.ptr, 1),
        f64 => p.y[0] = c.cblas_dnrm2(@intCast(p.x.len), p.x.ptr, 1),
        else => @compileError("Unsupported type for BLAS nrm2" ++ @typeName(T) ++ " for platform " ++ @tagName(builtin.target.os.tag)),
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

/// NOTE: Unstable
/// Forward pow implementation $y = x^\text{exp}$
pub fn pow_fwd(_: *const Self, T: type, p: opspec.pow_fwd(T)) void {
    // TODO: Stable impl
    // TODO: Specialized impls
    const exp = p.exp;
    for (p.x, p.y) |x, *y| y.* = std.math.pow(T, x, exp);
}

/// In-place forward pow implementation x = x^exp
pub fn pow_fwd_(_: *const Self, T: type, p: opspec.pow_fwd_(T)) void {
    for (p.x) |*x_val| x_val.* = std.math.pow(T, x_val.*, p.exp);
}

/// Backward pow implementation: x_g += exp * x^(exp-1) * y_g
/// Stable.
pub fn pow_bwd(_: *const Self, T: type, p: opspec.pow_bwd(T)) void {
    // TODO: Reorder. Specialized impls
    const exp = p.exp;
    const expm1 = p.exp - 1;
    const eps = p.eps;
    for (p.x, p.x_g, p.y_g) |x, *x_g, y_g| {
        if (x < eps) {
            x_g.* += 0.0;
        } else {
            x_g.* += exp * std.math.pow(T, x, expm1) * y_g;
        }
    }
}

// Forward square root implementation
pub fn sqrt_fwd(_: *const Self, T: type, p: opspec.sqrt_fwd(T)) void {
    for (p.x, p.y) |x_val, *y_val| {
        y_val.* = std.math.sqrt(x_val);
    }
}

/// In-place forward square root implementation: $x = \sqrt x$
pub fn sqrt_fwd_(_: *const Self, T: type, p: opspec.sqrt_fwd_(T)) void {
    for (p.x) |*x_val| x_val.* = std.math.sqrt(x_val.*);
}

/// Backward square root implementation
/// Stable. Uses subgradient for $\x_i < \epsilon$
pub fn sqrt_bwd(_: *const Self, T: type, p: opspec.sqrt_bwd(T)) void {
    const eps = p.eps;
    for (p.x, p.x_g, p.y_g) |x_val, *x_g_val, y_g_val| {
        // TODO: Reorder. Specialized impls
        if (x_val < eps) {
            // Subgradient
            x_g_val.* += 0.0;
        } else {
            // d/dx(sqrt(x)) = 1/(2*sqrt(x)) = 0.5 / sqrt(x)
            x_g_val.* += 0.5 / std.math.sqrt(x_val) * y_g_val;
        }
    }
}

/// L2 Inverse Sqrt
fn vrsqrt(T: type, a: []const T, inca: usize, r: []T, incr: usize, n: usize, eps: f32) void {
    if (inca == 1 and incr == 1) {
        vrsqrt_contig(T, a, r, eps);
    } else {
        vrsqrt_strided(T, a, inca, r, incr, n, eps);
    }
}

/// L2 Inverse Sqrt for arbitrary strides
fn vrsqrt_strided(T: type, a: []const T, inca: usize, r: []T, incr: usize, n: usize, eps: f32) void {
    switch (T) {
        f32 => if (@hasDecl(c, "vsInvSqrtI"))
            return c.vsInvSqrtI(
                @intCast(n),
                @ptrCast(a.ptr),
                @intCast(inca),
                @ptrCast(r.ptr),
                @intCast(incr),
            ),
        f64 => if (@hasDecl(c, "vdInvSqrtI"))
            return c.vdInvSqrtI(
                @intCast(n),
                @ptrCast(a.ptr),
                @intCast(inca),
                @ptrCast(r.ptr),
                @intCast(incr),
            ),
        else => {},
    }
    vrsqrt_strided_native(T, a, inca, r, incr, eps);
}

/// L2 Inverse Sqrt for contiguous vectors
fn vrsqrt_contig(T: type, a: []const T, r: []T, eps: f32) void {
    switch (T) {
        f32 => if (@hasDecl(c, "vvrsqrtf")) {
            return c.vvrsqrtf(@ptrCast(r.ptr), @ptrCast(a.ptr), @ptrCast(&@as(c_int, @intCast(r.len))));
        } else if (@hasDecl(c, "vsInvSqrt")) {
            return c.vsInvSqrt(@intCast(a.len), @ptrCast(a.ptr), @ptrCast(r.ptr));
        },
        f64 => if (@hasDecl(c, "vvrsqrt")) {
            return c.vvrsqrt(@ptrCast(r.ptr), @ptrCast(a.ptr), @ptrCast(&@as(c_int, @intCast(r.len))));
        } else if (@hasDecl(c, "vdInvSqrt")) {
            return c.vdInvSqrt(@intCast(a.len), @ptrCast(a.ptr), @ptrCast(r.ptr));
        },
        else => {},
    }
    vrsqrt_contig_native(T, a, r, eps);
}

/// Inverse square root native implementation
/// No bounds checking.
fn vrsqrt_strided_native(T: type, a: []const T, inca: usize, r: []T, incr: usize, eps: f32) void {
    var ai: usize = 0;
    var ri: usize = 0;
    while (ai < a.len) : ({
        ai += inca;
        ri += incr;
    }) {
        r[ri] = 1 / @sqrt(@max(a[ai], eps));
    }
}

/// Inverse square root contiguous native implementation
/// No bounds checking.
fn vrsqrt_contig_native(T: type, a: []const T, r: []T, eps: f32) void {
    for (r, a) |*ri, ai| ri.* = 1 / @sqrt(@max(ai, eps));
}

/// Forward inverse square root implementation
pub fn rsqrt_fwd(_: *const Self, T: type, p: opspec.rsqrt_fwd(T)) void {
    vrsqrt(T, p.x, 1, p.y, 1, p.x.len, p.eps);
}

/// In-place backward rsqrt implementation
/// Stable. Uses subgradient for x < eps.
pub fn rsqrt_bwd(
    _: *const Self,
    T: type,
    p: opspec.rsqrt_bwd(T),
) void {
    const eps = p.eps;
    for (p.x, p.x_g, p.y_g) |x_val, *x_g_val, y_g_val| {
        if (x_val < eps) {
            // subgradient: treat as zero
            x_g_val.* += 0.0;
        } else {
            // dy/dx = -0.5 * x^(-1.5)
            const inv_sqrt = std.math.sqrt(x_val); // sqrt(x)
            const inv_x_1p5 = 1.0 / (x_val * inv_sqrt); // x^(-1.5)
            x_g_val.* += (-0.5 * inv_x_1p5) * y_g_val;
        }
    }
}

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
                .y_shape = y_shape.slice(),
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
// reduce (using addition) doesn't have.
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

pub fn mem_cache_alloc(self: *Self, T: type, n: usize) !DeviceData(T) {
    return self.cache.alloc(T, n);
}

pub fn mem_cache_free(self: *Self, data: anytype) void {
    self.cache.free(data);
}

pub fn mem_cache_dupe(self: *Self, T: type, src: []const T) !DeviceData(T) {
    const dst = try self.cache.alloc(T, src.len);
    self.mem_copy(T, src, dst.raw);
    return dst;
}

pub fn mem_alloc(_: *const Self, T: type, n: usize) ![]T {
    if (n == 0) return &.{};

    const ptr = DataHandler.alloc(undefined, n * @sizeOf(T)) orelse
        return Error.DeviceOOM;

    const tptr: [*]T = @ptrCast(@alignCast(ptr));
    return tptr[0..n];
}

pub fn mem_free(_: *const Self, slice: anytype) void {
    if (slice.len == 0) return;
    DataHandler.free(undefined, std.mem.sliceAsBytes(slice));
}

pub fn mem_alloc_byte_mask(self: *Self, n: usize) ![]u8 {
    return self.mem_alloc(u8, @divFloor(n - 1, 8) + 1);
}

pub fn mem_dupe(self: *Self, T: type, src: []const T) ![]T {
    const dst = try self.mem_alloc(T, src.len);
    self.mem_copy(T, src, dst);
    return dst;
}

pub fn mem_scratch(self: *Self, T: type, n: usize) []T {
    return self.cache.alloc_scratch(T, n);
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

pub fn elwise_binop(T: type, x: []const T, y: []const T, z: []T, comptime op: anytype) void {
    if (x.len == 1 or y.len == 1) {
        return _elwise_binop_scalar_dispatch(T, x, y, z, op);
    } else if (x.len == y.len) {
        return _elwise_binop_equal_len(T, x, y, z, op);
    }

    const min_size = @min(x.len, y.len);
    const x_step = if (x.len == z.len) min_size else 0;
    const y_step = if (y.len == z.len) min_size else 0;
    const z_step = min_size;
    var _x, var _y, var _z = .{ x, y, z };
    while (0 < _z.len) {
        _elwise_binop_equal_len(T, _x[0..min_size], _y[0..min_size], _z[0..min_size], op);
        _x = _x[x_step..];
        _y = _y[y_step..];
        _z = _z[z_step..];
    }
}

fn _elwise_binop_scalar_dispatch(T: type, x: []const T, y: []const T, z: []T, comptime op: anytype) void {
    std.debug.assert(x.len == 1 or y.len == 1);
    if (z.len == 1) {
        z[0] = op(x[0], y[0]);
    } else if (x.len == 1) {
        _elwise_binop_scalar_lhs(T, x[0], y, z, op);
    } else {
        _elwise_binop_scalar_rhs(T, x, y[0], z, op);
    }
}

fn _elwise_binop_equal_len(T: type, x: []const T, y: []const T, z: []T, comptime op: anytype) void {
    std.debug.assert(x.len == y.len and y.len == z.len);
    for (0..z.len) |j| z[j] = op(x[j], y[j]);
}

fn _elwise_binop_scalar_lhs(T: type, x: T, y: []const T, z: []T, comptime op: anytype) void {
    std.debug.assert(y.len == z.len);
    for (0..z.len) |j| z[j] = op(x, y[j]);
}

fn _elwise_binop_scalar_rhs(T: type, x: []const T, y: T, z: []T, comptime op: anytype) void {
    std.debug.assert(x.len == z.len);
    for (0..z.len) |j| z[j] = op(x[j], y);
}

/// ## ADR
///
/// Regarding the vectorizable chunks, we dispatch to vadd_contig or similar.
///
/// There are tradeoffs between our options. We can force the correct instructions
/// to be used on Apple platforms using their (SIMD) primitives. If Zig already emits
/// the correct instructions, performance is likely better in Zig as we avoid overhead
/// and give the compiler more visibility.
///
/// For sufficiently long contiguous segments dispatching to vendor kernels would make sense.
/// However, I expect to provide specialized ops for when such a structure is know a-priori.
/// For this op I assuming we dont have large contiguous segments so dispatching isnt necessary.
///
/// In fact, there is an argument to be made for skipping all manual vectorization efforts.
///
/// Benchmarks needed on more platforms. Training benchmarks not showing significant differences
/// except on silicon, on x86 kernel has too much overhead. Analysis will give us a real direction
/// for this op and related ops.
pub fn scatter_add(_: *const Self, T: type, p: opspec.scatter_add(T)) void {
    var dst = p.dst;
    const src = p.src;
    const offs = p.offsets;

    // SIMD segments
    var i: usize = 0;
    const vector_size = comptime std.simd.suggestVectorLength(T) orelse 4;
    // Process vectorizable chunks when offsets allow
    while (i + vector_size <= src.len) {
        // Check if we can vectorize this chunk (consecutive or stride-pattern offsets)
        var can_vectorize = true;
        const base_offset = offs[i];
        for (i + 1..i + vector_size) |j| {
            if (offs[j] != base_offset + (j - i)) {
                can_vectorize = false;
                break;
            }
        }

        // Accumulate
        if (can_vectorize and base_offset + vector_size <= dst.len) {
            // Vectorize - See above notes
            // Dispatch to kernel:
            vadd(T, src[i..][0..vector_size], 1, dst[base_offset..][0..vector_size], 1, dst[base_offset..][0..vector_size], 1, vector_size);
            // Dont yet have an example of forced SIMD primitives

            // Zig manual SIMD:
            // const V = @Vector(vector_size, T);
            // var dst: V = dst[base_offset..][0..vector_size].*;
            // dst += @as(V, src[i..][0..vector_size].*);

            // Zig auto SIMD:
            // for (dst[base_offset..][0..vector_size], src[i..][0..vector_size]) |*di, si| {
            //     di.* += si;
            // }

            i += vector_size;
        } else {
            // Scalar
            dst[offs[i]] += src[i];
            i += 1;
        }
    }
    // Remainder
    while (i < src.len) : (i += 1) {
        dst[offs[i]] += src[i];
    }

    // Greedy exploit arbitrary length contig sequences
    // var i: usize = 0;
    // while (i < src.len) {
    //     var n_contig: usize = 0;
    //     const base_offset = offs[i];
    //     for (i + 1..src.len) |j| {
    //         if (offs[j] != base_offset + (j - i)) {
    //             break;
    //         } else {
    //             n_contig += 1;
    //         }
    //     }
    //     // Accumulate
    //     if (n_contig > 8 and base_offset + n_contig <= dst.len) {
    //         // Vectorize - See above notes
    //         // Dispatch to kernel:
    //         vadd(T, src[i..][0..n_contig], 1, dst[base_offset..][0..n_contig], 1, dst[base_offset..][0..n_contig], 1, n_contig);
    //
    //         // Zig auto SIMD:
    //         // for (dst[base_offset..][0..n_contig], src[i..][0..n_contig]) |*di, si| {
    //         //     di.* += si;
    //         // }
    //
    //         i += n_contig;
    //     } else {
    //         // Scalar
    //         dst[offs[i]] += src[i];
    //         i += 1;
    //     }
    // }

    // No vectorization, baseline for new platforms
    // for (offs, src) |oi, si| {
    //     dst[oi] += si;
    // }
}

/// Scatter add with contiguous subseqences in a strided layout.
/// This is a reduction for each subsequence with scattered accmulation.
///
/// Sums blocks of `stride` contiguous elements from `src` into corresponding blocks in `dst`
/// based on segment assignment in `indices`.
/// Each `src[i*stride:(i+1)*stride]` is added to `dst[indices[i]*stride:(indices[i]+1)*stride]`. I.e.,
///
/// `dst[indices[i], :] += src[i, :]`
///
/// This is equivalent to segment_sum but optimized for cases where
/// data is organized in fixed-size contiguous blocks and elements within a block
/// belong to the same segment
pub fn scatter_add_2d(_: *const Self, T: type, p: opspec.scatter_add_strided(T)) void {
    for (0..p.indices.len) |block_idx| {
        const target_segment = p.indices[block_idx];
        const src_offset = block_idx * p.stride;
        const dst_offset = target_segment * p.stride;

        // TODO: this should be vadd_contig
        vadd(
            T,
            p.src[src_offset .. src_offset + p.stride],
            1,
            p.dst[dst_offset .. dst_offset + p.stride],
            1,
            p.dst[dst_offset .. dst_offset + p.stride],
            1,
            @intCast(p.stride),
        );
    }
}

/// L2 vadd
pub fn vadd(T: type, a: []const T, inca: usize, b: []const T, incb: usize, r: []T, incr: usize, n: usize) void {
    switch (builtin.target.os.tag) {
        .macos => switch (T) {
            f32 => c.vDSP_vadd(
                @ptrCast(a.ptr),
                @intCast(inca),
                @ptrCast(b.ptr),
                @intCast(incb),
                @ptrCast(r.ptr),
                @intCast(incr),
                @intCast(n),
            ),
            f64 => c.vDSP_vaddD(
                @ptrCast(a.ptr),
                @intCast(inca),
                @ptrCast(b.ptr),
                @intCast(incb),
                @ptrCast(r.ptr),
                @intCast(incr),
                @intCast(n),
            ),
            else => @compileError("Unsupported type for vadd" ++ @typeName(T)),
        },
        .linux => if (using_mkl_rt) switch (T) {
            f32 => c.vsAddI(
                @intCast(n),
                @ptrCast(a.ptr),
                @intCast(inca),
                @ptrCast(b.ptr),
                @intCast(incb),
                @ptrCast(r.ptr),
                @intCast(incr),
            ),
            f64 => c.vdAddI(
                @intCast(n),
                @ptrCast(a.ptr),
                @intCast(inca),
                @ptrCast(b.ptr),
                @intCast(incb),
                @ptrCast(r.ptr),
                @intCast(incr),
            ),
            else => @compileError("Unsupported type for vadd" ++ @typeName(T)),
        } else vadd_native(T, a, inca, b, incb, r, incr),
        inline else => @panic("Unsupported os"),
    }
}

/// No bounds checking.
fn vadd_native(T: type, a: []const T, inca: usize, b: []const T, incb: usize, r: []T, incr: usize) void {
    var ai: usize = 0;
    var bi: usize = 0;
    var ri: usize = 0;
    while (ri < r.len) : ({
        ai += inca;
        bi += incb;
        ri += incr;
    }) {
        r[ri] = a[ai] + b[bi];
    }
}

// fn vadd_native_simd(T: type, a: []const T, inca: usize, b: []const T, incb: usize, r: []T, incr: usize, n: usize) type {
//     var i = 0;
//     if (comptime std.simd.suggestVectorLength(T)) |N| {
//         const V = @Vector(N, T);
//         while ((i + N) <= a.len) : (i += N) {
//             const av: V = a[i..][0..N].*;
//             const bv: V = b[i..][0..N].*;
//             r[i..][0..N].* = av + bv;
//         }
//     }
//     while (i < a.len) : (i += 1) {
//         r[i] = a[i] + b[i];
//     }
// }
//

/// Vector Sum
pub fn vsum(T: type, a: []const T, inca: usize) T {
    if (inca == 1) {
        return vsum_contig(T, a);
    } else {
        return vsum_strided(T, a, inca);
    }
}

/// Vector Sum with arbitrary stride
fn vsum_strided(T: type, a: []const T, inca: usize) T {
    switch (T) {
        f32 => if (@hasDecl(c, "vsSumI")) { // VML
            var result: f32 = undefined;
            _ = c.vsSumI(
                @intCast(a.len),
                @ptrCast(a.ptr),
                @intCast(inca),
                @ptrCast(&result),
            );
            return result;
        },
        f64 => if (@hasDecl(c, "vdSumI")) { // VML
            var result: f64 = undefined;
            _ = c.vdSumI(
                @intCast(a.len),
                @ptrCast(a.ptr),
                @intCast(inca),
                @ptrCast(&result),
            );
            return result;
        },
        else => {},
    }
    return vsum_strided_native(T, a, inca);
}

/// Vector Sum for contiguous arrays
fn vsum_contig(T: type, a: []const T) T {
    switch (T) {
        f32 => if (@hasDecl(c, "vvsumf")) { // Accelerate
            var result: f32 = undefined;
            _ = c.vvsumf(
                @ptrCast(&result),
                @ptrCast(a.ptr),
                @ptrCast(&@as(c_int, @intCast(a.len))),
            );
            return result;
        } else if (@hasDecl(c, "vsSum")) { // VML
            var result: f32 = undefined;
            _ = c.vsSum(@intCast(a.len), @ptrCast(a.ptr), @ptrCast(&result));
            return result;
        },
        f64 => if (@hasDecl(c, "vvsum")) { // Accelerate
            var result: f64 = undefined;
            _ = c.vvsum(
                @ptrCast(&result),
                @ptrCast(a.ptr),
                @ptrCast(&@as(c_int, @intCast(a.len))),
            );
            return result;
        } else if (@hasDecl(c, "vdSum")) { // VML
            var result: f64 = undefined;
            _ = c.vdSum(@intCast(a.len), @ptrCast(a.ptr), @ptrCast(&result));
            return result;
        },
        else => {},
    }
    return vsum_contig_native(T, a);
}

/// Native vector sum (strided)
fn vsum_strided_native(T: type, a: []const T, inca: usize) T {
    var s: T = 0;
    var i: usize = 0;
    while (i < a.len) : (i += inca) {
        s += a[i];
    }
    return s;
}

/// Native vector sum (contiguous)
fn vsum_contig_native(T: type, a: []const T) T {
    var s: T = 0;
    for (a) |x| s += x;
    return s;
}

/// See `scatter_add`
pub fn scatter_add_csr(_: *const Self, T: type, p: opspec.scatter_add_csr(T)) void {
    for (0..p.src.len) |i| {
        const start = p.row_ptr[i];
        const end = p.row_ptr[i + 1];
        const value = p.src[i];
        // Broadcast value to segment range
        for (start..end) |j| {
            p.dst[j] += value;
        }
    }
}

pub fn segment_sum_csr(_: *const Self, T: type, p: opspec.segment_sum_csr(T)) void {
    const src = p.src;
    const row_ptr = p.row_offsets;
    const dst = p.dst;

    std.debug.assert(row_ptr.len >= 2);
    std.debug.assert(dst.len + 1 == row_ptr.len);
    std.debug.assert(row_ptr[row_ptr.len - 1] == src.len);

    for (0..dst.len) |i| {
        const start = row_ptr[i];
        const end = row_ptr[i + 1];
        dst[i] = vsum_contig(T, src[start..end]);
    }
}

/// $dst[tgt] += deg[tgt] * deg[src] * h[src]$
/// Degree may be scaled by the user e.g.,
///     $dst[tgt] += deg[tgt]^{-0.5} * deg[src]^{-0.5} * h[src]$
pub fn scatter_gcn_deg_scaled(_: *const Self, T: type, p: opspec.scatter_gcn_deg_scaled(T)) void {
    for (0..p.n_edge) |i| {
        const src = p.src_indices[i];
        const tgt = p.tgt_indices[i];

        const w = p.deg[src] * p.deg[tgt];

        const src_off = src * p.stride;
        const tgt_off = tgt * p.stride;

        for (0..p.stride) |j| {
            p.dst[tgt_off + j] += w * p.h[src_off + j];
        }
    }
}

pub fn scatter_gcn_deg_scaled_bwd(_: *const Self, T: type, p: opspec.scatter_gcn_deg_scaled_bwd(T)) void {
    for (0..p.n_edge) |i| {
        const src = p.src_indices[i];
        const tgt = p.tgt_indices[i];

        const deg_src = p.deg[src];
        const deg_tgt = p.deg[tgt];

        const src_off = src * p.stride;
        const tgt_off = tgt * p.stride;

        var grad_deg_src: T = 0;
        var grad_deg_tgt: T = 0;

        for (0..p.stride) |j| {
            const grad_out = p.grad_output[tgt_off + j];
            const h_val = p.h[src_off + j];

            // dL/dh[src]
            p.grad_h[src_off + j] += grad_out * deg_src * deg_tgt;

            // dL/ddeg[src]
            grad_deg_src += grad_out * deg_tgt * h_val;

            // dL/ddeg[tgt]
            grad_deg_tgt += grad_out * deg_src * h_val;
        }

        p.grad_deg[src] += grad_deg_src;
        p.grad_deg[tgt] += grad_deg_tgt;
    }
}
