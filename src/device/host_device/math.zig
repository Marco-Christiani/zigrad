const std = @import("std");
const opspec = @import("../opspec.zig");
const HostDevice = @import("../host_device.zig");
const ByteMask = HostDevice.ByteMask;
const builtin = @import("builtin");
const build_options = @import("build_options");
const c = @import("blas.zig").c;
pub const using_mkl_blas = @import("blas.zig").using_mkl_blas;
pub const using_mkl_rt = build_options.enable_mkl;

pub fn max_fwd(_: *const HostDevice, T: type, p: opspec.max_fwd(T)) void {
    const idx = switch (T) {
        f32 => c.cblas_isamax(@intCast(p.x.len), p.x.ptr, 1),
        f64 => c.cblas_idamax(@intCast(p.x.len), p.x.ptr, 1),
        else => @compileError("Unsupported type for BLAS max" ++ @typeName(T)),
    };
    p.y[0] = p.x[idx];
}

pub fn max_bwd(_: *const HostDevice, T: type, p: opspec.max_bwd(T)) void {
    const val = p.y[0];
    const grd = p.y_g[0];
    for (p.x, p.x_g) |x, *x_g| {
        if (val == x) x_g.* += grd;
    }
}

pub fn clamp_fwd(_: *const HostDevice, T: type, p: opspec.clamp_fwd(T)) void {
    for (p.x, p.y) |x, *y| y.* = @min(p.max, @max(p.min, x));
}

pub fn clamp_bwd(_: *const HostDevice, T: type, p: opspec.clamp_bwd(T)) void {
    for (p.x, p.x_g, p.y_g) |x, *x_g, y_g| {
        if (p.min < x and x < p.max) x_g.* += y_g;
    }
}

pub fn clamp_mask_fwd(_: *const HostDevice, T: type, p: opspec.clamp_mask_fwd(T)) void {
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

pub fn clamp_mask_bwd(_: *const HostDevice, T: type, p: opspec.clamp_mask_bwd(T)) void {
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

/// NOTE: Unstable
/// Forward pow implementation $y = x^\text{exp}$
pub fn pow_fwd(_: *const HostDevice, T: type, p: opspec.pow_fwd(T)) void {
    // TODO: Stable impl
    // TODO: Specialized impls
    const exp = p.exp;
    for (p.x, p.y) |x, *y| y.* = std.math.pow(T, x, exp);
}

/// In-place forward pow implementation x = x^exp
pub fn pow_fwd_(_: *const HostDevice, T: type, p: opspec.pow_fwd_(T)) void {
    for (p.x) |*x_val| x_val.* = std.math.pow(T, x_val.*, p.exp);
}

/// Backward pow implementation: x_g += exp * x^(exp-1) * y_g
/// Stable.
pub fn pow_bwd(_: *const HostDevice, T: type, p: opspec.pow_bwd(T)) void {
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
pub fn sqrt_fwd(_: *const HostDevice, T: type, p: opspec.sqrt_fwd(T)) void {
    for (p.x, p.y) |x_val, *y_val| {
        y_val.* = std.math.sqrt(x_val);
    }
}

/// In-place forward square root implementation: $x = \sqrt x$
pub fn sqrt_fwd_(_: *const HostDevice, T: type, p: opspec.sqrt_fwd_(T)) void {
    for (p.x) |*x_val| x_val.* = std.math.sqrt(x_val.*);
}

/// Backward square root implementation
/// Stable. Uses subgradient for $\x_i < \epsilon$
pub fn sqrt_bwd(_: *const HostDevice, T: type, p: opspec.sqrt_bwd(T)) void {
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

pub fn accumulate_scaled_delta(_: *const HostDevice, T: type, p: opspec.accumulate_scaled_delta(T)) void {
    for (p.x, p.y, p.z) |x, y, *z| z.* += p.coef * (x - y);
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
pub fn rsqrt_fwd(_: *const HostDevice, T: type, p: opspec.rsqrt_fwd(T)) void {
    vrsqrt(T, p.x, 1, p.y, 1, p.x.len, p.eps);
}

/// In-place backward rsqrt implementation
/// Stable. Uses subgradient for x < eps.
pub fn rsqrt_bwd(_: *const HostDevice, T: type, p: opspec.rsqrt_bwd(T)) void {
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

pub fn exp_fwd(_: *const HostDevice, T: type, p: opspec.exp_fwd(T)) void {
    for (p.x, p.y) |x, *y| y.* = @exp(x);
}

pub fn exp_bwd(_: *const HostDevice, T: type, p: opspec.exp_bwd(T)) void {
    for (p.x_g, p.y, p.y_g) |*x_g, y, y_g| x_g.* += y * y_g;
}

///////////////////////////
//// Binary OPs ///////////

pub fn add(_: *const HostDevice, T: type, p: opspec.add(T)) void {
    return elwise_binop(T, p.x, p.y, p.z, add_op);
}

pub fn sub(_: *const HostDevice, T: type, p: opspec.sub(T)) void {
    return elwise_binop(T, p.x, p.y, p.z, sub_op);
}

pub fn mul(_: *const HostDevice, T: type, p: opspec.mul(T)) void {
    return elwise_binop(T, p.x, p.y, p.z, mul_op);
}

pub fn div(_: *const HostDevice, T: type, p: opspec.div(T)) void {
    return elwise_binop(T, p.x, p.y, p.z, div_op);
}

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
pub fn scatter_add(_: *const HostDevice, T: type, p: opspec.scatter_add(T)) void {
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
pub fn scatter_add_2d(_: *const HostDevice, T: type, p: opspec.scatter_add_strided(T)) void {
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
pub fn scatter_add_csr(_: *const HostDevice, T: type, p: opspec.scatter_add_csr(T)) void {
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

pub fn segment_sum_csr(_: *const HostDevice, T: type, p: opspec.segment_sum_csr(T)) void {
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
pub fn scatter_gcn_deg_scaled(_: *const HostDevice, T: type, p: opspec.scatter_gcn_deg_scaled(T)) void {
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

pub fn scatter_gcn_deg_scaled_bwd(_: *const HostDevice, T: type, p: opspec.scatter_gcn_deg_scaled_bwd(T)) void {
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
