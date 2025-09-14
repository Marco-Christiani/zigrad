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

pub fn sync(_: *const Self) void {}

////////////////////////////////////////
// Device Memory Functions /////////////

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

////////////////////////////////////////
const blas = @import("host_device/blas.zig");

pub const dot = blas.dot;
pub const axpy = blas.axpy;
pub const outer = blas.outer;

pub const matvec = blas.matvec;
pub const matmul = blas.matmul;
pub const bmm_acc = blas.bmm_acc;
pub const transpose = blas.transpose;

pub const sum = blas.sum;
pub const scale = blas.scale;
pub const nrm2 = blas.nrm2;
pub const clip_nrm2 = blas.clip_nrm2;

////////////////////////////////////////
const math = @import("host_device/math.zig");

pub const add = math.add;
pub const sub = math.sub;
pub const mul = math.mul;
pub const div = math.div;

pub const pow_fwd = math.pow_fwd;
pub const pow_fwd_ = math.pow_fwd_;
pub const pow_bwd = math.pow_bwd;

pub const sqrt_fwd = math.sqrt_fwd;
pub const sqrt_fwd_ = math.sqrt_fwd_;
pub const sqrt_bwd = math.sqrt_bwd;

pub const rsqrt_fwd = math.rsqrt_fwd;
pub const rsqrt_bwd = math.rsqrt_bwd;

pub const exp_fwd = math.exp_fwd;
pub const exp_bwd = math.exp_bwd;

pub const scatter_add = math.scatter_add;
pub const scatter_add_2d = math.scatter_add_2d;
pub const scatter_add_csr = math.scatter_add_scr;
pub const segment_sum_csr = math.segment_sum_csr;
pub const scatter_gcn_deg_scaled = math.scatter_gcn_deg_scaled;
pub const scatter_gcn_deg_scaled_bwd = math.scatter_gcn_deg_scaled_bwd;

pub const clamp_fwd = math.clamp_fwd;
pub const clamp_bwd = math.clamp_bwd;
pub const clamp_mask_fwd = math.clamp_mask_fwd;
pub const clamp_mask_bwd = math.clamp_mask_bwd;

pub const accumulate_scaled_delta = math.accumulate_scaled_delta;

////////////////////////////////////////
const nn = @import("host_device/nn.zig");

pub const relu_fwd = nn.relu_fwd;
pub const relu_bwd = nn.relu_bwd;
pub const relu_inplace_bwd = nn.relu_inplace_bwd;

pub const tanh_fwd = nn.tanh_fwd;
pub const tanh_bwd = nn.tanh_bwd;
pub const tanh_inplace_bwd = nn.tanh_inplace_bwd;

pub const sigm_fwd = nn.sigm_fwd;
pub const sigm_bwd = nn.sigm_bwd;
pub const sigm_inplace_bwd = nn.sigm_inplace_bwd;

pub const softmax_fwd = nn.softmax_fwd;
pub const softmax_bwd = nn.softmax_bwd;

pub const mse_fwd = nn.mse_fwd;
pub const mse_bwd = nn.mse_bwd;

pub const nll_fwd = nn.nll_fwd;
pub const nll_bwd = nn.nll_bwd;

////////////////////////////////////////
const reduce = @import("host_device/reduce.zig");

pub const unbroadcast = reduce.unbroadcast;
pub const broadcast = reduce.broadcast;
pub const sum_along = reduce.sum_along;
pub const max_along = reduce.max_along;
