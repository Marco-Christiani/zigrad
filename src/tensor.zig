//! Tensor
//!
//! Unified tensor type with four backing variants:
//!
//! 1. **traced**: compile-time. Bound to a `FunctionBuilder`. Each operation
//!     emits a PR op and returns a new traced Tensor. Used during program
//!     construction (tracing).
//! 2. **device**: runtime. Wraps a backend device buffer. Supports host
//!     transfer and cleanup.
//! 3. **host**: runtime. CPU-resident data with owned, borrowed, or
//!     memory-mapped memory. Supports typed access (`as_slice`, `item`)
//!     and transfer to device (`to_device`).
//! 4. **abstract**: an empty tensor not backed by data, specification only
//!     (dtype + shape, no data). Useful for describing an input spec
//!     without materializing any data. Note that `frontend.trace` reads
//!     only `dtype`/`shape` from each leaf, so it accepts **any** backing
//!     variant -- `abstract` is a convenience, not a requirement.
//!
//! All variants carry `dtype` and `shape` as direct fields for uniform access.
//! In traced mode these are copied from the underlying `Var.aval` at
//!  construction time, this is deliberate denormalization so callers don't
//!  need to switch on backing for basic type queries.
const std = @import("std");
const pr = @import("pr/pr.zig");
const Backend = @import("Backend.zig");
const utils = @import("utils.zig");
const HostBuffer = utils.HostBuffer;

const Tensor = @This();

dtype: pr.DType,
shape: pr.BoundedShape,
backing: Backing,

pub const max_rank = pr.max_rank;

/// What data this tensor is backed by, determined by the lifecycle stage.
pub const Backing = union(enum) {
    /// Compile-time: operations emit PR ops via the builder.
    traced: Traced,
    /// Runtime: wraps a device buffer for execution/transfer.
    device: Device,
    /// Runtime: CPU-resident data (owned, borrowed, or memory-mapped).
    host: HostBuffer,
    /// Specification: dtype + shape only, for defining compile input specs.
    abstract: void,
};

/// Traced-mode payload. Holds the PR SSA value and the builder that owns it.
pub const Traced = struct {
    var_ref: *pr.Var,
    builder: *pr.FunctionBuilder,
};

/// Device-mode payload. Holds the backend buffer and a handle to the backend
/// for transfer/cleanup operations.
pub const Device = struct {
    buffer: Backend.Buffer,
    backend: *Backend,
};

/// Source specification for constructing a host-backed tensor.
pub const HostSrc = union(enum) {
    /// Allocate zeroed host memory.
    alloc: std.mem.Allocator,
    /// Borrow externally-owned bytes (no-op on deinit). Caller must
    ///  ensure the bytes outlive this tensor.
    borrow: []const u8,
    /// Memory-map a file path (read-only, munmap on deinit).
    /// TODO: accept a file handle or std.fs.File for non-path-based mmap.
    mmap: []const u8,
};

// ============================================================================
// Construction
// ============================================================================

/// Create a traced tensor from a Var.
pub fn from_var(builder: *pr.FunctionBuilder, v: *pr.Var) Tensor {
    const t = v.as_tensor();
    return .{
        .dtype = t.dtype,
        .shape = .from_slice(t.shape.dims),
        .backing = .{ .traced = .{ .var_ref = v, .builder = builder } },
    };
}

/// Create a device tensor wrapping an existing backend buffer.
pub fn from_buffer(b: *Backend, buf: Backend.Buffer, dtype: pr.DType, shape: []const i64) Tensor {
    return .{
        .dtype = dtype,
        .shape = .from_slice(shape),
        .backing = .{ .device = .{ .buffer = buf, .backend = b } },
    };
}

/// Create a traced parameter tensor.
pub fn param(builder: *pr.FunctionBuilder, dtype: pr.DType, shape: []const i64) !Tensor {
    const v = try builder.param_tensor(dtype, shape);
    return from_var(builder, v);
}

/// Upload host data to a device tensor.
pub fn from_host_bytes(b: *Backend, device: Backend.Device, data: []const u8, dtype: pr.DType, shape: []const i64) !Tensor {
    const buf = try b.buffer_from_host(device, data, dtype, shape);
    return from_buffer(b, buf, dtype, shape);
}

/// Create a host-backed tensor.
///
/// The `src` parameter selects the memory strategy:
///  - `.alloc`: allocate (aligned) zeroed memory (caller fills via `as_slice`/`fill`).
///  - `.borrow`: wrap existing bytes without copying (caller manages lifetime).
///  - `.mmap`: memory-map a file path (read-only, unmapped on `deinit`).
pub fn host(dtype: pr.DType, shape: []const i64, src: HostSrc) !Tensor {
    const bounded = pr.BoundedShape.from_slice(shape);
    const hb: HostBuffer = switch (src) {
        .alloc => |a| try HostBuffer.init(a, bounded, dtype),
        .borrow => |bytes| blk: {
            const align_req = dtype.size_in_bytes();
            if (align_req > 1 and @intFromPtr(bytes.ptr) % align_req != 0) {
                @panic("Tensor.host: borrowed bytes are not aligned for dtype");
            }
            break :blk HostBuffer.borrow(bytes, bounded, dtype);
        },
        .mmap => |path| try HostBuffer.init_mmap(path, bounded, dtype),
    };
    return .{ .dtype = dtype, .shape = bounded, .backing = .{ .host = hb } };
}

/// Create an abstract tensor for specification purposes (shape/dtype only, no data).
pub fn abstract(dtype: pr.DType, shape: []const i64) Tensor {
    return .{ .dtype = dtype, .shape = .from_slice(shape), .backing = .abstract };
}

// ============================================================================
// Traced-mode operations
// ============================================================================

fn traced_builder(self: Tensor) !*pr.FunctionBuilder {
    return switch (self.backing) {
        .traced => |t| t.builder,
        .device, .host, .abstract => error.UnsupportedAval,
    };
}

fn traced_var(self: Tensor) !*pr.Var {
    return switch (self.backing) {
        .traced => |t| t.var_ref,
        .device, .host, .abstract => error.UnsupportedAval,
    };
}

pub fn add(self: Tensor, other: Tensor) !Tensor {
    const b = try self.traced_builder();
    const v = try b.add(try self.traced_var(), try other.traced_var());
    return from_var(b, v);
}

pub fn sub(self: Tensor, other: Tensor) !Tensor {
    const b = try self.traced_builder();
    const v = try b.subtract(try self.traced_var(), try other.traced_var());
    return from_var(b, v);
}

pub fn mul(self: Tensor, other: Tensor) !Tensor {
    const b = try self.traced_builder();
    const v = try b.multiply(try self.traced_var(), try other.traced_var());
    return from_var(b, v);
}

pub fn div(self: Tensor, other: Tensor) !Tensor {
    const b = try self.traced_builder();
    const v = try b.divide(try self.traced_var(), try other.traced_var());
    return from_var(b, v);
}

pub fn matmul(self: Tensor, other: Tensor) !Tensor {
    const b = try self.traced_builder();
    const v = try b.dot(try self.traced_var(), try other.traced_var());
    return from_var(b, v);
}

pub fn reshape(self: Tensor, new_dims: []const i64) !Tensor {
    const b = try self.traced_builder();
    const v = try b.reshape(try self.traced_var(), new_dims);
    return from_var(b, v);
}

pub fn broadcast_in_dim(self: Tensor, out_dims: []const i64, broadcast_dimensions: []const i64) !Tensor {
    const b = try self.traced_builder();
    const v = try b.broadcast_in_dim(try self.traced_var(), out_dims, broadcast_dimensions);
    return from_var(b, v);
}

pub fn transpose(self: Tensor, permutation: []const i64) !Tensor {
    const b = try self.traced_builder();
    const v = try b.transpose(try self.traced_var(), permutation);
    return from_var(b, v);
}

pub fn exp(self: Tensor) !Tensor {
    const b = try self.traced_builder();
    const v = try b.exp(try self.traced_var());
    return from_var(b, v);
}

pub fn log(self: Tensor) !Tensor {
    const b = try self.traced_builder();
    const v = try b.log(try self.traced_var());
    return from_var(b, v);
}

pub fn reduce_sum(self: Tensor, axes: []const i64) !Tensor {
    const b = try self.traced_builder();
    const v = try b.reduce_sum(try self.traced_var(), axes);
    return from_var(b, v);
}

pub fn reduce_max(self: Tensor, axes: []const i64) !Tensor {
    const b = try self.traced_builder();
    const v = try b.reduce_max(try self.traced_var(), axes);
    return from_var(b, v);
}

pub fn max(self: Tensor, other: Tensor) !Tensor {
    const b = try self.traced_builder();
    const v = try b.maximum(try self.traced_var(), try other.traced_var());
    return from_var(b, v);
}

pub fn rsqrt(self: Tensor) !Tensor {
    const b = try self.traced_builder();
    const v = try b.rsqrt(try self.traced_var());
    return from_var(b, v);
}

pub fn logistic(self: Tensor) !Tensor {
    const b = try self.traced_builder();
    const v = try b.logistic(try self.traced_var());
    return from_var(b, v);
}

pub fn convert(self: Tensor, out_dtype: pr.DType) !Tensor {
    const b = try self.traced_builder();
    const v = try b.convert(try self.traced_var(), out_dtype);
    return from_var(b, v);
}

pub fn compare(self: Tensor, other: Tensor, params: pr.CompareParams) !Tensor {
    const b = try self.traced_builder();
    const v = try b.compare(try self.traced_var(), try other.traced_var(), params);
    return from_var(b, v);
}

pub fn select(self: Tensor, cond: Tensor, on_false: Tensor) !Tensor {
    const b = try self.traced_builder();
    const v = try b.select(try cond.traced_var(), try self.traced_var(), try on_false.traced_var());
    return from_var(b, v);
}

pub fn slice(self: Tensor, start_indices: []const i64, limit_indices: []const i64, strides: []const i64) !Tensor {
    const b = try self.traced_builder();
    const v = try b.slice(try self.traced_var(), .{
        .start_indices = start_indices,
        .limit_indices = limit_indices,
        .strides = strides,
    });
    return from_var(b, v);
}

pub fn concatenate(self: Tensor, others: []const Tensor, axis: i64) !Tensor {
    const b = try self.traced_builder();

    // The `*Var` slice passed to the builder is only read during op
    //  construction (`emit` copies each pointer into its own operand
    //  array), so a stack buffer is sufficient and avoids leaving a dead
    //  allocation in the program arena.
    var stack: [max_concat_operands]*pr.Var = undefined;
    const n = others.len + 1;
    if (n > stack.len) return error.TooManyConcatOperands;
    const operands = stack[0..n];

    operands[0] = try self.traced_var();
    for (others, 0..) |t, i| {
        operands[i + 1] = try t.traced_var();
    }
    const v = try b.concatenate(operands, axis);
    return from_var(b, v);
}

/// Upper bound on the number of operands per `concatenate` call. Current
///  demo workloads top out at 3 (rope / loss gather); 16 is comfortable.
pub const max_concat_operands: usize = 16;

pub fn dot_general(self: Tensor, other: Tensor, params: pr.DotGeneralParams) !Tensor {
    const b = try self.traced_builder();
    const v = try b.dot_general(try self.traced_var(), try other.traced_var(), params);
    return from_var(b, v);
}

pub fn gather(self: Tensor, indices: Tensor, params: pr.GatherParams) !Tensor {
    const b = try self.traced_builder();
    const v = try b.gather(try self.traced_var(), try indices.traced_var(), params);
    return from_var(b, v);
}

/// Gather rows from a 2D tensor by index. `self` must be [N, hidden],
/// `indices` must be [M]. Returns [M, hidden].
pub fn gather_rows(self: Tensor, indices: Tensor) !Tensor {
    const self_shape = self.shape.const_slice();
    if (self_shape.len != 2) return error.InvalidGatherOperand;
    const indices_shape = indices.shape.const_slice();
    if (indices_shape.len != 1) return error.InvalidGatherIndices;

    const hidden = self_shape[1];
    return self.gather(indices, .{
        .slice_sizes = &.{ 1, hidden },
        .offset_dims = &.{1},
        .collapsed_slice_dims = &.{0},
        .start_index_map = &.{0},
        .index_vector_dim = 1,
    });
}

/// Gather individual elements from a 2D tensor by [row, col] pairs.
/// `self` must be [N, M], `indices` must be [K, 2]. Returns [K].
pub fn gather_2d(self: Tensor, indices: Tensor) !Tensor {
    const self_shape = self.shape.const_slice();
    if (self_shape.len != 2) return error.InvalidGatherOperand;
    const indices_shape = indices.shape.const_slice();
    if (indices_shape.len != 2) return error.InvalidGatherIndices;
    if (indices_shape[1] != 2) return error.InvalidGatherIndices;

    return self.gather(indices, .{
        .slice_sizes = &.{ 1, 1 },
        .offset_dims = &.{},
        .collapsed_slice_dims = &.{ 0, 1 },
        .start_index_map = &.{ 0, 1 },
        .index_vector_dim = 1,
    });
}

// ============================================================================
// Host-mode operations
// ============================================================================

/// View host data as a mutable typed slice. Host-backed (heap) only.
///
/// The returned slice points into the tensor's backing memory; no copy is made.
/// Panics if the backing is not mutable (mmap, borrowed) or not host-backed.
pub fn as_slice(self: Tensor, comptime T: type) []T {
    return switch (self.backing) {
        .host => |hb| @alignCast(std.mem.bytesAsSlice(T, switch (hb.backing) {
            .heap => |h| h.data,
            .mmap, .borrowed => @panic("as_slice requires mutable host backing (heap)"),
        })),
        .traced, .device, .abstract => @panic("as_slice requires host backing"),
    };
}

/// View host data as an immutable typed slice. Any host backing variant.
pub fn as_const_slice(self: Tensor, comptime T: type) []const T {
    return switch (self.backing) {
        .host => |hb| @alignCast(std.mem.bytesAsSlice(T, hb.data())),
        .traced, .device, .abstract => @panic("as_const_slice requires host backing"),
    };
}

/// Raw byte slice (read-only). Host-backed only.
pub fn host_data(self: Tensor) []const u8 {
    return switch (self.backing) {
        .host => |hb| hb.data(),
        .traced, .device, .abstract => @panic("host_data requires host backing"),
    };
}

/// Raw mutable byte slice. Host-backed (heap) only.
pub fn host_data_mut(self: Tensor) []u8 {
    return switch (self.backing) {
        .host => |hb| hb.data_mut(),
        .traced, .device, .abstract => @panic("host_data_mut requires host backing"),
    };
}

/// Fill host buffer with a scalar value. Host-backed (heap) only.
pub fn fill(self: Tensor, comptime T: type, value: T) void {
    const count = self.shape.num_elements();
    const typed: []T = @alignCast(std.mem.bytesAsSlice(T, self.host_data_mut()));
    for (typed[0..count]) |*elem| elem.* = value;
}

/// Extract a scalar value, transferring from device if needed, **implies a sync**.
///
/// Works on both host and device tensors. For device tensors, performs a
///  synchronous transfer into a stack buffer.
///
/// When `T` is a float type, decodes from the tensor's dtype via
///  `DType.decode`, e.g. `item(f32)` on a bf16 tensor decodes correctly.
/// For non-float `T`, reinterprets the raw bytes directly (caller must
///  ensure `T` matches the tensor's storage type).
///
/// Does not take ownership of the device buffer.
///
/// TODO: verify that PJRT_Event_Destroy on a non-awaited event is safe
///  per the PJRT spec. Current plugins appear to handle this correctly
///  (buffer_to_host chains behind execution), but the spec guarantee is
///  unconfirmed.
pub fn item(self: Tensor, comptime T: type) !T {
    // Stack buffer for device transfer, fill w a sentinal value.
    var buf: [8]u8 = "\xDE\xAD\xBE\xEF\xEF\xBE\xAD\xDE".*;

    const bytes: []const u8 = switch (self.backing) {
        .host => self.host_data(),
        .device => |d| blk: {
            const nbytes = self.dtype.size_in_bytes();
            if (try d.backend.buffer_to_host(d.buffer, buf[0..nbytes])) |event| {
                try d.backend.await_event(event);
                d.backend.deinit_event(event);
            }
            break :blk buf[0..nbytes];
        },
        .traced, .abstract => return error.UnsupportedAval,
    };
    return switch (self.dtype) {
        inline .f32, .f64, .bf16, .f16, .i32 => |tag| {
            return tag.decode(T, std.mem.bytesToValue(tag.StorageType(), bytes[0..@sizeOf(tag.StorageType())]));
        },
        else => @panic("item: unsupported dtype"),
    };
}

// ============================================================================
// Transfer operations
// ============================================================================

/// Copy device buffer contents to a caller-provided host byte slice.
///
/// Returns null when the transfer completed synchronously.
pub fn to_host_async(self: Tensor, dst: []u8) !?Backend.Event {
    return switch (self.backing) {
        .device => |d| d.backend.buffer_to_host(d.buffer, dst),
        .traced, .host, .abstract => error.UnsupportedAval,
    };
}

/// Copy device buffer to a caller-provided host byte slice, blocking until complete.
pub fn to_host_sync(self: Tensor, dst: []u8) !void {
    switch (self.backing) {
        .device => |d| {
            if (try d.backend.buffer_to_host(d.buffer, dst)) |event| {
                try d.backend.await_event(event);
                d.backend.deinit_event(event);
            }
        },
        .traced, .host, .abstract => return error.UnsupportedAval,
    }
}

/// Transfer a device tensor to a new host-backed tensor, blocking until complete.
///
/// Allocates a new host buffer and copies device data into it. The returned
/// tensor owns the host memory; call `deinit` to free it.
pub fn to_host(self: Tensor, allocator: std.mem.Allocator) !Tensor {
    switch (self.backing) {
        .device => |d| {
            var hb = try HostBuffer.init(allocator, self.shape, self.dtype);
            errdefer hb.deinit();
            if (try d.backend.buffer_to_host(d.buffer, hb.data_mut())) |event| {
                try d.backend.await_event(event);
                d.backend.deinit_event(event);
            }
            return .{ .dtype = self.dtype, .shape = self.shape, .backing = .{ .host = hb } };
        },
        .traced, .host, .abstract => return error.UnsupportedAval,
    }
}

/// Transfer a host tensor to a device, returning a new device-backed tensor.
///
/// The host data is copied to the device; the caller retains ownership
/// of the source host tensor.
pub fn to_device(self: Tensor, b: *Backend, device: Backend.Device) !Tensor {
    return switch (self.backing) {
        .host => |hb| {
            const buf = try b.buffer_from_host(device, hb.data(), self.dtype, self.shape.const_slice());
            return from_buffer(b, buf, self.dtype, self.shape.const_slice());
        },
        .traced, .device, .abstract => error.UnsupportedAval,
    };
}

// ============================================================================
// Lifecycle
// ============================================================================

/// Release resources held by this tensor.
///
/// - **device**: releases the device buffer via the backend.
/// - **host (heap)**: frees the backing allocation.
/// - **host (mmap)**: unmaps the memory region.
/// - **host (borrowed)**, **traced**, **abstract**: no-op.
pub fn deinit(self: *Tensor) void {
    switch (self.backing) {
        .device => |d| d.backend.deinit_buffer(d.buffer),
        .host => |*hb| hb.deinit(),
        .traced, .abstract => {},
    }
    self.* = undefined;
}

// ============================================================================
// Accessors
// ============================================================================

/// Return shape dims as a slice.
pub fn dims(self: *const Tensor) []const i64 {
    return self.shape.const_slice();
}

/// Shape rank.
pub fn rank(self: Tensor) usize {
    return self.shape.len;
}

/// Get the underlying backend buffer (device backing only).
pub fn buffer(self: Tensor) !Backend.Buffer {
    return switch (self.backing) {
        .device => |d| d.buffer,
        .traced, .host, .abstract => error.UnsupportedAval,
    };
}

/// Get the underlying Var (traced backing only).
pub fn get_var(self: Tensor) !*pr.Var {
    return self.traced_var();
}

/// Get the embedded HostBuffer (host backing only).
///
/// Useful for interop with APIs that still accept `HostBuffer` directly
/// (e.g. `Backend.transfer`).
/// TODO: unused, along with Backend.transfer, but .buffer() doesnt support
///  host which is questionable. Need to iron out this API.
pub fn get_host_buffer(self: *const Tensor) *const HostBuffer {
    return switch (self.backing) {
        .host => |*hb| hb,
        .traced, .device, .abstract => @panic("get_host_buffer requires host backing"),
    };
}

/// Create a scalar constant broadcast to match this tensor's dtype and shape.
pub fn constant_like(like: Tensor, val: f64) !Tensor {
    const b = try like.traced_builder();
    const s = from_var(b, try b.scalar(like.dtype, val));
    if (like.rank() == 0) return s;
    return s.broadcast_in_dim(like.shape.const_slice(), &.{});
}

/// Rectified linear unit: max(x, 0). Composite: broadcast scalar zero + max.
pub fn relu(self: Tensor) !Tensor {
    return self.max(try Tensor.constant_like(self, 0));
}
