//! Tensor
//!
//! Unified tensor type with three modes:
//!
//! - **traced**: compile-time. Bound to a `FunctionBuilder`. Each operation
//!   emits a PR op and returns a new traced Tensor. Used during program
//!   construction (tracing).
//! - **device**: runtime. Wraps a backend device buffer. Supports host
//!   transfer (`to_host_sync`, `to_host_async`) and cleanup (`deinit`).
//! - **abstract**: specification only (dtype + shape, no data). Used to
//!   define input specs for `frontend.compile`.
//!
//! All modes carry `dtype` and `shape` as direct fields for uniform access.
//! In traced mode these are copied from the underlying `Var.aval` at
//!  construction time, this is deliberate denormalization so callers don't
//!  need to switch on mode for basic type queries.
const std = @import("std");
const pr = @import("pr/pr.zig");
const backend_mod = @import("backend/root.zig");
const Backend = backend_mod.Backend;

const Tensor = @This();

dtype: pr.DType,
shape: pr.BoundedShape,
mode: Mode,

/// When `true`, the backend may reuse this input buffer for an output.
///
/// Mark abstract tensors as donatable when the compiled function is expected
///  to produce an updated version of the input (e.g. trainable parameters).
/// Non-donatable inputs (e.g. batch data) are borrowed, the caller manages
///  their lifetime.
donatable: bool = false,

pub const max_rank = pr.max_rank;

/// Which execution mode this tensor is in.
pub const Mode = union(enum) {
    /// Compile-time: operations emit PR ops via the builder.
    traced: Traced,
    /// Runtime: wraps a device buffer for execution/transfer.
    device: Device,
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

// ============================================================================
// Construction
// ============================================================================

/// Create a traced tensor from a Var.
pub fn from_var(builder: *pr.FunctionBuilder, v: *pr.Var) Tensor {
    const t = v.as_tensor();
    return .{
        .dtype = t.dtype,
        .shape = .from_slice(t.shape.dims),
        .mode = .{ .traced = .{ .var_ref = v, .builder = builder } },
    };
}

/// Create a device tensor wrapping an existing backend buffer.
pub fn from_buffer(b: *Backend, buf: Backend.Buffer, dtype: pr.DType, shape: []const i64) Tensor {
    return .{
        .dtype = dtype,
        .shape = .from_slice(shape),
        .mode = .{ .device = .{ .buffer = buf, .backend = b } },
    };
}

/// Create a traced parameter tensor.
pub fn param(builder: *pr.FunctionBuilder, dtype: pr.DType, shape: []const i64) !Tensor {
    const v = try builder.param_tensor(dtype, shape);
    return from_var(builder, v);
}

/// Upload host data to a device tensor.
pub fn from_host(b: *Backend, device: Backend.Device, data: []const u8, dtype: pr.DType, shape: []const i64) !Tensor {
    const buf = try b.buffer_from_host(device, data, dtype, shape);
    return from_buffer(b, buf, dtype, shape);
}

pub const AbstractOpts = struct {
    donatable: bool = false,
};

/// Create an abstract tensor for specification purposes (shape/dtype only, no data).
///
/// Use `opts.donatable = true` for inputs the compiled function will update
///  (e.g. trainable parameters). The donation flag flows through compilation
///  into the execute loop, controlling buffer reuse and ownership.
pub fn abstract(dtype: pr.DType, shape: []const i64, opts: AbstractOpts) Tensor {
    return .{ .dtype = dtype, .shape = .from_slice(shape), .mode = .abstract, .donatable = opts.donatable };
}

/// Shorthand for an abstract donatable tensor (trainable parameter spec).
pub fn abstract_donatable(dtype: pr.DType, shape: []const i64) Tensor {
    return abstract(dtype, shape, .{ .donatable = true });
}

// ============================================================================
// Traced-mode operations
// ============================================================================

fn traced_builder(self: Tensor) !*pr.FunctionBuilder {
    return switch (self.mode) {
        .traced => |t| t.builder,
        .device, .abstract => error.UnsupportedAval,
    };
}

fn traced_var(self: Tensor) !*pr.Var {
    return switch (self.mode) {
        .traced => |t| t.var_ref,
        .device, .abstract => error.UnsupportedAval,
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
    const a = b.alloc();
    const operands = try a.alloc(*pr.Var, others.len + 1);
    errdefer a.free(operands);
    operands[0] = try self.traced_var();
    for (others, 0..) |t, i| {
        operands[i + 1] = try t.traced_var();
    }
    const v = try b.concatenate(operands, axis);
    return from_var(b, v);
}

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
// Device-mode operations
// ============================================================================

/// Copy buffer contents to host memory.
///
/// Returns null when the transfer completed synchronously.
pub fn to_host_async(self: Tensor, dst: []u8) !?Backend.Event {
    return switch (self.mode) {
        .device => |d| d.backend.buffer_to_host(d.buffer, dst),
        .traced, .abstract => error.UnsupportedAval,
    };
}

/// Copy buffer to host, blocking until complete.
pub fn to_host_sync(self: Tensor, dst: []u8) !void {
    switch (self.mode) {
        .device => |d| {
            if (try d.backend.buffer_to_host(d.buffer, dst)) |event| {
                try d.backend.await_event(event);
                d.backend.deinit_event(event);
            }
        },
        .traced, .abstract => return error.UnsupportedAval,
    }
}

/// Release the device buffer.
pub fn deinit(self: Tensor) void {
    switch (self.mode) {
        .device => |d| d.backend.deinit_buffer(d.buffer),
        .traced, .abstract => {},
    }
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

/// Get the underlying backend buffer (device mode only).
pub fn buffer(self: Tensor) !Backend.Buffer {
    return switch (self.mode) {
        .device => |d| d.buffer,
        .traced, .abstract => error.UnsupportedAval,
    };
}

/// Get the underlying Var (traced mode only).
pub fn get_var(self: Tensor) !*pr.Var {
    return self.traced_var();
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
