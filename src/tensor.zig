//! Tensor
//!
//! A unified tensor type that plays two roles depending on mode:
//!  - `traced`: used at compile-time, bound to a FunctionBuilder (for program construction),
//!      emit PR equations.
//!  - `device`: runtime tensor backed by a device buffer (for execution).
//!
//! Both modes carry dtype and shape metadata.
const std = @import("std");
const pr = @import("pr/pr.zig");
const backend_mod = @import("backend/root.zig");
const Backend = backend_mod.Backend;

const Tensor = @This();

dtype: pr.DType,
shape: BoundedShape,
mode: Mode,

/// When `true`, the backend may reuse this input buffer for an output.
///
/// Mark abstract tensors as donatable when the compiled function is expected
///  to produce an updated version of the input (e.g. trainable parameters).
/// Non-donatable inputs (e.g. batch data) are borrowed, the caller manages
///  their lifetime.
donatable: bool = false,

pub const max_rank = pr.max_rank;
pub const BoundedShape = pr.BoundedShape;

pub const Mode = union(enum) {
    traced: Traced,
    device: Device,
    abstract: void,
};

pub const Traced = struct {
    id: pr.VarId,
    builder: *pr.FunctionBuilder,
};

pub const Device = struct {
    buffer: Backend.Buffer,
    backend: *Backend,
};

// ============================================================================
// Construction
// ============================================================================

/// Create a traced tensor from a FunctionBuilder VarId.
pub fn from_id(builder: *pr.FunctionBuilder, id: pr.VarId) !Tensor {
    if (@as(usize, @intCast(id)) >= builder.avals.items.len) return error.InvalidVarId;
    const aval = builder.avals.items[@intCast(id)];
    const t = aval.as_tensor() orelse return error.UnsupportedAval;
    return .{
        .dtype = t.dtype,
        .shape = bounded_from_slice(t.shape.dims),
        .mode = .{ .traced = .{ .id = id, .builder = builder } },
    };
}

/// Create a device tensor wrapping an existing backend buffer.
pub fn from_buffer(b: *Backend, buf: Backend.Buffer, dtype: pr.DType, shape: []const i64) Tensor {
    return .{
        .dtype = dtype,
        .shape = bounded_from_slice(shape),
        .mode = .{ .device = .{ .buffer = buf, .backend = b } },
    };
}

/// Create a traced parameter tensor.
pub fn param(builder: *pr.FunctionBuilder, dtype: pr.DType, shape: []const i64) !Tensor {
    const id = try builder.param_tensor(dtype, shape);
    return from_id(builder, id);
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
    return .{ .dtype = dtype, .shape = bounded_from_slice(shape), .mode = .abstract, .donatable = opts.donatable };
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

fn traced_id(self: Tensor) !pr.VarId {
    return switch (self.mode) {
        .traced => |t| t.id,
        .device, .abstract => error.UnsupportedAval,
    };
}

fn emit_traced(builder: *pr.FunctionBuilder, id: pr.VarId) !Tensor {
    return from_id(builder, id);
}

pub fn add(self: Tensor, other: Tensor) !Tensor {
    const b = try self.traced_builder();
    const id = try b.add(try self.traced_id(), try other.traced_id());
    return emit_traced(b, id);
}

pub fn sub(self: Tensor, other: Tensor) !Tensor {
    const b = try self.traced_builder();
    const id = try b.subtract(try self.traced_id(), try other.traced_id());
    return emit_traced(b, id);
}

pub fn mul(self: Tensor, other: Tensor) !Tensor {
    const b = try self.traced_builder();
    const id = try b.multiply(try self.traced_id(), try other.traced_id());
    return emit_traced(b, id);
}

pub fn div(self: Tensor, other: Tensor) !Tensor {
    const b = try self.traced_builder();
    const id = try b.divide(try self.traced_id(), try other.traced_id());
    return emit_traced(b, id);
}

pub fn matmul(self: Tensor, other: Tensor) !Tensor {
    const b = try self.traced_builder();
    const id = try b.dot(try self.traced_id(), try other.traced_id());
    return emit_traced(b, id);
}

pub fn reshape(self: Tensor, new_dims: []const i64) !Tensor {
    const b = try self.traced_builder();
    const a = b.program.allocator();
    const dims_copy = try a.dupe(i64, new_dims);
    const id = try b.emit(.reshape, &.{try self.traced_id()}, &.{.{ .out_shape = dims_copy }});
    return emit_traced(b, id);
}

pub fn broadcast_in_dim(self: Tensor, out_dims: []const i64, broadcast_dimensions: []const i64) !Tensor {
    const b = try self.traced_builder();
    const a = b.program.allocator();
    const out_copy = try a.dupe(i64, out_dims);
    const bd_copy = try a.dupe(i64, broadcast_dimensions);
    const id = try b.emit(.broadcast_in_dim, &.{try self.traced_id()}, &.{
        .{ .out_shape = out_copy },
        .{ .broadcast_dimensions = bd_copy },
    });
    return emit_traced(b, id);
}

pub fn transpose(self: Tensor, permutation: []const i64) !Tensor {
    const b = try self.traced_builder();
    const a = b.program.allocator();
    const perm_copy = try a.dupe(i64, permutation);
    const id = try b.emit(.transpose, &.{try self.traced_id()}, &.{.{ .permutation = perm_copy }});
    return emit_traced(b, id);
}

pub fn exp(self: Tensor) !Tensor {
    const b = try self.traced_builder();
    const id = try b.emit(.exp, &.{try self.traced_id()}, &.{});
    return emit_traced(b, id);
}

pub fn log(self: Tensor) !Tensor {
    const b = try self.traced_builder();
    const id = try b.emit(.log, &.{try self.traced_id()}, &.{});
    return emit_traced(b, id);
}

pub fn reduce_sum(self: Tensor, axes: []const i64) !Tensor {
    const b = try self.traced_builder();
    const a = b.program.allocator();
    const axes_copy = try a.dupe(i64, axes);
    const sid = try b.emit(.reduce_sum, &.{try self.traced_id()}, &.{.{ .reduce_axes = axes_copy }});
    return emit_traced(b, sid);
}

pub fn reduce_max(self: Tensor, axes: []const i64) !Tensor {
    const b = try self.traced_builder();
    const a = b.program.allocator();
    const axes_copy = try a.dupe(i64, axes);
    const rid = try b.emit(.reduce_max, &.{try self.traced_id()}, &.{.{ .reduce_axes = axes_copy }});
    return emit_traced(b, rid);
}

pub fn max(self: Tensor, other: Tensor) !Tensor {
    const b = try self.traced_builder();
    const mid = try b.emit(.maximum, &.{ try self.traced_id(), try other.traced_id() }, &.{});
    return emit_traced(b, mid);
}

pub fn rsqrt(self: Tensor) !Tensor {
    const b = try self.traced_builder();
    const rid = try b.emit(.rsqrt, &.{try self.traced_id()}, &.{});
    return emit_traced(b, rid);
}

pub fn logistic(self: Tensor) !Tensor {
    const b = try self.traced_builder();
    const lid = try b.emit(.logistic, &.{try self.traced_id()}, &.{});
    return emit_traced(b, lid);
}

pub fn convert(self: Tensor, out_dtype: pr.DType) !Tensor {
    const b = try self.traced_builder();
    const cid = try b.emit(.convert, &.{try self.traced_id()}, &.{.{ .out_dtype = out_dtype }});
    return emit_traced(b, cid);
}

pub fn compare(self: Tensor, other: Tensor, params: pr.CompareParams) !Tensor {
    const b = try self.traced_builder();
    const cid = try b.emit(.compare, &.{ try self.traced_id(), try other.traced_id() }, &.{.{ .compare = params }});
    return emit_traced(b, cid);
}

pub fn select(self: Tensor, cond: Tensor, on_false: Tensor) !Tensor {
    const b = try self.traced_builder();
    const sid = try b.emit(.select, &.{ try cond.traced_id(), try self.traced_id(), try on_false.traced_id() }, &.{});
    return emit_traced(b, sid);
}

pub fn slice(self: Tensor, start_indices: []const i64, limit_indices: []const i64, strides: []const i64) !Tensor {
    const b = try self.traced_builder();
    const a = b.program.allocator();
    const start_copy = try a.dupe(i64, start_indices);
    const limit_copy = try a.dupe(i64, limit_indices);
    const stride_copy = try a.dupe(i64, strides);
    const sid = try b.emit(.slice, &.{try self.traced_id()}, &.{.{ .slice = .{
        .start_indices = start_copy,
        .limit_indices = limit_copy,
        .strides = stride_copy,
    } }});
    return emit_traced(b, sid);
}

pub fn concatenate(self: Tensor, others: []const Tensor, axis: i64) !Tensor {
    const b = try self.traced_builder();
    const a = b.program.allocator();
    const inputs = try a.alloc(pr.VarId, others.len + 1);
    inputs[0] = try self.traced_id();
    for (others, 0..) |t, i| {
        inputs[i + 1] = try t.traced_id();
    }
    const cid = try b.emit(.concatenate, inputs, &.{.{ .concat_axis = axis }});
    return emit_traced(b, cid);
}

pub fn dot_general(self: Tensor, other: Tensor, params: pr.DotGeneralParams) !Tensor {
    const b = try self.traced_builder();
    const a = b.program.allocator();
    const lhs_batch = try a.dupe(i64, params.lhs_batch_dims);
    const rhs_batch = try a.dupe(i64, params.rhs_batch_dims);
    const lhs_contract = try a.dupe(i64, params.lhs_contracting_dims);
    const rhs_contract = try a.dupe(i64, params.rhs_contracting_dims);
    const did = try b.emit(.dot_general, &.{ try self.traced_id(), try other.traced_id() }, &.{.{ .dot_general = .{
        .lhs_batch_dims = lhs_batch,
        .rhs_batch_dims = rhs_batch,
        .lhs_contracting_dims = lhs_contract,
        .rhs_contracting_dims = rhs_contract,
    } }});
    return emit_traced(b, did);
}

pub fn gather(self: Tensor, indices: Tensor, params: pr.GatherParams) !Tensor {
    const b = try self.traced_builder();
    const a = b.program.allocator();
    const gparams: pr.GatherParams = .{
        .slice_sizes = try a.dupe(i64, params.slice_sizes),
        .offset_dims = try a.dupe(i64, params.offset_dims),
        .collapsed_slice_dims = try a.dupe(i64, params.collapsed_slice_dims),
        .start_index_map = try a.dupe(i64, params.start_index_map),
        .index_vector_dim = params.index_vector_dim,
    };
    const gid = try b.emit(.gather, &.{ try self.traced_id(), try indices.traced_id() }, &.{.{ .gather = gparams }});
    return emit_traced(b, gid);
}

pub fn gather_rows(self: Tensor, indices: Tensor) !Tensor {
    const b = try self.traced_builder();
    const self_shape = self.shape.const_slice();
    if (self_shape.len != 2) return error.InvalidGatherOperand;
    const indices_shape = indices.shape.const_slice();
    if (indices_shape.len != 1) return error.InvalidGatherIndices;

    const hidden = self_shape[1];
    const a = b.program.allocator();
    const gparams: pr.GatherParams = .{
        .slice_sizes = try a.dupe(i64, &.{ 1, hidden }),
        .offset_dims = try a.dupe(i64, &.{1}),
        .collapsed_slice_dims = try a.dupe(i64, &.{0}),
        .start_index_map = try a.dupe(i64, &.{0}),
        .index_vector_dim = 1,
    };
    const gid = try b.emit(.gather, &.{ try self.traced_id(), try indices.traced_id() }, &.{.{ .gather = gparams }});
    return emit_traced(b, gid);
}

pub fn gather_2d(self: Tensor, indices: Tensor) !Tensor {
    const b = try self.traced_builder();
    const self_shape = self.shape.const_slice();
    if (self_shape.len != 2) return error.InvalidGatherOperand;
    const indices_shape = indices.shape.const_slice();
    if (indices_shape.len != 2) return error.InvalidGatherIndices;
    if (indices_shape[1] != 2) return error.InvalidGatherIndices;

    const a = b.program.allocator();
    const gparams: pr.GatherParams = .{
        .slice_sizes = try a.dupe(i64, &.{ 1, 1 }),
        .offset_dims = try a.dupe(i64, &.{}),
        .collapsed_slice_dims = try a.dupe(i64, &.{ 0, 1 }),
        .start_index_map = try a.dupe(i64, &.{ 0, 1 }),
        .index_vector_dim = 1,
    };
    const gid = try b.emit(.gather, &.{ try self.traced_id(), try indices.traced_id() }, &.{.{ .gather = gparams }});
    return emit_traced(b, gid);
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

/// Get the underlying VarId (traced mode only).
pub fn get_id(self: Tensor) !pr.VarId {
    return self.traced_id();
}

pub fn relu(self: Tensor) !Tensor {
    const b = try self.traced_builder();
    const zero = try scalar_literal(b, zero_literal_value(self.dtype));
    const broadcast = try zero.broadcast_in_dim(self.shape.const_slice(), &.{});
    return self.max(broadcast);
}

// ============================================================================
// Helpers
// ============================================================================

fn scalar_literal(builder: *pr.FunctionBuilder, lit: pr.Literal) !Tensor {
    const id = try builder.literal_scalar(lit);
    return from_id(builder, id);
}

fn zero_literal_value(dtype: pr.DType) pr.Literal {
    return switch (dtype) {
        .f16 => .{ .f16 = 0 },
        .bf16 => .{ .bf16 = 0 },
        .f32 => .{ .f32 = 0.0 },
        .f64 => .{ .f64 = 0.0 },
        .i8 => .{ .i8 = 0 },
        .u8 => .{ .u8 = 0 },
        .i32 => .{ .i32 = 0 },
        .i64 => .{ .i64 = 0 },
        .u32 => .{ .u32 = 0 },
        .u64 => .{ .u64 = 0 },
        .bool => .{ .bool = false },
    };
}

fn bounded_from_slice(s: []const i64) BoundedShape {
    return BoundedShape.from_slice(s);
}
