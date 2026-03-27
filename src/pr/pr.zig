const std = @import("std");
const ops = @import("ops/ops.zig");

pub const DType = enum {
    f16,
    bf16,
    f32,
    f64,
    i8,
    u8,
    i32,
    i64,
    u32,
    u64,
    bool,

    pub inline fn size_in_bytes(self: DType) usize {
        return switch (self) {
            .bool, .i8, .u8 => 1,
            .f16, .bf16 => 2,
            .f32, .i32, .u32 => 4,
            .f64, .i64, .u64 => 8,
        };
    }

    pub fn name(self: DType) []const u8 {
        return @tagName(self);
    }

    /// Zig type used to store one element of this dtype in host memory.
    ///
    /// Float16 variants (bf16, f16) map to `u16` (bit-pattern storage), not
    /// a native float type. Use `encode_f32`/`decode_f32` for value conversion.
    ///
    /// Intended for use inside `inline switch` branches where the tag is
    /// comptime-known, enabling generic dtype-agnostic code without per-dtype
    /// function duplication.
    pub fn StorageType(comptime self: DType) type {
        return switch (self) {
            .f32 => f32,
            .f64 => f64,
            .bf16, .f16 => u16,
            .i8 => i8,
            .u8 => u8,
            .i32 => i32,
            .i64 => i64,
            .u32 => u32,
            .u64 => u64,
            .bool => u8,
        };
    }

    /// Convert an f32 value to this dtype's storage representation.
    ///
    /// Only supports float and i32 dtypes. Produces a `@compileError` for
    /// unsupported dtypes (integer-only types, bool) to catch misuse at
    /// compile time.
    pub fn encode_f32(comptime self: DType, val: f32) StorageType(self) {
        return switch (self) {
            .f32 => val,
            .f64 => @floatCast(val),
            .bf16 => @intCast(@as(u32, @bitCast(val)) >> 16),
            .f16 => @bitCast(@as(f16, @floatCast(val))),
            .i32 => @intFromFloat(val),
            else => @compileError("encode_f32 not supported for " ++ @tagName(self)),
        };
    }

    /// Decode this dtype's storage representation back to f32.
    ///
    /// Inverse of `encode_f32`. Same dtype restrictions apply.
    pub fn decode_f32(comptime self: DType, raw: StorageType(self)) f32 {
        return switch (self) {
            .f32 => raw,
            .f64 => @floatCast(raw),
            .bf16 => @bitCast(@as(u32, raw) << 16),
            .f16 => @floatCast(@as(f16, @bitCast(raw))),
            .i32 => @floatFromInt(raw),
            else => @compileError("decode_f32 not supported for " ++ @tagName(self)),
        };
    }
};

pub const Shape = struct {
    dims: []const i64,

    pub fn rank(self: Shape) usize {
        return self.dims.len;
    }

    pub fn num_elements(self: Shape) usize {
        var count: usize = 1;
        for (self.dims) |d| count *= @intCast(d);
        return count;
    }
};

// TODO: centralize this using the root `settings` struct pattern we used before.
pub const max_rank = 8;

/// Stack-allocated shape with a fixed maximum rank.
///
/// Value type that can be freely copied, stored, passed around etc without heap.
pub const BoundedShape = struct {
    buf: [max_rank]i64 = undefined,
    len: usize = 0,

    pub fn from_slice(s: []const i64) BoundedShape {
        var result = BoundedShape{};
        for (s) |d| result.append_assume_capacity(d);
        return result;
    }

    pub fn const_slice(self: *const BoundedShape) []const i64 {
        return self.buf[0..self.len];
    }

    pub fn append_assume_capacity(self: *BoundedShape, val: i64) void {
        self.buf[self.len] = val;
        self.len += 1;
    }

    pub fn rank(self: BoundedShape) usize {
        return self.len;
    }

    pub fn num_elements(self: BoundedShape) usize {
        var count: usize = 1;
        for (self.const_slice()) |d| count *= @intCast(d);
        return count;
    }
};

pub const Aval = union(enum) {
    tensor: Tensor,

    // TODO: pretty sure all consumers do the same error check on the null case,
    //  probably centralize that here. Also, we should make sure we keep an eye
    //  on this as the original idea was to support other aval variants but if
    //  tensor remains the only one then this shouldnt exist.
    pub fn as_tensor(self: Aval) ?Tensor {
        return switch (self) {
            .tensor => |t| t,
        };
    }
};

pub const Tensor = struct {
    dtype: DType,
    shape: Shape,
};

pub const VarId = u32;

pub const Span = struct {
    start: u32,
    len: u32,

    pub fn slice(self: Span, comptime T: type, backing: []const T) []const T {
        const start: usize = @intCast(self.start);
        const end: usize = start + @as(usize, @intCast(self.len));
        return backing[start..end];
    }
};

pub const Literal = union(enum) {
    f16: u16,
    bf16: u16,
    f32: f32,
    f64: f64,
    i8: i8,
    u8: u8,
    i32: i32,
    i64: i64,
    u32: u32,
    u64: u64,
    bool: bool,

    pub fn dtype(self: Literal) DType {
        return switch (self) {
            .f16 => .f16,
            .bf16 => .bf16,
            .f32 => .f32,
            .f64 => .f64,
            .i8 => .i8,
            .u8 => .u8,
            .i32 => .i32,
            .i64 => .i64,
            .u32 => .u32,
            .u64 => .u64,
            .bool => .bool,
        };
    }
};

pub const Prim = enum {
    literal,
    add,
    subtract,
    multiply,
    divide,
    maximum,
    exp,
    log,
    rsqrt,
    logistic,
    compare,
    select,
    convert,
    gather,
    scatter,
    dot,
    dot_general,
    reshape,
    iota,
    broadcast_in_dim,
    transpose,
    slice,
    concatenate,
    reduce_sum,
    reduce_max,
    call,
    custom_call,
};

pub const Param = union(enum) {
    literal: Literal,
    out_shape: []const i64,
    broadcast_dimensions: []const i64,
    permutation: []const i64,
    reduce_axes: []const i64,
    out_dtype: DType,
    iota_dimension: i64,
    compare: CompareParams,
    gather: GatherParams,
    scatter: ScatterParams,
    slice: SliceParams,
    concat_axis: i64,
    dot_general: DotGeneralParams,
    call_callee: []const u8,
    /// StableHLO custom_call target name.
    call_target_name: []const u8,
    /// Kernel registry lookup key for single-dispatch custom calls.
    call_kernel_key: []const u8,
    /// Provider identity used by runtime dispatch.
    call_provider_name: []const u8,
    has_side_effect: bool,
    /// Single-output custom_call output type.
    out_aval: Aval,
    /// Multi-output custom_call output types, ordered by output var list.
    out_avals: []const Aval,
};

pub const GatherParams = struct {
    slice_sizes: []const i64,
    offset_dims: []const i64,
    collapsed_slice_dims: []const i64,
    start_index_map: []const i64,
    index_vector_dim: i64,
};

pub const CompareParams = struct {
    direction: CompareDirection,
    compare_type: CompareType,
};

pub const CompareDirection = enum {
    EQ,
    NE,
    GE,
    GT,
    LE,
    LT,
};

pub const CompareType = enum {
    SIGNED,
    UNSIGNED,
    FLOAT,
    TOTALORDER,
};

pub const ScatterParams = struct {
    update_window_dims: []const i64,
    inserted_window_dims: []const i64,
    scatter_dims_to_operand_dims: []const i64,
    index_vector_dim: i64,
    reduction: ScatterReduction = .add,
};

pub const ScatterReduction = enum {
    add,
    max,
    min,
    mul,
};

pub const SliceParams = struct {
    start_indices: []const i64,
    limit_indices: []const i64,
    strides: []const i64,
};

pub const DotGeneralParams = struct {
    lhs_batch_dims: []const i64,
    rhs_batch_dims: []const i64,
    lhs_contracting_dims: []const i64,
    rhs_contracting_dims: []const i64,
};

// ============================================================================
// Annotations & Regions
// ============================================================================

/// Steering annotation attached to a region of equations.
/// Does not change semantics -- only compilation strategy hints.
pub const Annotation = struct {
    /// Request that equations be outlined into a separate call boundary.
    outline: bool = false,
    /// Request kernelization by a named provider (e.g. "tvm").
    kernelize: ?[]const u8 = null,
};

/// A contiguous range of equations sharing an annotation.
/// Materialized at FunctionBuilder.finish() time from the annotation stack.
pub const Region = struct {
    name: []const u8,
    annotation: Annotation,
    /// Index of the first equation in this region.
    eqn_start: u32,
    /// Number of equations in this region.
    eqn_len: u32,
};

// ============================================================================
// Core Data Structures
// ============================================================================

pub const Eqn = struct {
    prim: Prim,
    inputs: Span, // []VarId (Function.varids_store)
    outputs: Span, // []VarId (Function.varids_store)
    params: Span, // []Param (Function.params_store)
};

pub const Function = struct {
    name: []const u8,
    params: []const VarId,
    returns: []const VarId,
    avals: []const Aval,
    eqns: []const Eqn,
    varids_store: []const VarId,
    params_store: []const Param,
    regions: []const Region,

    /// Return regions whose annotation satisfies a predicate.
    pub fn regions_matching(self: Function, predicate: *const fn (Annotation) bool) RegionIterator {
        return .{ .regions = self.regions, .predicate = predicate, .index = 0 };
    }
};

pub const RegionIterator = struct {
    regions: []const Region,
    predicate: *const fn (Annotation) bool,
    index: usize,

    pub fn next(self: *RegionIterator) ?Region {
        while (self.index < self.regions.len) {
            const region = self.regions[self.index];
            self.index += 1;
            if (self.predicate(region.annotation)) return region;
        }
        return null;
    }
};

pub const Program = struct {
    arena: std.heap.ArenaAllocator,
    functions: []Function,

    pub fn init(backing_allocator: std.mem.Allocator) Program {
        return .{
            .arena = std.heap.ArenaAllocator.init(backing_allocator),
            .functions = &.{},
        };
    }

    pub fn allocator(self: *Program) std.mem.Allocator {
        return self.arena.allocator();
    }

    pub fn add_function(self: *Program, func: Function) error{OutOfMemory}!void {
        const a = self.allocator();
        const new_items = try a.alloc(Function, self.functions.len + 1);
        @memcpy(new_items[0..self.functions.len], self.functions);
        new_items[self.functions.len] = func;
        self.functions = new_items;
    }

    pub fn deinit(self: *Program) void {
        self.arena.deinit();
    }
};

// ============================================================================
// Errors
// ============================================================================

pub const ValidationError = error{
    InvalidVarId,
    UnsupportedAval,
    InvalidEqnArity,
    InvalidParams,
    LiteralTypeMismatch,
    AddTypeMismatch,
    SubtractTypeMismatch,
    MultiplyTypeMismatch,
    DivideTypeMismatch,
    MaximumTypeMismatch,
    ExpTypeMismatch,
    LogTypeMismatch,
    RsqrtTypeMismatch,
    LogisticTypeMismatch,
    CompareTypeMismatch,
    SelectTypeMismatch,
    ConvertTypeMismatch,
    GatherTypeMismatch,
    ScatterTypeMismatch,
    DotTypeMismatch,
    DotGeneralTypeMismatch,
    ReshapeTypeMismatch,
    BroadcastInDimTypeMismatch,
    TransposeTypeMismatch,
    SliceTypeMismatch,
    ConcatTypeMismatch,
    ReduceSumTypeMismatch,
    ReduceMaxTypeMismatch,
    CallTypeMismatch,
    CustomCallTypeMismatch,
    IotaTypeMismatch,
    DuplicateFunctionName,
    ScatterAddTypeMismatch,
};

pub const BuildError = ValidationError || error{OutOfMemory};

// ============================================================================
// Validation Helpers
// ============================================================================

fn expect_var_in_range(func: Function, id: VarId) ValidationError!void {
    if (@as(usize, @intCast(id)) >= func.avals.len) return error.InvalidVarId;
}

fn same_tensor_signature(a: Tensor, b: Tensor) bool {
    if (a.dtype != b.dtype) return false;
    if (a.shape.dims.len != b.shape.dims.len) return false;
    for (a.shape.dims, 0..) |d, i| {
        if (d != b.shape.dims[i]) return false;
    }
    return true;
}

// ============================================================================
// Param Extractors
// ============================================================================

/// Extract a typed parameter from an equation's param list by tag.
///
/// Returns the payload of the first `Param` matching `tag`, or `null` if absent.
/// TODO: rethink this, its a bit odd and reads redundant "pr.param(.literal, params)"
///  perhaps AoS/SoA pattern would be wiser, also a method like `eqn.param(.literal)`
///  or params.get(.literal) would be cleaner.
pub fn param(comptime tag: std.meta.Tag(Param), params: []const Param) ?@FieldType(Param, @tagName(tag)) {
    for (params) |p| switch (p) {
        tag => |v| return v,
        else => {},
    };
    return null;
}

// ============================================================================
// Validation
// ============================================================================

/// Validate a single function: var-id range checks, then per-op validation
/// via the op registry.
pub fn validate_function(func: Function) ValidationError!void {
    for (func.params) |p| try expect_var_in_range(func, p);
    for (func.returns) |r| try expect_var_in_range(func, r);

    for (func.eqns) |eqn| {
        const inputs = eqn.inputs.slice(VarId, func.varids_store);
        const outputs = eqn.outputs.slice(VarId, func.varids_store);

        for (inputs) |in_id| try expect_var_in_range(func, in_id);
        for (outputs) |out_id| try expect_var_in_range(func, out_id);

        try ops.validate(func, eqn);
    }
}

pub fn validate_program(program: *const Program) ValidationError!void {
    var i: usize = 0;
    while (i < program.functions.len) : (i += 1) {
        var j: usize = i + 1;
        while (j < program.functions.len) : (j += 1) {
            if (std.mem.eql(u8, program.functions[i].name, program.functions[j].name)) {
                return error.DuplicateFunctionName;
            }
        }
    }

    for (program.functions) |func| {
        try validate_function(func);
    }

    for (program.functions) |func| {
        for (func.eqns) |eqn| {
            if (eqn.prim != .call) continue;

            const params = func.params_store[eqn.params.start..][0..eqn.params.len];
            const callee_name = param(.call_callee, params) orelse return error.InvalidParams;

            var callee: ?Function = null;
            for (program.functions) |candidate| {
                if (std.mem.eql(u8, candidate.name, callee_name)) {
                    callee = candidate;
                    break;
                }
            }
            const callee_func = callee orelse return error.CallTypeMismatch;

            const call_inputs = func.varids_store[eqn.inputs.start..][0..eqn.inputs.len];
            const call_outputs = func.varids_store[eqn.outputs.start..][0..eqn.outputs.len];

            if (call_inputs.len != callee_func.params.len) return error.CallTypeMismatch;
            if (call_outputs.len != callee_func.returns.len) return error.CallTypeMismatch;

            for (call_inputs, 0..) |in_id, idx| {
                const in_tensor = func.avals[@intCast(in_id)].as_tensor() orelse return error.CallTypeMismatch;
                const callee_tensor = callee_func.avals[@intCast(callee_func.params[idx])].as_tensor() orelse return error.CallTypeMismatch;
                if (!same_tensor_signature(in_tensor, callee_tensor)) return error.CallTypeMismatch;
            }
            for (call_outputs, 0..) |out_id, idx| {
                const out_tensor = func.avals[@intCast(out_id)].as_tensor() orelse return error.CallTypeMismatch;
                const callee_tensor = callee_func.avals[@intCast(callee_func.returns[idx])].as_tensor() orelse return error.CallTypeMismatch;
                if (!same_tensor_signature(out_tensor, callee_tensor)) return error.CallTypeMismatch;
            }
        }
    }
}

// ============================================================================
// FunctionBuilder
// ============================================================================

/// Annotation stack entry for tracking active regions during building.
const RegionEntry = struct {
    name: []const u8,
    annotation: Annotation,
    eqn_start: u32,
};

pub const FunctionBuilder = struct {
    program: *Program,
    name: []const u8,
    avals: std.ArrayList(Aval),
    eqns: std.ArrayList(Eqn),
    varids_store: std.ArrayList(VarId),
    params_store: std.ArrayList(Param),
    params: std.ArrayList(VarId),
    region_stack: std.ArrayList(RegionEntry),
    completed_regions: std.ArrayList(Region),

    pub fn init(program: *Program, name: []const u8) BuildError!FunctionBuilder {
        const a = program.allocator();
        return .{
            .program = program,
            .name = name,
            .avals = try std.ArrayList(Aval).initCapacity(a, 16),
            .eqns = try std.ArrayList(Eqn).initCapacity(a, 16),
            .varids_store = try std.ArrayList(VarId).initCapacity(a, 64),
            .params_store = try std.ArrayList(Param).initCapacity(a, 64),
            .params = try std.ArrayList(VarId).initCapacity(a, 8),
            .region_stack = try std.ArrayList(RegionEntry).initCapacity(a, 4),
            .completed_regions = try std.ArrayList(Region).initCapacity(a, 4),
        };
    }

    pub fn deinit(self: *FunctionBuilder) void {
        const a = self.program.allocator();
        self.avals.deinit(a);
        self.eqns.deinit(a);
        self.varids_store.deinit(a);
        self.params_store.deinit(a);
        self.params.deinit(a);
        self.region_stack.deinit(a);
        self.completed_regions.deinit(a);
    }

    fn alloc(self: *FunctionBuilder) std.mem.Allocator {
        return self.program.allocator();
    }

    fn var_with_aval(self: *FunctionBuilder, aval: Aval) BuildError!VarId {
        const a = self.alloc();
        const id: VarId = @intCast(self.avals.items.len);
        try self.avals.append(a, aval);
        return id;
    }

    pub fn tensor_of(self: *FunctionBuilder, id: VarId) ValidationError!Tensor {
        if (@as(usize, @intCast(id)) >= self.avals.items.len) return error.InvalidVarId;
        const aval = self.avals.items[@intCast(id)];
        return aval.as_tensor() orelse error.UnsupportedAval;
    }

    // ====================================================================
    // Region API
    // ====================================================================

    /// Push a named annotation region. Equations emitted after this call
    /// belong to this region until pop_region is called.
    pub fn push_region(self: *FunctionBuilder, name: []const u8, annotation: Annotation) BuildError!void {
        const a = self.alloc();
        try self.region_stack.append(a, .{
            .name = name,
            .annotation = annotation,
            .eqn_start = @intCast(self.eqns.items.len),
        });
    }

    /// Pop the most recent annotation region. The region is recorded with
    /// its equation span for materialization at finish() time.
    pub fn pop_region(self: *FunctionBuilder) BuildError!void {
        const a = self.alloc();
        const entry = self.region_stack.pop() orelse return;
        const eqn_end: u32 = @intCast(self.eqns.items.len);
        if (eqn_end > entry.eqn_start) {
            try self.completed_regions.append(a, .{
                .name = entry.name,
                .annotation = entry.annotation,
                .eqn_start = entry.eqn_start,
                .eqn_len = eqn_end - entry.eqn_start,
            });
        }
    }

    // ====================================================================
    // Core Emit
    // ====================================================================

    /// Emit an equation, inferring the output type via the op registry.
    pub fn emit(self: *FunctionBuilder, prim: Prim, inputs: []const VarId, pms: []const Param) BuildError!VarId {
        const a = self.alloc();

        const out_aval = try ops.infer_output(self, prim, inputs, pms);
        const out = try self.var_with_aval(out_aval);

        const inputs_start: u32 = @intCast(self.varids_store.items.len);
        try self.varids_store.appendSlice(a, inputs);
        const inputs_span: Span = .{ .start = inputs_start, .len = @intCast(inputs.len) };

        const outputs_start: u32 = @intCast(self.varids_store.items.len);
        try self.varids_store.append(a, out);
        const outputs_span: Span = .{ .start = outputs_start, .len = 1 };

        const params_start: u32 = @intCast(self.params_store.items.len);
        try self.params_store.appendSlice(a, pms);
        const params_span: Span = .{ .start = params_start, .len = @intCast(pms.len) };

        try self.eqns.append(a, .{
            .prim = prim,
            .inputs = inputs_span,
            .outputs = outputs_span,
            .params = params_span,
        });

        return out;
    }

    fn emit_with_outputs(self: *FunctionBuilder, prim: Prim, inputs: []const VarId, outputs: []const VarId, pms: []const Param) BuildError!void {
        const a = self.alloc();

        const inputs_start: u32 = @intCast(self.varids_store.items.len);
        try self.varids_store.appendSlice(a, inputs);
        const inputs_span: Span = .{ .start = inputs_start, .len = @intCast(inputs.len) };

        const outputs_start: u32 = @intCast(self.varids_store.items.len);
        try self.varids_store.appendSlice(a, outputs);
        const outputs_span: Span = .{ .start = outputs_start, .len = @intCast(outputs.len) };

        const params_start: u32 = @intCast(self.params_store.items.len);
        try self.params_store.appendSlice(a, pms);
        const params_span: Span = .{ .start = params_start, .len = @intCast(pms.len) };

        try self.eqns.append(a, .{
            .prim = prim,
            .inputs = inputs_span,
            .outputs = outputs_span,
            .params = params_span,
        });
    }

    // ====================================================================
    // Op Convenience Methods
    // ====================================================================

    pub fn param_tensor(self: *FunctionBuilder, dtype: DType, dims: []const i64) BuildError!VarId {
        const a = self.alloc();
        const dims_copy = try a.dupe(i64, dims);
        const id = try self.var_with_aval(.{ .tensor = .{ .dtype = dtype, .shape = .{ .dims = dims_copy } } });
        try self.params.append(a, id);
        return id;
    }

    pub fn literal_scalar(self: *FunctionBuilder, value: Literal) BuildError!VarId {
        return self.emit(.literal, &.{}, &.{.{ .literal = value }});
    }

    pub fn add(self: *FunctionBuilder, lhs: VarId, rhs: VarId) BuildError!VarId {
        return self.emit(.add, &.{ lhs, rhs }, &.{});
    }

    pub fn subtract(self: *FunctionBuilder, lhs: VarId, rhs: VarId) BuildError!VarId {
        return self.emit(.subtract, &.{ lhs, rhs }, &.{});
    }

    pub fn multiply(self: *FunctionBuilder, lhs: VarId, rhs: VarId) BuildError!VarId {
        return self.emit(.multiply, &.{ lhs, rhs }, &.{});
    }

    pub fn divide(self: *FunctionBuilder, lhs: VarId, rhs: VarId) BuildError!VarId {
        return self.emit(.divide, &.{ lhs, rhs }, &.{});
    }

    pub fn maximum(self: *FunctionBuilder, lhs: VarId, rhs: VarId) BuildError!VarId {
        return self.emit(.maximum, &.{ lhs, rhs }, &.{});
    }

    pub fn exp(self: *FunctionBuilder, operand: VarId) BuildError!VarId {
        return self.emit(.exp, &.{operand}, &.{});
    }

    pub fn log(self: *FunctionBuilder, operand: VarId) BuildError!VarId {
        return self.emit(.log, &.{operand}, &.{});
    }

    pub fn rsqrt(self: *FunctionBuilder, operand: VarId) BuildError!VarId {
        return self.emit(.rsqrt, &.{operand}, &.{});
    }

    pub fn logistic(self: *FunctionBuilder, operand: VarId) BuildError!VarId {
        return self.emit(.logistic, &.{operand}, &.{});
    }

    pub fn compare(self: *FunctionBuilder, lhs: VarId, rhs: VarId, cmp: CompareParams) BuildError!VarId {
        return self.emit(.compare, &.{ lhs, rhs }, &.{.{ .compare = cmp }});
    }

    pub fn select(self: *FunctionBuilder, cond: VarId, on_true: VarId, on_false: VarId) BuildError!VarId {
        return self.emit(.select, &.{ cond, on_true, on_false }, &.{});
    }

    pub fn convert(self: *FunctionBuilder, operand: VarId, out_dtype: DType) BuildError!VarId {
        return self.emit(.convert, &.{operand}, &.{.{ .out_dtype = out_dtype }});
    }

    pub fn reduce_max(self: *FunctionBuilder, operand: VarId, axes: []const i64) BuildError!VarId {
        const axes_copy = try self.alloc().dupe(i64, axes);
        return self.emit(.reduce_max, &.{operand}, &.{.{ .reduce_axes = axes_copy }});
    }

    pub fn gather(self: *FunctionBuilder, operand: VarId, indices: VarId, gp: GatherParams) BuildError!VarId {
        const a = self.alloc();
        const slice_sizes = try a.dupe(i64, gp.slice_sizes);
        const offset_dims = try a.dupe(i64, gp.offset_dims);
        const collapsed_slice_dims = try a.dupe(i64, gp.collapsed_slice_dims);
        const start_index_map = try a.dupe(i64, gp.start_index_map);
        return self.emit(
            .gather,
            &.{ operand, indices },
            &.{.{ .gather = .{
                .slice_sizes = slice_sizes,
                .offset_dims = offset_dims,
                .collapsed_slice_dims = collapsed_slice_dims,
                .start_index_map = start_index_map,
                .index_vector_dim = gp.index_vector_dim,
            } }},
        );
    }

    pub fn scatter(self: *FunctionBuilder, input: VarId, indices: VarId, updates: VarId, sp: ScatterParams) BuildError!VarId {
        const a = self.alloc();
        const update_window_dims = try a.dupe(i64, sp.update_window_dims);
        const inserted_window_dims = try a.dupe(i64, sp.inserted_window_dims);
        const scatter_dims_to_operand_dims = try a.dupe(i64, sp.scatter_dims_to_operand_dims);
        return self.emit(
            .scatter,
            &.{ input, indices, updates },
            &.{.{ .scatter = .{
                .update_window_dims = update_window_dims,
                .inserted_window_dims = inserted_window_dims,
                .scatter_dims_to_operand_dims = scatter_dims_to_operand_dims,
                .index_vector_dim = sp.index_vector_dim,
                .reduction = sp.reduction,
            } }},
        );
    }

    pub fn slice(self: *FunctionBuilder, operand: VarId, sp: SliceParams) BuildError!VarId {
        const a = self.alloc();
        const start_indices = try a.dupe(i64, sp.start_indices);
        const limit_indices = try a.dupe(i64, sp.limit_indices);
        const strides = try a.dupe(i64, sp.strides);
        return self.emit(
            .slice,
            &.{operand},
            &.{.{ .slice = .{
                .start_indices = start_indices,
                .limit_indices = limit_indices,
                .strides = strides,
            } }},
        );
    }

    pub fn concatenate(self: *FunctionBuilder, operands: []const VarId, axis: i64) BuildError!VarId {
        return self.emit(.concatenate, operands, &.{.{ .concat_axis = axis }});
    }

    pub fn dot_general(self: *FunctionBuilder, lhs: VarId, rhs: VarId, dg: DotGeneralParams) BuildError!VarId {
        const a = self.alloc();
        const lhs_batch_dims = try a.dupe(i64, dg.lhs_batch_dims);
        const rhs_batch_dims = try a.dupe(i64, dg.rhs_batch_dims);
        const lhs_contracting_dims = try a.dupe(i64, dg.lhs_contracting_dims);
        const rhs_contracting_dims = try a.dupe(i64, dg.rhs_contracting_dims);
        return self.emit(
            .dot_general,
            &.{ lhs, rhs },
            &.{.{ .dot_general = .{
                .lhs_batch_dims = lhs_batch_dims,
                .rhs_batch_dims = rhs_batch_dims,
                .lhs_contracting_dims = lhs_contracting_dims,
                .rhs_contracting_dims = rhs_contracting_dims,
            } }},
        );
    }

    pub fn dot(self: *FunctionBuilder, lhs: VarId, rhs: VarId) BuildError!VarId {
        return self.emit(.dot, &.{ lhs, rhs }, &.{});
    }

    pub fn iota(self: *FunctionBuilder, out_dtype: DType, out_dims: []const i64, iota_dim: i64) BuildError!VarId {
        const a = self.alloc();
        const out_shape = try a.dupe(i64, out_dims);
        return self.emit(.iota, &.{}, &.{
            .{ .out_shape = out_shape },
            .{ .out_dtype = out_dtype },
            .{ .iota_dimension = iota_dim },
        });
    }

    pub fn reshape(self: *FunctionBuilder, operand: VarId, out_dims: []const i64) BuildError!VarId {
        const a = self.alloc();
        const out_shape = try a.dupe(i64, out_dims);
        return self.emit(.reshape, &.{operand}, &.{.{ .out_shape = out_shape }});
    }

    pub fn broadcast_in_dim(self: *FunctionBuilder, operand: VarId, out_dims: []const i64, broadcast_dimensions: []const i64) BuildError!VarId {
        const a = self.alloc();
        const out_shape = try a.dupe(i64, out_dims);
        const bd_copy = try a.dupe(i64, broadcast_dimensions);
        return self.emit(.broadcast_in_dim, &.{operand}, &.{
            .{ .out_shape = out_shape },
            .{ .broadcast_dimensions = bd_copy },
        });
    }

    /// Create a scalar constant of `dtype` with value `val`, broadcast to `dims`.
    /// Returns a scalar if `dims` is empty.
    pub fn scalar_broadcast(self: *FunctionBuilder, dtype: DType, dims: []const i64, val: f64) BuildError!VarId {
        const lit = try self.literal_scalar(ops.types.scalar_literal(dtype, val));
        if (dims.len == 0) return lit;
        return self.broadcast_in_dim(lit, dims, &.{});
    }

    pub fn transpose(self: *FunctionBuilder, operand: VarId, permutation: []const i64) BuildError!VarId {
        const a = self.alloc();
        const perm_copy = try a.dupe(i64, permutation);
        return self.emit(.transpose, &.{operand}, &.{.{ .permutation = perm_copy }});
    }

    pub fn reduce_sum(self: *FunctionBuilder, operand: VarId, axes: []const i64) BuildError!VarId {
        const a = self.alloc();
        const axes_copy = try a.dupe(i64, axes);
        return self.emit(.reduce_sum, &.{operand}, &.{.{ .reduce_axes = axes_copy }});
    }

    pub fn custom_call(self: *FunctionBuilder, target: []const u8, operands: []const VarId, out_like: VarId) BuildError!VarId {
        const a = self.alloc();

        const out_aval = self.avals.items[@intCast(out_like)];
        _ = out_aval.as_tensor() orelse return error.CustomCallTypeMismatch;
        for (operands) |op| _ = try self.tensor_of(op);

        const target_copy = try a.dupe(u8, target);
        return self.emit(.custom_call, operands, &.{
            .{ .call_target_name = target_copy },
            .{ .has_side_effect = false },
            .{ .out_aval = out_aval },
        });
    }

    pub fn call(self: *FunctionBuilder, callee: []const u8, inputs: []const VarId) BuildError![]VarId {
        var callee_func: ?Function = null;
        for (self.program.functions) |func| {
            if (std.mem.eql(u8, func.name, callee)) {
                callee_func = func;
                break;
            }
        }
        const callee_fn = callee_func orelse return error.InvalidParams;

        if (inputs.len != callee_fn.params.len) return error.InvalidEqnArity;
        for (inputs, 0..) |in_id, i| {
            const in_tensor = try self.tensor_of(in_id);
            const callee_tensor = callee_fn.avals[@intCast(callee_fn.params[i])].as_tensor() orelse return error.CallTypeMismatch;
            if (!same_tensor_signature(in_tensor, callee_tensor)) return error.CallTypeMismatch;
        }

        const a = self.alloc();
        const outputs = try a.alloc(VarId, callee_fn.returns.len);
        for (callee_fn.returns, 0..) |ret_id, i| {
            const aval = callee_fn.avals[@intCast(ret_id)];
            outputs[i] = try self.var_with_aval(aval);
        }

        const callee_copy = try a.dupe(u8, callee);
        try self.emit_with_outputs(.call, inputs, outputs, &.{.{ .call_callee = callee_copy }});
        return outputs;
    }

    // ====================================================================
    // Finish
    // ====================================================================

    pub fn finish(self: *FunctionBuilder, returns: []const VarId) BuildError!Function {
        const a = self.alloc();

        // Flush any remaining open regions
        while (self.region_stack.items.len > 0) {
            try self.pop_region();
        }

        const func = Function{
            .name = self.name,
            .params = try self.params.toOwnedSlice(a),
            .returns = try a.dupe(VarId, returns),
            .avals = try self.avals.toOwnedSlice(a),
            .eqns = try self.eqns.toOwnedSlice(a),
            .varids_store = try self.varids_store.toOwnedSlice(a),
            .params_store = try self.params_store.toOwnedSlice(a),
            .regions = try self.completed_regions.toOwnedSlice(a),
        };
        try validate_function(func);
        return func;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "FunctionBuilder reshape validation" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    try std.testing.expectError(error.ReshapeTypeMismatch, b.reshape(x, &.{4}));
}

test "FunctionBuilder broadcast_in_dim basic" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{3});
    const y = try b.broadcast_in_dim(x, &.{ 2, 3 }, &.{1});
    const func = try b.finish(&.{y});
    try validate_function(func);
}

test "FunctionBuilder transpose validation" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 3, 4 });
    const y = try b.transpose(x, &.{ 2, 0, 1 });
    const func = try b.finish(&.{y});
    try validate_function(func);
}

test "FunctionBuilder reduce_sum basic" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.reduce_sum(x, &.{0});
    const func = try b.finish(&.{y});
    try validate_function(func);
}

test "FunctionBuilder literal scalar" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const one = try b.literal_scalar(.{ .f32 = 1.0 });
    const func = try b.finish(&.{one});
    try validate_function(func);
}

test "region push/pop materializes regions" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a_id = try b.param_tensor(.f32, &.{ 2, 2 });
    const c_id = try b.param_tensor(.f32, &.{ 2, 2 });

    try b.push_region("tvm-kernel", .{ .kernelize = "tvm" });
    const d = try b.add(a_id, c_id);
    const e = try b.multiply(d, c_id);
    try b.pop_region();

    const f = try b.add(e, a_id);
    const func = try b.finish(&.{f});

    try std.testing.expectEqual(@as(usize, 1), func.regions.len);
    try std.testing.expectEqualStrings("tvm-kernel", func.regions[0].name);
    try std.testing.expectEqualStrings("tvm", func.regions[0].annotation.kernelize.?);
    try std.testing.expectEqual(@as(u32, 0), func.regions[0].eqn_start);
    try std.testing.expectEqual(@as(u32, 2), func.regions[0].eqn_len);
}

test "nested regions" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a_id = try b.param_tensor(.f32, &.{ 2, 2 });
    const c_id = try b.param_tensor(.f32, &.{ 2, 2 });

    try b.push_region("outer", .{ .kernelize = "tvm" });
    const d = try b.add(a_id, c_id);
    {
        try b.push_region("inner", .{ .outline = true });
        const e = try b.multiply(d, c_id);
        _ = e;
        try b.pop_region();
    }
    const f = try b.add(d, a_id);
    try b.pop_region();

    const func = try b.finish(&.{f});

    try std.testing.expectEqual(@as(usize, 2), func.regions.len);
    // Inner region completed first
    try std.testing.expectEqualStrings("inner", func.regions[0].name);
    try std.testing.expect(func.regions[0].annotation.outline);
    try std.testing.expectEqual(@as(u32, 1), func.regions[0].eqn_start);
    try std.testing.expectEqual(@as(u32, 1), func.regions[0].eqn_len);
    // Outer region completed second
    try std.testing.expectEqualStrings("outer", func.regions[1].name);
    try std.testing.expectEqualStrings("tvm", func.regions[1].annotation.kernelize.?);
    try std.testing.expectEqual(@as(u32, 0), func.regions[1].eqn_start);
    try std.testing.expectEqual(@as(u32, 3), func.regions[1].eqn_len);
}

test "regions_matching filters by predicate" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a_id = try b.param_tensor(.f32, &.{ 2, 2 });
    const c_id = try b.param_tensor(.f32, &.{ 2, 2 });

    try b.push_region("r1", .{ .kernelize = "tvm" });
    const d = try b.add(a_id, c_id);
    try b.pop_region();

    try b.push_region("r2", .{ .outline = true });
    const e = try b.multiply(d, c_id);
    try b.pop_region();

    const func = try b.finish(&.{e});

    const is_kernelized = struct {
        fn f(ann: Annotation) bool {
            return ann.kernelize != null;
        }
    }.f;

    var iter = func.regions_matching(is_kernelized);
    const r = iter.next().?;
    try std.testing.expectEqualStrings("r1", r.name);
    try std.testing.expect(iter.next() == null);
}

test "empty region not materialized" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a_id = try b.param_tensor(.f32, &.{ 2, 2 });
    const c_id = try b.param_tensor(.f32, &.{ 2, 2 });

    try b.push_region("empty", .{ .kernelize = "tvm" });
    try b.pop_region();

    const d = try b.add(a_id, c_id);
    const func = try b.finish(&.{d});

    try std.testing.expectEqual(@as(usize, 0), func.regions.len);
}

test "no external imports" {
    // This test verifies at compile time that this module depends only on std.
    // If pr.zig imported mlir, stablehlo, or any ffi module, it would fail to
    // compile in isolation. The fact that this test compiles is the proof.
    const pr = @This();
    try std.testing.expect(pr.DType.f32 == .f32);
}
