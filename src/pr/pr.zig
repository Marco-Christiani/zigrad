//! Toolchain-neutral program representation.
//!
//! A `Program` arena stores functions, operations, SSA values, operands, and
//!  parameter slices. `Program.deinit` releases the arena in one operation.
//! Lowering remains an explicit compilation operation outside PR.
//!
//! ## Design invariants
//!
//! 1. Operations have one result except `call` and `custom_call`.
//! 2. `FunctionBuilder.finish` validates through the operation registry.
//! 3. Builder methods copy slice parameters into the program arena.
const std = @import("std");
const ops = @import("ops/ops.zig");
const pr_log = std.log.scoped(.@"zg/pr");
const Allocator = std.mem.Allocator;

pub const DType = @import("../dtype.zig").DType;

/// Shape represented by a dimension slice.
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

/// Maximum rank stored by `BoundedShape`.
pub const max_rank = 8;

/// Stack-allocated shape with a fixed maximum rank.
pub const BoundedShape = struct {
    /// TODO(shape): Define the dimension scalar and host-index conversion policy.
    buf: [max_rank]i64 = undefined,

    // TODO(shape): Reevaluate the separate length field against a slice-backed
    //  representation before changing fixed-capacity shape storage.
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

    /// Format shape for display.
    pub fn format(shape: BoundedShape, allocator: std.mem.Allocator) ![]const u8 {
        const dims = shape.const_slice();
        if (dims.len == 0) return try allocator.dupe(u8, "scalar");

        var result = std.ArrayList(u8).initCapacity(allocator, 32) catch
            return try allocator.dupe(u8, "[...]");
        defer result.deinit(allocator);

        const writer = result.writer(allocator);
        try writer.writeAll("[");
        for (dims, 0..) |d, i| {
            if (i > 0) try writer.writeAll(", ");
            try writer.print("{d}", .{d});
        }
        try writer.writeAll("]");

        return try result.toOwnedSlice(allocator);
    }
};

/// Abstract value: the type of a Var without its data.
///
/// ## ADR
/// The union leaves room for non-tensor values without changing the `Var`
///  layout.
pub const Aval = union(enum) {
    tensor: Tensor,

    /// Extract the tensor type. Exhaustive over Aval variants.
    pub fn as_tensor(self: Aval) Tensor {
        return switch (self) {
            .tensor => |t| t,
        };
    }
};

/// PR tensor descriptor containing a data type and shape.
///
/// It carries no data, device, or buffer.
/// TODO(pr): Rename this descriptor and its `Aval` tag to distinguish them
///  from runtime tensors.
pub const Tensor = struct {
    dtype: DType,
    shape: Shape,
};

/// Scalar constant keyed by its `DType`.
/// Float16 variants store the raw u16 bit pattern.
pub const Literal = union(DType) {
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
        return std.meta.activeTag(self);
    }

    /// Create a scalar literal by converting from f64.
    pub fn from_f64(value_dtype: DType, value: f64) Literal {
        return switch (value_dtype) {
            .f16 => .{ .f16 = DType.f16.encode(f64, value) },
            .bf16 => .{ .bf16 = DType.bf16.encode(f64, value) },
            .f32 => .{ .f32 = @floatCast(value) },
            .f64 => .{ .f64 = value },
            .i8 => .{ .i8 = @intFromFloat(value) },
            .u8 => .{ .u8 = @intFromFloat(value) },
            .i32 => .{ .i32 = @intFromFloat(value) },
            .i64 => .{ .i64 = @intFromFloat(value) },
            .u32 => .{ .u32 = @intFromFloat(value) },
            .u64 => .{ .u64 = @intFromFloat(value) },
            // TODO(pr): Define boolean conversion semantics for NaN.
            .bool => .{ .bool = value != 0.0 },
        };
    }
};

/// Primitive operation tag.
///
/// Each variant corresponds to a Params payload type and an entry in the op
///  registry (ops/ops.zig) providing at least `validate` and `infer_output`.
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
    mm,
    bmm,
    dot_general,
    convolution,
    reshape,
    iota,
    broadcast_in_dim,
    transpose,
    slice,
    concatenate,
    reduce,
    call,
    custom_call,
};

/// Parameters for gather (indexed read from a tensor).
///
/// Given an operand and an indices tensor, extracts slices from the operand at
///  positions given by indices.
pub const GatherParams = struct {
    /// Size of the slice extracted along each operand dimension.
    slice_sizes: []const i64,
    /// Dimensions of the output that correspond to the slice window.
    offset_dims: []const i64,
    /// Slice dimensions that are collapsed (size must be 1).
    collapsed_slice_dims: []const i64,
    /// Maps each index vector element to an operand dimension.
    start_index_map: []const i64,
    /// Which dimension of the indices tensor holds the index vector.
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

/// Parameters for scatter (indexed write into a tensor).
///
/// Writes `updates` into `operand` at positions given by `indices`, combining
///  with `reduction`.
pub const ScatterParams = struct {
    /// Dimensions of `updates` that correspond to the update window.
    update_window_dims: []const i64,
    /// Window dimensions that are inserted (size 1) rather than present in updates.
    inserted_window_dims: []const i64,
    /// Maps each index vector element to an operand dimension.
    scatter_dims_to_operand_dims: []const i64,
    /// Which dimension of the indices tensor holds the index vector.
    index_vector_dim: i64,
    /// How overlapping updates are combined.
    reduction: ScatterReduction = .add,
};

pub const ScatterReduction = enum {
    add,
    max,
    min,
    mul,
};

/// Parameters for slice (contiguous sub-tensor extraction).
pub const SliceParams = struct {
    /// Per-dimension inclusive start offsets.
    start_indices: []const i64,
    /// Per-dimension exclusive end offsets.
    limit_indices: []const i64,
    /// Per-dimension step sizes (1 = dense).
    strides: []const i64,
};

/// Parameters for generalized dot product.
///
/// Contracting dims are reduced.
/// Batch dims are preserved as leading output dimensions.
/// Remaining dims become the non-contracted output dimensions.
pub const DotGeneralParams = struct {
    lhs_batch_dims: []const i64,
    rhs_batch_dims: []const i64,
    lhs_contracting_dims: []const i64,
    rhs_contracting_dims: []const i64,
};

/// Maps the logical batch, feature, and spatial axes onto tensor dimensions.
pub const ConvolutionDimensionNumbers = struct {
    input_batch_dimension: i64,
    input_feature_dimension: i64,
    input_spatial_dimensions: []const i64,
    kernel_input_feature_dimension: i64,
    kernel_output_feature_dimension: i64,
    kernel_spatial_dimensions: []const i64,
    output_batch_dimension: i64,
    output_feature_dimension: i64,
    output_spatial_dimensions: []const i64,
};

/// Defines the convolution window, axis mapping, and grouping semantics.
pub const ConvolutionParams = struct {
    /// Step between adjacent windows along each spatial axis.
    window_strides: []const i64,
    /// Low and high padding flattened as `[low0, high0, low1, high1, ...]`.
    padding: []const i64,
    /// Spacing between input elements along each spatial axis.
    lhs_dilation: []const i64,
    /// Spacing between kernel elements along each spatial axis.
    rhs_dilation: []const i64,
    /// Reverse each selected spatial window before contraction.
    window_reversal: []const bool,
    /// Physical axis mappings for the input, kernel, and output tensors.
    dimensions: ConvolutionDimensionNumbers,
    /// Independent groups partitioning the input and output features.
    feature_group_count: i64 = 1,
    /// Independent groups partitioning the input batch and kernel outputs.
    batch_group_count: i64 = 1,
};

pub const ReshapeParams = struct {
    out_shape: []const i64,
};

/// Parameters for generating a tensor of incrementing indices.
pub const IotaParams = struct {
    out_shape: []const i64,
    out_dtype: DType,
    /// Which dimension varies (0-indexed). E.g. for shape [2,3] with
    /// dimension=1, produces [[0,1,2],[0,1,2]].
    dimension: i64,
};

/// Parameters for (symbolically) expanding a tensor to a larger shape.
pub const BroadcastInDimParams = struct {
    /// Target shape of the output tensor.
    out_shape: []const i64,
    /// Maps each input dimension to its position in the output shape.
    /// Length must equal the input rank. E.g. for a [3] input broadcast
    /// to [2,3], dimensions = &.{1} means input dim 0 maps to output dim 1.
    dimensions: []const i64,
};

pub const TransposeParams = struct {
    permutation: []const i64,
};

pub const ConcatenateParams = struct {
    axis: i64,
};

pub const Reduction = enum {
    sum,
    maximum,
};

pub const ReduceParams = struct {
    axes: []const i64,
    operation: Reduction,
};

/// Identity of a registered function within a `Program`.
///
/// Appending or replacing functions preserves existing identities. An identity
///  assigned to a removed function is not reused by the same program.
pub const FunctionId = enum(u32) { _ };

/// Failures from registering a function with a program.
pub const FunctionRegistrationError = Allocator.Error || error{
    DuplicateFunctionName,
    FunctionIdExhausted,
};

const FunctionEntry = struct {
    id: FunctionId,
    function: Function,
};

pub const CallParams = struct {
    callee: FunctionId,
};

/// Parameters for an opaque runtime-dispatched custom call operation.
pub const CustomCallParams = struct {
    /// Dispatch identity interpreted by the selected execution integration.
    target_name: []const u8,
    /// Whether the call has observable effects not represented by its results.
    has_side_effect: bool,
    /// Target-defined semantic data embedded in the operation.
    payload: []const u8 = &.{},
};

/// Parameter union keyed by `Prim`.
pub const Params = union(Prim) {
    literal: Literal,
    add: void,
    subtract: void,
    multiply: void,
    divide: void,
    maximum: void,
    exp: void,
    log: void,
    rsqrt: void,
    logistic: void,
    compare: CompareParams,
    select: void,
    convert: DType,
    gather: GatherParams,
    scatter: ScatterParams,
    dot: void,
    mm: void,
    bmm: void,
    dot_general: DotGeneralParams,
    convolution: ConvolutionParams,
    reshape: ReshapeParams,
    iota: IotaParams,
    broadcast_in_dim: BroadcastInDimParams,
    transpose: TransposeParams,
    slice: SliceParams,
    concatenate: ConcatenateParams,
    reduce: ReduceParams,
    call: CallParams,
    custom_call: CustomCallParams,
};

/// SSA value with inline type and intrinsic use-list
///
/// Analogous to mlir::Value.
pub const Var = struct {
    id: u32,
    aval: Aval,
    /// The op that produces this var, or null for function parameters.
    defining_op: ?*const Op = null,
    /// Head of doubly-linked use-list.
    first_use: ?*Operand = null,

    pub fn as_tensor(self: *const Var) Tensor {
        return self.aval.as_tensor();
    }

    pub fn is_unused(self: *const Var) bool {
        return self.first_use == null;
    }

    pub fn has_one_use(self: *const Var) bool {
        const first = self.first_use orelse return false;
        return first.next == null;
    }

    /// Returns the sole consuming operation of this value.
    ///
    /// Function returns do not add nodes to the intrusive use list. Returns
    ///  null when this value has zero or several consuming operations. The
    ///  returned pointer shares this value's program lifetime.
    pub fn only_user(self: *const Var) ?*const Op {
        const first = self.first_use orelse return null;
        if (first.next != null) return null;
        return first.owner;
    }

    /// Checks if var has *at least* n uses.
    pub fn has_n_uses(self: *const Var, n: u32) bool {
        var count: u32 = 0;
        var cur = self.first_use;
        while (cur) |use| {
            count += 1;
            if (count >= n) return true;
            cur = use.next;
        }
        return false;
    }

    /// Replace all uses of this var with `new`. Transfers the use-list.
    pub fn replace_all_uses_with(self: *Var, new: *Var) void {
        var cur = self.first_use;
        while (cur) |use| {
            const next = use.next;
            use.value = new;
            // unlink from self
            use.prev = null;
            use.next = new.first_use;
            if (new.first_use) |head| head.prev = use;
            new.first_use = use;
            cur = next;
        }
        self.first_use = null;
    }
};

/// A use-chain node linking a consuming Op to the Var it reads.
pub const Operand = struct {
    value: *Var,
    owner: *const Op,
    index: u32,
    prev: ?*Operand = null,
    next: ?*Operand = null,

    /// Rebind this use-site to `new`, preserving def-use invariants.
    ///
    /// This moves one `Operand` node from the old value's use-list to `new`'s use-list
    ///  (inserted at head). List order is maintenance order, not program execution order,
    ///  so its not really interpretable.
    ///
    /// Before:
    /// ```
    /// old.first_use -> A <-> self <-> C
    /// new.first_use -> N1 <-> N2
    /// ```
    ///
    /// After:
    /// ```
    /// old.first_use -> A <-> C
    /// new.first_use -> self <-> N1 <-> N2
    /// ```
    ///
    /// Special case (self is old head):
    ///   `old.first_use = self.next`
    ///
    /// ## ADR
    ///
    /// Another intrusive linking design pattern is to use `prev: ?*?*Operand` which
    ///  is what llvm does, but I find this less clear and not entirely sure why you
    ///  would want this.
    pub fn set(self: *Operand, new: *Var) void {
        const old = self.value;
        if (old == new) return;

        // unlink from old var's use-list
        if (self.prev) |p| {
            p.next = self.next;
        } else {
            old.first_use = self.next;
        }
        if (self.next) |n| n.prev = self.prev;
        // link into new var's use-list
        self.value = new;
        self.prev = null;
        self.next = new.first_use;
        if (new.first_use) |head| head.prev = self;
        new.first_use = self;
    }
};

/// An operation in the IR.
pub const Op = struct {
    id: u32,
    inputs: []Operand,
    outputs: []*Var,
    params: Params,

    pub fn prim(self: *const Op) Prim {
        return std.meta.activeTag(self.params);
    }

    pub fn operand(self: *const Op, i: usize) *Var {
        return self.inputs[i].value;
    }

    pub fn result(self: *const Op, i: usize) *Var {
        return self.outputs[i];
    }

    /// True if all results are unused.
    pub fn use_empty(self: *const Op) bool {
        for (self.outputs) |out| {
            if (!out.is_unused()) return false;
        }
        return true;
    }
};

/// Value carried by a named IR annotation.
pub const AnnotationValue = union(enum) {
    unit,
    boolean: bool,
    integer: i64,
    floating_point: f64,
    string: []const u8,
    strings: []const []const u8,
    bytes: []const u8,

    /// Return the string payload or null for another storage kind.
    pub fn as_string(self: AnnotationValue) ?[]const u8 {
        return switch (self) {
            .string => |value| value,
            else => null,
        };
    }
};

/// A namespaced compiler annotation.
///
/// Annotation names define their own value contracts. PR stores unknown names
///  without requiring registration in a central set.
pub const Annotation = struct {
    /// Namespaced contract name interpreted by an owning pass.
    name: []const u8,
    /// Contract payload retained by PR without interpreting its meaning.
    value: AnnotationValue,
};

/// A named set of PR ops with attached steering metadata.
/// Materialized at `FunctionBuilder.finish()` from the annotation stack by default.
pub const Region = struct {
    id: u32,
    name: []const u8,
    /// Compiler metadata attached to this operation group.
    annotations: []const Annotation,
    /// Stable ids of member ops, in program order.
    op_ids: []const u32,

    /// Find an annotation by its exact namespaced name.
    pub fn find_annotation(self: Region, name: []const u8) ?*const Annotation {
        for (self.annotations) |*annotation| {
            if (std.mem.eql(u8, annotation.name, name)) return annotation;
        }
        return null;
    }
};

/// A named function in the program: parameter vars, a linear op sequence,
///  return vars, and optional annotation regions.
///
/// Constructed by `FunctionBuilder.finish()`, which validates all ops.
/// Rewrites may mutate values and operands after construction.
/// TODO(pr): Define and enforce the permitted mutation surface after
///  `FunctionBuilder.finish()`.
pub const Function = struct {
    name: []const u8,
    /// Compiler metadata attached to this callable unit.
    annotations: []const Annotation = &.{},
    params: []*Var,
    returns: []*Var,
    ops: []*Op,
    regions: []const Region,
    var_count: u32,

    /// Find an annotation by its exact namespaced name.
    pub fn find_annotation(self: Function, name: []const u8) ?*const Annotation {
        for (self.annotations) |*annotation| {
            if (std.mem.eql(u8, annotation.name, name)) return annotation;
        }
        return null;
    }

    /// Return regions that satisfy a predicate.
    pub fn regions_matching(self: Function, predicate: *const fn (Region) bool) RegionIterator {
        return .{ .regions = self.regions, .predicate = predicate, .index = 0 };
    }

    pub fn op_by_id(self: Function, id: u32) ?*Op {
        for (self.ops) |op| {
            if (op.id == id) return op;
        }
        return null;
    }

    pub fn op_index_by_id(self: Function, id: u32) ?usize {
        for (self.ops, 0..) |op, index| {
            if (op.id == id) return index;
        }
        return null;
    }
};

pub const RegionIterator = struct {
    regions: []const Region,
    predicate: *const fn (Region) bool,
    index: usize,

    pub fn next(self: *RegionIterator) ?Region {
        while (self.index < self.regions.len) {
            const region = self.regions[self.index];
            self.index += 1;
            if (self.predicate(region)) return region;
        }
        return null;
    }
};

/// Top-level container for a PR program.
///
/// Owns an arena allocator that backs all IR nodes (Vars, Ops, Operands,
///  param slices, Functions). Call `deinit` to free everything in bulk.
///  Individual nodes are never freed separately.
pub const Program = struct {
    arena: std.heap.ArenaAllocator,
    function_entries: std.MultiArrayList(FunctionEntry),
    function_index_by_id: std.AutoHashMapUnmanaged(FunctionId, usize),
    function_id_by_name: std.StringHashMapUnmanaged(FunctionId),
    next_function_id: u64,
    reserved_function_names: std.ArrayList([]const u8),

    /// Append-only program state used to discard later functions and name
    ///  reservations.
    pub const AppendCheckpoint = struct {
        function_count: usize,
        reservation_count: usize,
    };

    pub fn init(backing_allocator: std.mem.Allocator) Program {
        return .{
            .arena = std.heap.ArenaAllocator.init(backing_allocator),
            .function_entries = .empty,
            .function_index_by_id = .empty,
            .function_id_by_name = .empty,
            .next_function_id = 0,
            .reserved_function_names = .empty,
        };
    }

    pub fn allocator(self: *Program) std.mem.Allocator {
        return self.arena.allocator();
    }

    /// Append a function whose name is not already present and return its
    ///  program identity.
    pub fn add_function(
        self: *Program,
        func: Function,
    ) FunctionRegistrationError!FunctionId {
        if (self.function_id_by_name.contains(func.name)) return error.DuplicateFunctionName;

        const raw_id = std.math.cast(u32, self.next_function_id) orelse
            return error.FunctionIdExhausted;
        const id: FunctionId = @enumFromInt(raw_id);
        const index = self.function_entries.len;
        const arena = self.allocator();

        try self.function_entries.append(arena, .{ .id = id, .function = func });
        errdefer self.function_entries.shrinkRetainingCapacity(index);
        try self.function_index_by_id.put(arena, id, index);
        errdefer std.debug.assert(self.function_index_by_id.remove(id));
        try self.function_id_by_name.put(arena, func.name, id);

        self.next_function_id += 1;
        return id;
    }

    /// Return the program functions in insertion order.
    ///
    /// The slice is borrowed. Do not retain it across program mutation.
    pub fn functions(self: *const Program) []const Function {
        return self.function_entries.slice().items(.function);
    }

    /// Return function identities in the same insertion order as `functions`.
    ///
    /// The slice is borrowed. Do not retain it across program mutation.
    pub fn function_ids(self: *const Program) []const FunctionId {
        return self.function_entries.slice().items(.id);
    }

    /// Replace a function without changing its identity or name.
    pub fn replace_function(
        self: *Program,
        id: FunctionId,
        func: Function,
    ) error{ FunctionIdOutOfRange, FunctionNameMismatch }!void {
        const index = self.function_index_by_id.get(id) orelse
            return error.FunctionIdOutOfRange;
        const functions_column = self.function_entries.slice().items(.function);
        if (!std.mem.eql(u8, functions_column[index].name, func.name))
            return error.FunctionNameMismatch;
        functions_column[index] = func;
    }

    /// Capture function and name-reservation counts for error cleanup.
    pub fn checkpoint_appends(self: *const Program) AppendCheckpoint {
        return .{
            .function_count = self.function_entries.len,
            .reservation_count = self.reserved_function_names.items.len,
        };
    }

    /// Discard functions and name reservations added after `saved`.
    ///
    /// Function replacements and arena allocations are unaffected.
    pub fn restore_appends(self: *Program, saved: AppendCheckpoint) void {
        std.debug.assert(saved.function_count <= self.function_entries.len);
        std.debug.assert(saved.reservation_count <= self.reserved_function_names.items.len);
        const entries = self.function_entries.slice();
        const ids = entries.items(.id);
        const functions_column = entries.items(.function);
        for (saved.function_count..self.function_entries.len) |index| {
            std.debug.assert(self.function_index_by_id.remove(ids[index]));
            std.debug.assert(self.function_id_by_name.remove(functions_column[index].name));
        }
        self.function_entries.shrinkRetainingCapacity(saved.function_count);
        self.reserved_function_names.shrinkRetainingCapacity(saved.reservation_count);
    }

    /// Look up a function by name.
    pub fn get_function(self: *const Program, name: []const u8) ?Function {
        const id = self.get_function_id(name) orelse return null;
        return self.get_function_by_id(id);
    }

    /// Look up a function identity by name.
    pub fn get_function_id(self: *const Program, name: []const u8) ?FunctionId {
        return self.function_id_by_name.get(name);
    }

    /// Look up a function by its stable program identity.
    pub fn get_function_by_id(self: *const Program, id: FunctionId) ?Function {
        const index = self.function_index_by_id.get(id) orelse return null;
        return self.function_entries.slice().items(.function)[index];
    }

    /// Allocate and reserve a function name not present in the program.
    ///
    /// The returned name is owned by `self` and is reserved against subsequent
    /// calls, including nested construction before the function is registered.
    /// The first request returns `base` when available, followed by names with
    /// increasing numeric suffixes. Reservations remain until error cleanup or
    /// program deinitialization.
    pub fn reserve_unique_function_name(self: *Program, base: []const u8) Allocator.Error![]const u8 {
        const a = self.allocator();
        var suffix: usize = 0;
        while (true) : (suffix += 1) {
            const name = if (suffix == 0)
                try a.dupe(u8, base)
            else
                try std.fmt.allocPrint(a, "{s}_{d}", .{ base, suffix });
            if (!self.function_name_available(name)) continue;

            try self.reserved_function_names.append(a, name);
            return name;
        }
    }

    fn function_name_available(self: *const Program, name: []const u8) bool {
        if (self.function_id_by_name.contains(name)) return false;
        for (self.reserved_function_names.items) |reserved| {
            if (std.mem.eql(u8, reserved, name)) return false;
        }
        return true;
    }

    pub fn deinit(self: *Program) void {
        self.arena.deinit();
    }
};

/// Errors from post-construction validation (op registry checks, call
///  signature matching, duplicate function names).
/// Each op type has its own `*TypeMismatch` variant for targeted diagnostics.
pub const ValidationError = error{
    InvalidVar,
    UnsupportedAval,
    InvalidOpArity,
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
    MMTypeMismatch,
    BMMTypeMismatch,
    DotGeneralTypeMismatch,
    ConvolutionTypeMismatch,
    ReshapeTypeMismatch,
    BroadcastInDimTypeMismatch,
    TransposeTypeMismatch,
    SliceTypeMismatch,
    ConcatTypeMismatch,
    ReduceTypeMismatch,
    CallUnresolvedCallee,
    CallArityMismatch,
    CallTypeMismatch,
    CustomCallTypeMismatch,
    IotaTypeMismatch,
    DuplicateFunctionName,
    FunctionIdentityMismatch,
    ScatterAddTypeMismatch,
    DuplicateAnnotationName,
};

/// Errors from FunctionBuilder: validation failures (caught eagerly at
///  `finish`) plus allocation failures from the program arena.
pub const BuildError = ValidationError || Allocator.Error;

fn same_tensor_signature(a: Tensor, b: Tensor) bool {
    if (a.dtype != b.dtype) return false;
    if (a.shape.dims.len != b.shape.dims.len) return false;
    for (a.shape.dims, 0..) |d, i| {
        if (d != b.shape.dims[i]) return false;
    }
    return true;
}

/// Validate a single function: per-op validation via the op registry.
pub fn validate_ops_in_func(func: Function) ValidationError!void {
    for (func.ops) |op| try ops.validate(op);
}

pub fn validate_program(program: *const Program) ValidationError!void {
    if (program.function_index_by_id.count() != program.functions().len or
        program.function_id_by_name.count() != program.functions().len)
        return error.FunctionIdentityMismatch;
    for (program.functions(), program.function_ids(), 0..) |func, id, i| {
        if (program.get_function_id(func.name) != id or
            program.function_index_by_id.get(id) != i or
            program.next_function_id <= @intFromEnum(id))
            return error.FunctionIdentityMismatch;
        try validate_ops_in_func(func);
        try validate_annotations(func.annotations);
        for (func.regions) |region| {
            try validate_annotations(region.annotations);
        }

        for (func.ops) |op| {
            if (op.prim() != .call) continue;

            const call_params = op.params.call;
            const callee_func = program.get_function_by_id(call_params.callee) orelse {
                pr_log.err("call references unknown function id {d} in '{s}'", .{ @intFromEnum(call_params.callee), func.name });
                return error.CallUnresolvedCallee;
            };

            if (op.inputs.len != callee_func.params.len or op.outputs.len != callee_func.returns.len) {
                pr_log.err("call arity mismatch: '{s}' expects {d} inputs/{d} outputs, got {d}/{d}", .{
                    callee_func.name,
                    callee_func.params.len,
                    callee_func.returns.len,
                    op.inputs.len,
                    op.outputs.len,
                });
                return error.CallArityMismatch;
            }

            for (op.inputs, 0..) |operand, idx| {
                const in_tensor = operand.value.as_tensor();
                const callee_tensor = callee_func.params[idx].as_tensor();
                if (!same_tensor_signature(in_tensor, callee_tensor)) return error.CallTypeMismatch;
            }
            for (op.outputs, 0..) |out_var, idx| {
                const out_tensor = out_var.as_tensor();
                const callee_tensor = callee_func.returns[idx].as_tensor();
                if (!same_tensor_signature(out_tensor, callee_tensor)) return error.CallTypeMismatch;
            }
        }
    }
}

fn validate_annotations(annotations: []const Annotation) ValidationError!void {
    for (annotations, 0..) |annotation, index| {
        for (annotations[0..index]) |prior| {
            if (std.mem.eql(u8, prior.name, annotation.name))
                return error.DuplicateAnnotationName;
        }
    }
}

/// Annotation stack entry for tracking active regions during building.
const RegionEntry = struct {
    id: u32,
    name: []const u8,
    annotations: []const Annotation,
    start_index: u32,
};

/// Copy annotations and their variable-length values into `allocator`.
pub fn dupe_annotations(allocator: Allocator, annotations: []const Annotation) BuildError![]const Annotation {
    const result = try allocator.alloc(Annotation, annotations.len);
    for (annotations, 0..) |annotation, index| {
        for (annotations[0..index]) |prior| {
            if (std.mem.eql(u8, prior.name, annotation.name))
                return error.DuplicateAnnotationName;
        }

        result[index] = .{
            .name = try allocator.dupe(u8, annotation.name),
            .value = switch (annotation.value) {
                .string => |value| .{ .string = try allocator.dupe(u8, value) },
                .strings => |values| strings: {
                    const copied = try allocator.alloc([]const u8, values.len);
                    for (values, copied) |value, *destination| {
                        destination.* = try allocator.dupe(u8, value);
                    }
                    break :strings .{ .strings = copied };
                },
                .bytes => |value| .{ .bytes = try allocator.dupe(u8, value) },
                else => annotation.value,
            },
        };
    }
    return result;
}

/// Mutable builder for constructing a Function.
///
/// All allocations go through the program arena.
/// Slice parameters (dims, axes, permutations) are duped into the arena by
///  convenience methods, so callers can pass stack-local slices.
///
/// Use `push_region` / `pop_region` to annotate op ranges with compilation
///  hints (outlining, kernelization). Regions are materialized at `finish`.
pub const FunctionBuilder = struct {
    program: *Program,
    name: []const u8,
    params_list: std.ArrayList(*Var),
    ops_list: std.ArrayList(*Op),
    next_var_id: u32,
    next_op_id: u32,
    next_region_id: u32,
    region_stack: std.ArrayList(RegionEntry),
    completed_regions: std.ArrayList(Region),

    pub fn init(program: *Program, name: []const u8) BuildError!FunctionBuilder {
        const a = program.allocator();
        return .{
            .program = program,
            .name = try a.dupe(u8, name),
            .params_list = try std.ArrayList(*Var).initCapacity(a, 8),
            .ops_list = try std.ArrayList(*Op).initCapacity(a, 16),
            .next_var_id = 0,
            .next_op_id = 0,
            .next_region_id = 0,
            .region_stack = try std.ArrayList(RegionEntry).initCapacity(a, 4),
            .completed_regions = try std.ArrayList(Region).initCapacity(a, 4),
        };
    }

    pub fn deinit(self: *FunctionBuilder) void {
        const a = self.program.allocator();
        self.params_list.deinit(a);
        self.ops_list.deinit(a);
        self.region_stack.deinit(a);
        self.completed_regions.deinit(a);
    }

    pub fn alloc(self: *FunctionBuilder) std.mem.Allocator {
        return self.program.allocator();
    }

    fn next_id(self: *FunctionBuilder) u32 {
        const id = self.next_var_id;
        self.next_var_id += 1;
        return id;
    }

    fn next_op(self: *FunctionBuilder) u32 {
        const id = self.next_op_id;
        self.next_op_id += 1;
        return id;
    }

    fn next_region(self: *FunctionBuilder) u32 {
        const id = self.next_region_id;
        self.next_region_id += 1;
        return id;
    }

    fn create_var(self: *FunctionBuilder, aval: Aval) BuildError!*Var {
        const a = self.alloc();
        const v = try a.create(Var);
        v.* = .{ .id = self.next_id(), .aval = aval };
        return v;
    }

    // Region API

    /// Push a named annotation region.
    /// Ops emitted after this call belong to this region until pop_region is called.
    pub fn push_region(self: *FunctionBuilder, name: []const u8, annotations: []const Annotation) BuildError!void {
        const a = self.alloc();
        try self.region_stack.ensureUnusedCapacity(a, 1);
        const name_copy = try a.dupe(u8, name);
        const annotations_copy = try dupe_annotations(a, annotations);
        self.region_stack.appendAssumeCapacity(.{
            .id = self.next_region(),
            .name = name_copy,
            .annotations = annotations_copy,
            .start_index = @intCast(self.ops_list.items.len),
        });
    }

    /// Pop the most recent annotation region.
    pub fn pop_region(self: *FunctionBuilder) BuildError!void {
        const a = self.alloc();
        const entry = self.region_stack.getLastOrNull() orelse return;
        const op_end: u32 = @intCast(self.ops_list.items.len);
        if (op_end > entry.start_index) {
            try self.completed_regions.ensureUnusedCapacity(a, 1);
            const op_ids = try a.alloc(u32, op_end - entry.start_index);
            for (self.ops_list.items[entry.start_index..op_end], 0..) |op, i| {
                op_ids[i] = op.id;
            }
            _ = self.region_stack.pop();
            self.completed_regions.appendAssumeCapacity(.{
                .id = entry.id,
                .name = entry.name,
                .annotations = entry.annotations,
                .op_ids = op_ids,
            });
        } else {
            _ = self.region_stack.pop();
        }
    }

    // Core Emit

    /// Emit a single-output op, inferring the output type via the op registry.
    ///
    /// Input vars have their use-lists updated: each input gains an Operand
    ///  node linking it to the new op.
    pub fn emit(self: *FunctionBuilder, params: Params, inputs: []const *Var) BuildError!*Var {
        const a = self.alloc();

        try self.ops_list.ensureUnusedCapacity(a, 1);
        const out_aval = try ops.infer_output(a, params, inputs);
        const operands = try a.alloc(Operand, inputs.len);
        const op = try a.create(Op);
        const out_slice = try a.alloc(*Var, 1);
        const out_var = try self.create_var(out_aval);
        out_slice[0] = out_var;

        op.* = .{ .id = self.next_op(), .inputs = operands, .outputs = out_slice, .params = params };
        out_var.defining_op = op;

        for (inputs, 0..) |in_var, i| {
            operands[i] = .{
                .value = in_var,
                .owner = op,
                .index = @intCast(i),
            };
            operands[i].next = in_var.first_use;
            if (in_var.first_use) |head| head.prev = &operands[i];
            in_var.first_use = &operands[i];
        }

        self.ops_list.appendAssumeCapacity(op);
        return out_var;
    }

    /// Emit an operation with explicit output abstract values.
    ///
    /// Compiler transforms and variadic-result operations use this when result
    ///  types cannot be inferred from inputs. Input use lists are updated.
    pub fn emit_outputs(self: *FunctionBuilder, params: Params, inputs: []const *Var, out_avals: []const Aval) BuildError![]*Var {
        const a = self.alloc();

        try self.ops_list.ensureUnusedCapacity(a, 1);
        const out_vars = try a.alloc(*Var, out_avals.len);
        const vars = try a.alloc(Var, out_avals.len);
        const operands = try a.alloc(Operand, inputs.len);
        const op = try a.create(Op);

        for (out_avals, vars, out_vars) |aval, *out_var, *out_var_ptr| {
            out_var.* = .{ .id = self.next_id(), .aval = aval };
            out_var_ptr.* = out_var;
        }

        op.* = .{ .id = self.next_op(), .inputs = operands, .outputs = out_vars, .params = params };
        for (out_vars) |v| v.defining_op = op;

        for (inputs, 0..) |in_var, i| {
            operands[i] = .{
                .value = in_var,
                .owner = op,
                .index = @intCast(i),
            };
            operands[i].next = in_var.first_use;
            if (in_var.first_use) |head| head.prev = &operands[i];
            in_var.first_use = &operands[i];
        }

        self.ops_list.appendAssumeCapacity(op);
        return out_vars;
    }

    pub fn param_tensor(self: *FunctionBuilder, dtype: DType, dims: []const i64) BuildError!*Var {
        const a = self.alloc();
        try self.params_list.ensureUnusedCapacity(a, 1);
        const dims_copy = try a.dupe(i64, dims);
        const v = try self.create_var(.{ .tensor = .{ .dtype = dtype, .shape = .{ .dims = dims_copy } } });
        self.params_list.appendAssumeCapacity(v);
        return v;
    }

    pub fn literal_scalar(self: *FunctionBuilder, value: Literal) BuildError!*Var {
        return try self.emit(.{ .literal = value }, &.{});
    }

    /// Emit a scalar constant of `dtype` from an `f64` value.
    ///
    /// Handles dtype-aware conversion from f64. Prefer this over the two-step
    /// `Literal.from_f64` + `literal_scalar` pattern.
    pub fn scalar(self: *FunctionBuilder, dtype: DType, val: f64) BuildError!*Var {
        return try self.literal_scalar(Literal.from_f64(dtype, val));
    }

    pub fn add(self: *FunctionBuilder, lhs: *Var, rhs: *Var) BuildError!*Var {
        return try self.emit(.{ .add = {} }, &.{ lhs, rhs });
    }

    pub fn subtract(self: *FunctionBuilder, lhs: *Var, rhs: *Var) BuildError!*Var {
        return try self.emit(.{ .subtract = {} }, &.{ lhs, rhs });
    }

    pub fn multiply(self: *FunctionBuilder, lhs: *Var, rhs: *Var) BuildError!*Var {
        return try self.emit(.{ .multiply = {} }, &.{ lhs, rhs });
    }

    pub fn divide(self: *FunctionBuilder, lhs: *Var, rhs: *Var) BuildError!*Var {
        return try self.emit(.{ .divide = {} }, &.{ lhs, rhs });
    }

    pub fn maximum(self: *FunctionBuilder, lhs: *Var, rhs: *Var) BuildError!*Var {
        return try self.emit(.{ .maximum = {} }, &.{ lhs, rhs });
    }

    pub fn exp(self: *FunctionBuilder, operand_var: *Var) BuildError!*Var {
        return try self.emit(.{ .exp = {} }, &.{operand_var});
    }

    pub fn log(self: *FunctionBuilder, operand_var: *Var) BuildError!*Var {
        return try self.emit(.{ .log = {} }, &.{operand_var});
    }

    pub fn rsqrt(self: *FunctionBuilder, operand_var: *Var) BuildError!*Var {
        return try self.emit(.{ .rsqrt = {} }, &.{operand_var});
    }

    pub fn logistic(self: *FunctionBuilder, operand_var: *Var) BuildError!*Var {
        return try self.emit(.{ .logistic = {} }, &.{operand_var});
    }

    pub fn compare(self: *FunctionBuilder, lhs: *Var, rhs: *Var, cmp: CompareParams) BuildError!*Var {
        return try self.emit(.{ .compare = cmp }, &.{ lhs, rhs });
    }

    pub fn select(self: *FunctionBuilder, cond: *Var, on_true: *Var, on_false: *Var) BuildError!*Var {
        return try self.emit(.{ .select = {} }, &.{ cond, on_true, on_false });
    }

    pub fn convert(self: *FunctionBuilder, operand_var: *Var, out_dtype: DType) BuildError!*Var {
        if (operand_var.as_tensor().dtype == out_dtype) return operand_var;
        return try self.emit(.{ .convert = out_dtype }, &.{operand_var});
    }

    pub fn gather(self: *FunctionBuilder, operand_var: *Var, indices: *Var, gp: GatherParams) BuildError!*Var {
        const a = self.alloc();
        return try self.emit(.{ .gather = .{
            .slice_sizes = try a.dupe(i64, gp.slice_sizes),
            .offset_dims = try a.dupe(i64, gp.offset_dims),
            .collapsed_slice_dims = try a.dupe(i64, gp.collapsed_slice_dims),
            .start_index_map = try a.dupe(i64, gp.start_index_map),
            .index_vector_dim = gp.index_vector_dim,
        } }, &.{ operand_var, indices });
    }

    pub fn scatter(self: *FunctionBuilder, input: *Var, indices: *Var, updates: *Var, sp: ScatterParams) BuildError!*Var {
        const a = self.alloc();
        return try self.emit(.{ .scatter = .{
            .update_window_dims = try a.dupe(i64, sp.update_window_dims),
            .inserted_window_dims = try a.dupe(i64, sp.inserted_window_dims),
            .scatter_dims_to_operand_dims = try a.dupe(i64, sp.scatter_dims_to_operand_dims),
            .index_vector_dim = sp.index_vector_dim,
            .reduction = sp.reduction,
        } }, &.{ input, indices, updates });
    }

    pub fn slice(self: *FunctionBuilder, operand_var: *Var, sp: SliceParams) BuildError!*Var {
        const a = self.alloc();
        return try self.emit(.{ .slice = .{
            .start_indices = try a.dupe(i64, sp.start_indices),
            .limit_indices = try a.dupe(i64, sp.limit_indices),
            .strides = try a.dupe(i64, sp.strides),
        } }, &.{operand_var});
    }

    pub fn concatenate(self: *FunctionBuilder, operands: []const *Var, axis: i64) BuildError!*Var {
        return try self.emit(.{ .concatenate = .{ .axis = axis } }, operands);
    }

    pub fn dot_general(self: *FunctionBuilder, lhs: *Var, rhs: *Var, dg: DotGeneralParams) BuildError!*Var {
        const a = self.alloc();
        return try self.emit(.{ .dot_general = .{
            .lhs_batch_dims = try a.dupe(i64, dg.lhs_batch_dims),
            .rhs_batch_dims = try a.dupe(i64, dg.rhs_batch_dims),
            .lhs_contracting_dims = try a.dupe(i64, dg.lhs_contracting_dims),
            .rhs_contracting_dims = try a.dupe(i64, dg.rhs_contracting_dims),
        } }, &.{ lhs, rhs });
    }

    /// Computes the scalar dot product of two equal-length vectors.
    pub fn dot(self: *FunctionBuilder, lhs: *Var, rhs: *Var) BuildError!*Var {
        return try self.emit(.{ .dot = {} }, &.{ lhs, rhs });
    }

    /// Multiplies two rank-two matrices.
    pub fn mm(self: *FunctionBuilder, lhs: *Var, rhs: *Var) BuildError!*Var {
        return try self.emit(.{ .mm = {} }, &.{ lhs, rhs });
    }

    /// Multiplies matrices over one or more identical prefix batch dimensions.
    pub fn bmm(self: *FunctionBuilder, lhs: *Var, rhs: *Var) BuildError!*Var {
        return try self.emit(.{ .bmm = {} }, &.{ lhs, rhs });
    }

    pub fn convolution(self: *FunctionBuilder, lhs: *Var, rhs: *Var, params: ConvolutionParams) BuildError!*Var {
        const a = self.alloc();
        return try self.emit(.{ .convolution = .{
            .window_strides = try a.dupe(i64, params.window_strides),
            .padding = try a.dupe(i64, params.padding),
            .lhs_dilation = try a.dupe(i64, params.lhs_dilation),
            .rhs_dilation = try a.dupe(i64, params.rhs_dilation),
            .window_reversal = try a.dupe(bool, params.window_reversal),
            .dimensions = .{
                .input_batch_dimension = params.dimensions.input_batch_dimension,
                .input_feature_dimension = params.dimensions.input_feature_dimension,
                .input_spatial_dimensions = try a.dupe(i64, params.dimensions.input_spatial_dimensions),
                .kernel_input_feature_dimension = params.dimensions.kernel_input_feature_dimension,
                .kernel_output_feature_dimension = params.dimensions.kernel_output_feature_dimension,
                .kernel_spatial_dimensions = try a.dupe(i64, params.dimensions.kernel_spatial_dimensions),
                .output_batch_dimension = params.dimensions.output_batch_dimension,
                .output_feature_dimension = params.dimensions.output_feature_dimension,
                .output_spatial_dimensions = try a.dupe(i64, params.dimensions.output_spatial_dimensions),
            },
            .feature_group_count = params.feature_group_count,
            .batch_group_count = params.batch_group_count,
        } }, &.{ lhs, rhs });
    }

    pub fn iota(self: *FunctionBuilder, out_dtype: DType, out_dims: []const i64, iota_dim: i64) BuildError!*Var {
        const a = self.alloc();
        return try self.emit(.{ .iota = .{
            .out_shape = try a.dupe(i64, out_dims),
            .out_dtype = out_dtype,
            .dimension = iota_dim,
        } }, &.{});
    }

    pub fn reshape(self: *FunctionBuilder, operand_var: *Var, out_dims: []const i64) BuildError!*Var {
        const a = self.alloc();
        return try self.emit(.{ .reshape = .{
            .out_shape = try a.dupe(i64, out_dims),
        } }, &.{operand_var});
    }

    pub fn broadcast_in_dim(self: *FunctionBuilder, operand_var: *Var, out_dims: []const i64, broadcast_dimensions: []const i64) BuildError!*Var {
        const a = self.alloc();
        return try self.emit(.{ .broadcast_in_dim = .{
            .out_shape = try a.dupe(i64, out_dims),
            .dimensions = try a.dupe(i64, broadcast_dimensions),
        } }, &.{operand_var});
    }

    /// Create a scalar constant of `dtype` with value `val`, broadcast to `dims`.
    ///
    /// Returns a scalar if `dims` is empty.
    pub fn scalar_broadcast(self: *FunctionBuilder, dtype: DType, dims: []const i64, val: f64) BuildError!*Var {
        const lit = try self.scalar(dtype, val);
        if (dims.len == 0) return lit;
        return try self.broadcast_in_dim(lit, dims, &.{});
    }

    pub fn transpose(self: *FunctionBuilder, operand_var: *Var, permutation: []const i64) BuildError!*Var {
        const a = self.alloc();
        return try self.emit(.{ .transpose = .{
            .permutation = try a.dupe(i64, permutation),
        } }, &.{operand_var});
    }

    /// Reduces `operand_var` over the selected axes.
    pub fn reduce(self: *FunctionBuilder, operand_var: *Var, params: ReduceParams) BuildError!*Var {
        const a = self.alloc();
        return try self.emit(.{ .reduce = .{
            .axes = try a.dupe(i64, params.axes),
            .operation = params.operation,
        } }, &.{operand_var});
    }

    /// Emit a custom call with explicit output types.
    ///
    /// The program arena copies the target name and payload.
    pub fn custom_call(
        self: *FunctionBuilder,
        params: CustomCallParams,
        inputs: []const *Var,
        out_avals: []const Aval,
    ) BuildError![]*Var {
        const a = self.alloc();
        const stored_params = CustomCallParams{
            .target_name = try a.dupe(u8, params.target_name),
            .has_side_effect = params.has_side_effect,
            .payload = try a.dupe(u8, params.payload),
        };
        return try self.emit_outputs(.{ .custom_call = stored_params }, inputs, out_avals);
    }

    /// Emit a call to a function registered in the builder's program.
    ///
    /// Input arity and tensor signatures must match the registered function.
    pub fn call(self: *FunctionBuilder, callee_id: FunctionId, inputs: []const *Var) BuildError![]*Var {
        const callee_fn = self.program.get_function_by_id(callee_id) orelse {
            pr_log.err("call references unknown function id {d}", .{@intFromEnum(callee_id)});
            return error.CallUnresolvedCallee;
        };

        if (inputs.len != callee_fn.params.len) {
            pr_log.err("call arity mismatch: '{s}' expects {d} inputs, got {d}", .{ callee_fn.name, callee_fn.params.len, inputs.len });
            return error.CallArityMismatch;
        }
        for (inputs, 0..) |in_var, i| {
            const in_tensor = in_var.aval.as_tensor();
            const callee_tensor = callee_fn.params[i].aval.as_tensor();
            if (!same_tensor_signature(in_tensor, callee_tensor)) return error.CallTypeMismatch;
        }

        const a = self.alloc();
        const out_avals = try a.alloc(Aval, callee_fn.returns.len);
        for (callee_fn.returns, 0..) |ret_var, i| {
            out_avals[i] = ret_var.aval;
        }

        return try self.emit_outputs(.{ .call = .{ .callee = callee_id } }, inputs, out_avals);
    }

    /// Finalize and validate the function.
    ///
    /// Open regions are closed before validation.
    ///
    /// The builder must not be used again except to call `deinit`.
    pub fn finish(self: *FunctionBuilder, returns: []const *Var) BuildError!Function {
        const a = self.alloc();

        for (self.ops_list.items) |op| try ops.validate(op);
        while (self.region_stack.items.len > 0) {
            try self.pop_region();
        }

        const stored_returns = try a.dupe(*Var, returns);
        const func = Function{
            .name = self.name,
            .params = self.params_list.items,
            .returns = stored_returns,
            .ops = self.ops_list.items,
            .regions = self.completed_regions.items,
            .var_count = self.next_var_id,
        };
        self.params_list = .empty;
        self.ops_list = .empty;
        self.completed_regions = .empty;
        return func;
    }
};

// Tests.

test {
    std.testing.refAllDecls(@This());
    _ = @import("tests/eval.zig");
    _ = @import("tests/grad_check.zig");
}

test "Program.add_function rejects duplicate names" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    const function = Function{
        .name = "main",
        .params = &.{},
        .returns = &.{},
        .ops = &.{},
        .regions = &.{},
        .var_count = 0,
    };

    _ = try program.add_function(function);
    try std.testing.expectError(error.DuplicateFunctionName, program.add_function(function));
}

test "Program.reserve_unique_function_name increments occupied names" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var first = try FunctionBuilder.init(&program, "generated");
    defer first.deinit();
    _ = try program.add_function(try first.finish(&.{}));

    var second = try FunctionBuilder.init(&program, "generated_1");
    defer second.deinit();
    _ = try program.add_function(try second.finish(&.{}));

    try std.testing.expectEqualStrings("available", try program.reserve_unique_function_name("available"));
    try std.testing.expectEqualStrings("available_1", try program.reserve_unique_function_name("available"));
    try std.testing.expectEqualStrings("generated_2", try program.reserve_unique_function_name("generated"));
}

test "Program.restore discards appended functions and reservations" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var first = try FunctionBuilder.init(&program, "first");
    defer first.deinit();
    const first_id = try program.add_function(try first.finish(&.{}));
    try std.testing.expectEqual(@as(FunctionId, @enumFromInt(0)), first_id);

    const saved = program.checkpoint_appends();
    _ = try program.reserve_unique_function_name("reserved");
    var second = try FunctionBuilder.init(&program, "second");
    defer second.deinit();
    const second_id = try program.add_function(try second.finish(&.{}));
    try std.testing.expectEqual(@as(FunctionId, @enumFromInt(1)), second_id);

    program.restore_appends(saved);
    try std.testing.expectEqual(@as(usize, 1), program.functions().len);
    try std.testing.expectEqual(@as(usize, 0), program.checkpoint_appends().reservation_count);
    try std.testing.expect(program.get_function("first") != null);
    try std.testing.expect(program.get_function("second") == null);
    try std.testing.expect(program.get_function_by_id(@enumFromInt(1)) == null);

    var third = try FunctionBuilder.init(&program, "third");
    defer third.deinit();
    const third_id = try program.add_function(try third.finish(&.{}));
    try std.testing.expectEqual(@as(FunctionId, @enumFromInt(2)), third_id);
    try std.testing.expect(program.get_function_by_id(second_id) == null);
}

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
    try validate_ops_in_func(func);
}

test "FunctionBuilder transpose validation" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 3, 4 });
    const y = try b.transpose(x, &.{ 2, 0, 1 });
    const func = try b.finish(&.{y});
    try validate_ops_in_func(func);
}

test "FunctionBuilder reduce basic" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.reduce(x, .{ .axes = &.{0}, .operation = .sum });
    const func = try b.finish(&.{y});
    try validate_ops_in_func(func);
}

test "FunctionBuilder finish preserves state after validation failure" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var builder = try FunctionBuilder.init(&program, "main");
    defer builder.deinit();

    const lhs = try builder.param_tensor(.f32, &.{ 2, 3 });
    const rhs = try builder.param_tensor(.f32, &.{ 2, 3 });
    const output = try builder.add(lhs, rhs);
    const valid_aval = output.aval;
    output.aval = .{ .tensor = .{ .dtype = .f64, .shape = .{ .dims = &.{ 2, 3 } } } };

    try std.testing.expectError(error.AddTypeMismatch, builder.finish(&.{output}));
    try std.testing.expectEqual(@as(usize, 2), builder.params_list.items.len);
    try std.testing.expectEqual(@as(usize, 1), builder.ops_list.items.len);

    output.aval = valid_aval;
    const function = try builder.finish(&.{output});
    try std.testing.expectEqual(@as(usize, 2), function.params.len);
    try std.testing.expectEqual(@as(usize, 1), function.ops.len);
}

test "FunctionBuilder releases allocations after every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn build(allocator: Allocator) !void {
            var program = Program.init(allocator);
            defer program.deinit();
            var builder = try FunctionBuilder.init(&program, "main");
            defer builder.deinit();

            const lhs = try builder.param_tensor(.f32, &.{ 2, 3 });
            const rhs = try builder.param_tensor(.f32, &.{ 2, 3 });
            try builder.push_region("sum", &.{});
            const output = try builder.add(lhs, rhs);
            const function = try builder.finish(&.{output});
            _ = try program.add_function(function);
        }
    }.build, .{});
}

test "FunctionBuilder literal scalar" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const one = try b.literal_scalar(.{ .f32 = 1.0 });
    const func = try b.finish(&.{one});
    try validate_ops_in_func(func);
}

test "Literal.from_f64 tags by DType" {
    const i = Literal.from_f64(.i32, -7.0);
    const f = Literal.from_f64(.f32, 1.5);
    try std.testing.expectEqual(DType.i32, i.dtype());
    try std.testing.expectEqual(DType.f32, f.dtype());
    const g = Literal.from_f64(.f16, std.math.floatMax(f32));
    try std.testing.expectEqual(DType.f16, g.dtype());
}

test "region push/pop materializes regions" {
    const kernel = @import("../kernel.zig");
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a_id = try b.param_tensor(.f32, &.{ 2, 2 });
    const c_id = try b.param_tensor(.f32, &.{ 2, 2 });

    try b.push_region("tvm-kernel", &.{kernel.provider_annotation("tvm")});
    const d = try b.add(a_id, c_id);
    const e = try b.multiply(d, c_id);
    try b.pop_region();

    const f = try b.add(e, a_id);
    const func = try b.finish(&.{f});

    try std.testing.expectEqual(@as(usize, 1), func.regions.len);
    try std.testing.expectEqual(@as(u32, 0), func.regions[0].id);
    try std.testing.expectEqualStrings("tvm-kernel", func.regions[0].name);
    try std.testing.expectEqualStrings("tvm", (try kernel.requested_providers(func.regions[0])).?.at(0));
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, func.regions[0].op_ids);
}

test "region annotations accept namespaced values" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var builder = try FunctionBuilder.init(&program, "annotations");
    defer builder.deinit();

    var annotation_name = [_]u8{ 'e', 'x', 'a', 'm', 'p', 'l', 'e', '.', 'i', 'd' };
    var annotation_value = [_]u8{ 'v', '1' };
    try builder.push_region("annotated", &.{.{
        .name = &annotation_name,
        .value = .{ .string = &annotation_value },
    }});
    const value = try builder.param_tensor(.f32, &.{});
    const result = try builder.log(value);
    try builder.pop_region();

    annotation_name[0] = 'X';
    annotation_value[0] = 'X';

    const func = try builder.finish(&.{result});
    const found = func.regions[0].find_annotation("example.id").?;
    try std.testing.expectEqualStrings("v1", found.value.as_string().?);
}

test "region annotation names are unique" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var builder = try FunctionBuilder.init(&program, "duplicate_annotations");
    defer builder.deinit();

    try std.testing.expectError(error.DuplicateAnnotationName, builder.push_region("duplicate", &.{
        .{ .name = "example.value", .value = .unit },
        .{ .name = "example.value", .value = .{ .integer = 1 } },
    }));
}

test "nested regions" {
    const kernel = @import("../kernel.zig");
    const outline = @import("transform/outline.zig");
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a_id = try b.param_tensor(.f32, &.{ 2, 2 });
    const c_id = try b.param_tensor(.f32, &.{ 2, 2 });

    try b.push_region("outer", &.{kernel.provider_annotation("tvm")});
    const d = try b.add(a_id, c_id);
    {
        try b.push_region("inner", &.{outline.annotation});
        const e = try b.multiply(d, c_id);
        _ = e;
        try b.pop_region();
    }
    const f = try b.add(d, a_id);
    try b.pop_region();

    const func = try b.finish(&.{f});

    try std.testing.expectEqual(@as(usize, 2), func.regions.len);
    // Inner region completed first
    try std.testing.expectEqual(@as(u32, 1), func.regions[0].id);
    try std.testing.expectEqualStrings("inner", func.regions[0].name);
    try std.testing.expect(try outline.is_requested(func.regions[0]));
    try std.testing.expectEqualSlices(u32, &.{1}, func.regions[0].op_ids);
    // Outer region completed second
    try std.testing.expectEqual(@as(u32, 0), func.regions[1].id);
    try std.testing.expectEqualStrings("outer", func.regions[1].name);
    try std.testing.expectEqualStrings("tvm", (try kernel.requested_providers(func.regions[1])).?.at(0));
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, func.regions[1].op_ids);
}

test "regions_matching filters by predicate" {
    const kernel = @import("../kernel.zig");
    const outline = @import("transform/outline.zig");
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a_id = try b.param_tensor(.f32, &.{ 2, 2 });
    const c_id = try b.param_tensor(.f32, &.{ 2, 2 });

    try b.push_region("r1", &.{kernel.provider_annotation("tvm")});
    const d = try b.add(a_id, c_id);
    try b.pop_region();

    try b.push_region("r2", &.{outline.annotation});
    const e = try b.multiply(d, c_id);
    try b.pop_region();

    const func = try b.finish(&.{e});

    const is_kernelized = struct {
        fn f(region: Region) bool {
            return (kernel.requested_providers(region) catch null) != null;
        }
    }.f;

    var iter = func.regions_matching(is_kernelized);
    const r = iter.next().?;
    try std.testing.expectEqualStrings("r1", r.name);
    try std.testing.expect(iter.next() == null);
}

test "empty region not materialized" {
    const kernel = @import("../kernel.zig");
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a_id = try b.param_tensor(.f32, &.{ 2, 2 });
    const c_id = try b.param_tensor(.f32, &.{ 2, 2 });

    try b.push_region("empty", &.{kernel.provider_annotation("tvm")});
    try b.pop_region();

    const d = try b.add(a_id, c_id);
    const func = try b.finish(&.{d});

    try std.testing.expectEqual(@as(usize, 0), func.regions.len);
}

test "use-list basics" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});
    const y = try b.param_tensor(.f32, &.{2});

    // x is unused at this point
    try std.testing.expect(x.is_unused());

    const z = try b.add(x, y);
    // x and y each have one use now
    try std.testing.expect(x.has_one_use());
    try std.testing.expect(y.has_one_use());

    const w = try b.multiply(z, x);
    // x now has two uses (add and multiply)
    try std.testing.expect(x.has_n_uses(2));
    try std.testing.expect(!x.has_one_use());
    // z has one use
    try std.testing.expect(z.has_one_use());

    _ = try b.finish(&.{w});
}

test "FunctionBuilder assigns stable op ids" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});
    const y = try b.param_tensor(.f32, &.{2});

    const z = try b.add(x, y);
    const w = try b.multiply(z, y);
    const func = try b.finish(&.{w});

    try std.testing.expectEqual(@as(u32, 0), func.ops[0].id);
    try std.testing.expectEqual(@as(u32, 1), func.ops[1].id);
    try std.testing.expectEqual(@as(u32, 0), z.defining_op.?.id);
    try std.testing.expectEqual(@as(u32, 1), w.defining_op.?.id);
}

test "Var.replace_all_uses_with" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});
    const y = try b.param_tensor(.f32, &.{2});

    const z = try b.add(x, y);
    _ = try b.multiply(z, x);

    // x has 2 uses. Replace all uses of x with y.
    try std.testing.expect(x.has_n_uses(2));
    @constCast(x).replace_all_uses_with(@constCast(y));
    try std.testing.expect(x.is_unused());
    // y now has the uses that x had plus its original use
    try std.testing.expect(y.has_n_uses(3));
}
