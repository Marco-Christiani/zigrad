//! Program Representation (PR)
//!
//! PR is the compiler's core intermediate representation: a toolchain-neutral
//!  graph of tensor operations. It sits between user-facing tracing (Tensor /
//!  FunctionBuilder) and target-specific lowering (MLIR / StableHLO).
//!
//! ## Key types
//!
//!  1. `Program` - top-level container; owns an arena that backs all IR nodes.
//!  2. `Function` - named function with params, return vars, and a linear op list.
//!  3. `FunctionBuilder` - mutable builder for constructing a Function:
//!      `init`, `emit` ops, and `finish` (validates on `finish`).
//!  4. `Op` - a single operation: typed `Params` union + input `Operand` list +
//!      output `Var` list.
//!  5. `Var` - SSA value with an `Aval` (abstract type) and an intrinsic
//!       doubly-linked use-list.
//!  6. `Operand` - use-chain node linking a consuming Op to the Var it reads.
//!
//! ## Ownership
//!
//! All IR nodes (Var, Op, Operand, param slices) are arena-allocated via
//!  `Program.allocator()`. The arena is freed in bulk by `Program.deinit()`.
//!  Callers never free individual nodes.
//!
//! ## Design invariants
//!
//!  - PR is toolchain-neutral
//!  - Ops are single-output (except `call` / `custom_call` which use
//!     `emit_with_outputs`).
//!  - FunctionBuilder validates eagerly on `finish` via the op registry.
//!  - Slice params (dims, axes, etc.) are duped into the arena by builder
//!     convenience methods so callers can pass stack/temp slices.
const std = @import("std");
const ops = @import("ops/ops.zig");
const pr_log = std.log.scoped(.@"zg/pr");
const Allocator = std.mem.Allocator;

/// Element data type for tensors.
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
    ///  a native float type. Use `encode`/`decode` for value conversion.
    ///
    /// Intended for use inside `inline switch` branches where the tag is
    ///  comptime-known, enabling generic dtype-agnostic code without per-dtype
    ///  function duplication.
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

    /// Convert a float value to this dtype's storage representation.
    pub inline fn encode(comptime self: DType, comptime T: type, val: T) StorageType(self) {
        return switch (self) {
            .f32 => @floatCast(val),
            .f64 => @floatCast(val),
            .bf16 => @intCast(@as(u32, @bitCast(@as(f32, @floatCast(val)))) >> 16),
            .f16 => @bitCast(@as(f16, @floatCast(val))),
            .i32 => @intFromFloat(val),
            else => @compileError("encode not supported for " ++ @tagName(self)),
        };
    }

    /// Decode this dtype's storage representation to a float type.
    pub inline fn decode(comptime self: DType, comptime T: type, raw: StorageType(self)) T {
        return switch (self) {
            .f32 => @floatCast(raw),
            .f64 => @floatCast(raw),
            .bf16 => @floatCast(@as(f32, @bitCast(@as(u32, raw) << 16))),
            .f16 => @floatCast(@as(f16, @bitCast(raw))),
            .i32 => @floatFromInt(raw),
            else => @compileError("decode not supported for " ++ @tagName(self)),
        };
    }
};

/// Heap-backed shape (slice into arena memory).
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
pub const BoundedShape = struct {
    /// TODO: use of i64 here looks odd, yes, but third party libs want i64,
    ///  the remaining concern is us casting to usize, consider i32
    ///  and cast at boundary...? also, this will ripple. Not worth it rn.
    buf: [max_rank]i64 = undefined,
    // TODO: oh yeah we are also wasting a field here tbh. so this could be
    //  better in many ways, but I wonder if this should exist at all?
    //  improving this will quickly start to balloon and we would have to
    //  move it out, so im defering this, its a micro-optimization anyways.
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

        return result.toOwnedSlice(allocator);
    }
};

/// Abstract value: the type of a Var without its data.
///
/// ## ADR
/// Currently only `tensor` (dtype + shape). The union exists so PR can
///  support non-tensor types in the future without changing the Var layout.
pub const Aval = union(enum) {
    /// TODO: rename this to reduce confusion
    tensor: Tensor,

    /// Extract the tensor type. Exhaustive over Aval variants.
    pub fn as_tensor(self: Aval) Tensor {
        return switch (self) {
            .tensor => |t| t,
        };
    }
};

/// Tensor type descriptor (dtype + shape).
///
/// This is PR's notion of a tensor *type*, not a runtime tensor, so it carries
///  no data, no device, no buffer.
/// TODO: rename this to reduce confusion
pub const Tensor = struct {
    dtype: DType,
    shape: Shape,
};

/// Typed scalar constant.
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

    /// Create a typed scalar literal by converting from f64.
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
            // TODO: what about nan? check on this.
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

// ============================================================================
// Param Structs
// ============================================================================

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

pub const ReduceParams = struct {
    axes: []const i64,
};

pub const CallParams = struct {
    callee: []const u8,
};

/// Parameters for an opaque backend-dispatched custom_call operation.
pub const CustomCallParams = struct {
    /// Backend dispatch key used at runtime (e.g. "zigrad.kernel.matmul_add").
    target_name: []const u8,
    has_side_effect: bool,
    /// Output types must be provided explicitly (not inferrable from inputs).
    out_avals: []const Aval,
    /// Optional: identifies a pre-compiled kernel artifact in the kernel store.
    kernel_key: ?[]const u8 = null,
    /// Optional: which kernel provider compiled this (e.g. "tvm", "mirage").
    provider_name: ?[]const u8 = null,
};

// ============================================================================
// Params Tagged Union (keyed by Prim)
// ============================================================================

/// Typed parameter union keyed by Prim. Replaces the heterogeneous `[]const Param`
/// list and `pr.param(.tag, slice)` linear scan pattern.
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
    dot_general: DotGeneralParams,
    reshape: ReshapeParams,
    iota: IotaParams,
    broadcast_in_dim: BroadcastInDimParams,
    transpose: TransposeParams,
    slice: SliceParams,
    concatenate: ConcatenateParams,
    reduce_sum: ReduceParams,
    reduce_max: ReduceParams,
    call: CallParams,
    custom_call: CustomCallParams,
};

// ============================================================================
// Core IR Types
// ============================================================================

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

// ============================================================================
// Annotations & Regions
// ============================================================================

/// Steering annotation attached to a region of ops.
/// Does not change semantics -- only compiler hints.
pub const Annotation = struct {
    /// Request that ops be outlined into a separate call boundary.
    outline: bool = false,
    /// Request kernelization by a named provider (e.g. "tvm").
    /// TODO: consider a rename
    kernelize: ?[]const u8 = null,
};

/// A contiguous range of ops sharing an annotation.
/// Materialized at `FunctionBuilder.finish()` from the annotation stack by default.
pub const Region = struct {
    name: []const u8,
    annotation: Annotation,
    /// Index of the first op in this region.
    op_start: u32,
    /// Number of ops in this region.
    op_len: u32,
};

// ============================================================================
// Core Data Structures
// ============================================================================

/// A named function in the program: parameter vars, a linear op sequence,
///  return vars, and optional annotation regions.
///
/// Constructed by `FunctionBuilder.finish()`, which validates all ops.
/// Functions are immutable once built, modification requires rebuilding.
/// TODO: do we want to stick to the immutable model? after def-use additions,
///  we effectively have a hybrid model
pub const Function = struct {
    name: []const u8,
    params: []*Var,
    returns: []*Var,
    ops: []*Op,
    regions: []const Region,
    var_count: u32,

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

/// Top-level container for a PR program.
///
/// Owns an arena allocator that backs all IR nodes (Vars, Ops, Operands,
///  param slices, Functions). Call `deinit` to free everything in bulk.
///  Individual nodes are never freed separately.
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

    /// NOTE: this is currently out of any hot path, but not ideal design anymore,
    ///  many call sites use the `functions` field so an update will trigger a refactor
    ///  but this is likely to land at some point.
    /// TODO: where do we check for dupe names? it happens, i think pipeline or lowering,
    ///  but probably add checks around here
    pub fn add_function(self: *Program, func: Function) Allocator.Error!void {
        const a = self.allocator();
        const new_items = try a.alloc(Function, self.functions.len + 1);
        @memcpy(new_items[0..self.functions.len], self.functions);
        new_items[self.functions.len] = func;
        self.functions = new_items;
    }

    /// Look up a function by name.
    pub fn get_function(self: *const Program, name: []const u8) ?Function {
        for (self.functions) |f| {
            if (std.mem.eql(u8, f.name, name)) return f;
        }
        return null;
    }

    /// Number of input parameters for a named function.
    pub fn input_arity(self: *const Program, name: []const u8) usize {
        const f = self.get_function(name) orelse return 0;
        return f.params.len;
    }

    /// Number of output values for a named function.
    pub fn output_arity(self: *const Program, name: []const u8) usize {
        const f = self.get_function(name) orelse return 0;
        return f.returns.len;
    }

    pub fn deinit(self: *Program) void {
        self.arena.deinit();
    }
};

// ============================================================================
// Errors
// ============================================================================

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
    DotGeneralTypeMismatch,
    ReshapeTypeMismatch,
    BroadcastInDimTypeMismatch,
    TransposeTypeMismatch,
    SliceTypeMismatch,
    ConcatTypeMismatch,
    ReduceSumTypeMismatch,
    ReduceMaxTypeMismatch,
    CallUnresolvedCallee,
    CallArityMismatch,
    CallTypeMismatch,
    CustomCallTypeMismatch,
    IotaTypeMismatch,
    DuplicateFunctionName,
    ScatterAddTypeMismatch,
};

/// Errors from FunctionBuilder: validation failures (caught eagerly at
///  `finish`) plus allocation failures from the program arena.
pub const BuildError = ValidationError || Allocator.Error;

// ============================================================================
// Validation
// ============================================================================

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
        // validate each op
        try validate_ops_in_func(func);
        // validate calls
        for (func.ops) |op| {
            if (op.prim() != .call) continue;

            const call_params = op.params.call;
            const callee_name = call_params.callee;

            var callee: ?Function = null;
            for (program.functions) |candidate| {
                if (std.mem.eql(u8, candidate.name, callee_name)) {
                    callee = candidate;
                    break;
                }
            }
            const callee_func = callee orelse {
                pr_log.err("call references unknown function '{s}' in '{s}'", .{ callee_name, func.name });
                return error.CallUnresolvedCallee;
            };

            if (op.inputs.len != callee_func.params.len or op.outputs.len != callee_func.returns.len) {
                pr_log.err("call arity mismatch: '{s}' expects {d} inputs/{d} outputs, got {d}/{d}", .{
                    callee_name,
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

// ============================================================================
// FunctionBuilder
// ============================================================================

/// Annotation stack entry for tracking active regions during building.
const RegionEntry = struct {
    name: []const u8,
    annotation: Annotation,
    op_start: u32,
};

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
    region_stack: std.ArrayList(RegionEntry),
    completed_regions: std.ArrayList(Region),

    pub fn init(program: *Program, name: []const u8) BuildError!FunctionBuilder {
        const a = program.allocator();
        return .{
            .program = program,
            .name = name,
            .params_list = try std.ArrayList(*Var).initCapacity(a, 8),
            .ops_list = try std.ArrayList(*Op).initCapacity(a, 16),
            .next_var_id = 0,
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

    fn create_var(self: *FunctionBuilder, aval: Aval) BuildError!*Var {
        const a = self.alloc();
        const v = try a.create(Var);
        v.* = .{ .id = self.next_id(), .aval = aval };
        return v;
    }

    // ====================================================================
    // Region API
    // ====================================================================

    /// Push a named annotation region.
    /// Ops emitted after this call belong to this region until pop_region is called.
    pub fn push_region(self: *FunctionBuilder, name: []const u8, annotation: Annotation) BuildError!void {
        const a = self.alloc();
        try self.region_stack.append(a, .{
            .name = name,
            .annotation = annotation,
            .op_start = @intCast(self.ops_list.items.len),
        });
    }

    /// Pop the most recent annotation region.
    pub fn pop_region(self: *FunctionBuilder) BuildError!void {
        const a = self.alloc();
        const entry = self.region_stack.pop() orelse return;
        const op_end: u32 = @intCast(self.ops_list.items.len);
        if (op_end > entry.op_start) {
            try self.completed_regions.append(a, .{
                .name = entry.name,
                .annotation = entry.annotation,
                .op_start = entry.op_start,
                .op_len = op_end - entry.op_start,
            });
        }
    }

    // ====================================================================
    // Core Emit
    // ====================================================================

    /// Emit a single-output op, inferring the output type via the op registry.
    ///
    /// Input vars have their use-lists updated: each input gains an Operand
    ///  node linking it to the new op.
    pub fn emit(self: *FunctionBuilder, params: Params, inputs: []const *Var) BuildError!*Var {
        const a = self.alloc();

        const out_aval = try ops.infer_output(a, params, inputs);
        const out_var = try self.create_var(out_aval);

        const operands = try a.alloc(Operand, inputs.len);
        const op = try a.create(Op);
        const out_slice = try a.alloc(*Var, 1);
        out_slice[0] = out_var;

        op.* = .{ .inputs = operands, .outputs = out_slice, .params = params };
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

        try self.ops_list.append(a, op);
        return out_var;
    }

    /// Emit a multi-output op with explicit output types.
    ///
    /// Input vars have their use-lists updated (see `emit`).
    fn emit_with_outputs(self: *FunctionBuilder, params: Params, inputs: []const *Var, out_avals: []const Aval) BuildError![]*Var {
        const a = self.alloc();

        const out_vars = try a.alloc(*Var, out_avals.len);
        for (out_avals, 0..) |aval, i| {
            out_vars[i] = try self.create_var(aval);
        }

        const operands = try a.alloc(Operand, inputs.len);
        const op = try a.create(Op);

        op.* = .{ .inputs = operands, .outputs = out_vars, .params = params };
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

        try self.ops_list.append(a, op);
        return out_vars;
    }

    pub fn param_tensor(self: *FunctionBuilder, dtype: DType, dims: []const i64) BuildError!*Var {
        const a = self.alloc();
        const dims_copy = try a.dupe(i64, dims);
        const v = try self.create_var(.{ .tensor = .{ .dtype = dtype, .shape = .{ .dims = dims_copy } } });
        try self.params_list.append(a, v);
        return v;
    }

    pub fn literal_scalar(self: *FunctionBuilder, value: Literal) BuildError!*Var {
        return self.emit(.{ .literal = value }, &.{});
    }

    /// Emit a scalar constant of `dtype` from an `f64` value.
    ///
    /// Handles dtype-aware conversion from f64. Prefer this over the two-step
    /// `Literal.from_f64` + `literal_scalar` pattern.
    pub fn scalar(self: *FunctionBuilder, dtype: DType, val: f64) BuildError!*Var {
        return self.literal_scalar(Literal.from_f64(dtype, val));
    }

    pub fn add(self: *FunctionBuilder, lhs: *Var, rhs: *Var) BuildError!*Var {
        return self.emit(.{ .add = {} }, &.{ lhs, rhs });
    }

    pub fn subtract(self: *FunctionBuilder, lhs: *Var, rhs: *Var) BuildError!*Var {
        return self.emit(.{ .subtract = {} }, &.{ lhs, rhs });
    }

    pub fn multiply(self: *FunctionBuilder, lhs: *Var, rhs: *Var) BuildError!*Var {
        return self.emit(.{ .multiply = {} }, &.{ lhs, rhs });
    }

    pub fn divide(self: *FunctionBuilder, lhs: *Var, rhs: *Var) BuildError!*Var {
        return self.emit(.{ .divide = {} }, &.{ lhs, rhs });
    }

    pub fn maximum(self: *FunctionBuilder, lhs: *Var, rhs: *Var) BuildError!*Var {
        return self.emit(.{ .maximum = {} }, &.{ lhs, rhs });
    }

    pub fn exp(self: *FunctionBuilder, operand_var: *Var) BuildError!*Var {
        return self.emit(.{ .exp = {} }, &.{operand_var});
    }

    pub fn log(self: *FunctionBuilder, operand_var: *Var) BuildError!*Var {
        return self.emit(.{ .log = {} }, &.{operand_var});
    }

    pub fn rsqrt(self: *FunctionBuilder, operand_var: *Var) BuildError!*Var {
        return self.emit(.{ .rsqrt = {} }, &.{operand_var});
    }

    pub fn logistic(self: *FunctionBuilder, operand_var: *Var) BuildError!*Var {
        return self.emit(.{ .logistic = {} }, &.{operand_var});
    }

    pub fn compare(self: *FunctionBuilder, lhs: *Var, rhs: *Var, cmp: CompareParams) BuildError!*Var {
        return self.emit(.{ .compare = cmp }, &.{ lhs, rhs });
    }

    pub fn select(self: *FunctionBuilder, cond: *Var, on_true: *Var, on_false: *Var) BuildError!*Var {
        return self.emit(.{ .select = {} }, &.{ cond, on_true, on_false });
    }

    pub fn convert(self: *FunctionBuilder, operand_var: *Var, out_dtype: DType) BuildError!*Var {
        // Short circuit converting to the same dtype is a no-op.
        // Save callers from needing to guard with `if (a.dtype == b.dtype)` everywhere.
        // Backend would fold these away anyway, but keeping the IR clean at construction time
        //  helps denoise IR dumps saves AD passes from walking dead converts if no DCE pass.
        if (operand_var.as_tensor().dtype == out_dtype) return operand_var;
        return self.emit(.{ .convert = out_dtype }, &.{operand_var});
    }

    pub fn reduce_max(self: *FunctionBuilder, operand_var: *Var, axes: []const i64) BuildError!*Var {
        const axes_copy = try self.alloc().dupe(i64, axes);
        return self.emit(.{ .reduce_max = .{ .axes = axes_copy } }, &.{operand_var});
    }

    pub fn gather(self: *FunctionBuilder, operand_var: *Var, indices: *Var, gp: GatherParams) BuildError!*Var {
        const a = self.alloc();
        return self.emit(.{ .gather = .{
            .slice_sizes = try a.dupe(i64, gp.slice_sizes),
            .offset_dims = try a.dupe(i64, gp.offset_dims),
            .collapsed_slice_dims = try a.dupe(i64, gp.collapsed_slice_dims),
            .start_index_map = try a.dupe(i64, gp.start_index_map),
            .index_vector_dim = gp.index_vector_dim,
        } }, &.{ operand_var, indices });
    }

    pub fn scatter(self: *FunctionBuilder, input: *Var, indices: *Var, updates: *Var, sp: ScatterParams) BuildError!*Var {
        const a = self.alloc();
        return self.emit(.{ .scatter = .{
            .update_window_dims = try a.dupe(i64, sp.update_window_dims),
            .inserted_window_dims = try a.dupe(i64, sp.inserted_window_dims),
            .scatter_dims_to_operand_dims = try a.dupe(i64, sp.scatter_dims_to_operand_dims),
            .index_vector_dim = sp.index_vector_dim,
            .reduction = sp.reduction,
        } }, &.{ input, indices, updates });
    }

    pub fn slice(self: *FunctionBuilder, operand_var: *Var, sp: SliceParams) BuildError!*Var {
        const a = self.alloc();
        return self.emit(.{ .slice = .{
            .start_indices = try a.dupe(i64, sp.start_indices),
            .limit_indices = try a.dupe(i64, sp.limit_indices),
            .strides = try a.dupe(i64, sp.strides),
        } }, &.{operand_var});
    }

    pub fn concatenate(self: *FunctionBuilder, operands: []const *Var, axis: i64) BuildError!*Var {
        return self.emit(.{ .concatenate = .{ .axis = axis } }, operands);
    }

    pub fn dot_general(self: *FunctionBuilder, lhs: *Var, rhs: *Var, dg: DotGeneralParams) BuildError!*Var {
        const a = self.alloc();
        return self.emit(.{ .dot_general = .{
            .lhs_batch_dims = try a.dupe(i64, dg.lhs_batch_dims),
            .rhs_batch_dims = try a.dupe(i64, dg.rhs_batch_dims),
            .lhs_contracting_dims = try a.dupe(i64, dg.lhs_contracting_dims),
            .rhs_contracting_dims = try a.dupe(i64, dg.rhs_contracting_dims),
        } }, &.{ lhs, rhs });
    }

    pub fn dot(self: *FunctionBuilder, lhs: *Var, rhs: *Var) BuildError!*Var {
        return self.emit(.{ .dot = {} }, &.{ lhs, rhs });
    }

    pub fn iota(self: *FunctionBuilder, out_dtype: DType, out_dims: []const i64, iota_dim: i64) BuildError!*Var {
        const a = self.alloc();
        return self.emit(.{ .iota = .{
            .out_shape = try a.dupe(i64, out_dims),
            .out_dtype = out_dtype,
            .dimension = iota_dim,
        } }, &.{});
    }

    pub fn reshape(self: *FunctionBuilder, operand_var: *Var, out_dims: []const i64) BuildError!*Var {
        const a = self.alloc();
        return self.emit(.{ .reshape = .{
            .out_shape = try a.dupe(i64, out_dims),
        } }, &.{operand_var});
    }

    pub fn broadcast_in_dim(self: *FunctionBuilder, operand_var: *Var, out_dims: []const i64, broadcast_dimensions: []const i64) BuildError!*Var {
        const a = self.alloc();
        return self.emit(.{ .broadcast_in_dim = .{
            .out_shape = try a.dupe(i64, out_dims),
            .dimensions = try a.dupe(i64, broadcast_dimensions),
        } }, &.{operand_var});
    }

    /// Create a scalar constant of `dtype` with value `val`, broadcast to `dims`.
    /// Returns a scalar if `dims` is empty.
    pub fn scalar_broadcast(self: *FunctionBuilder, dtype: DType, dims: []const i64, val: f64) BuildError!*Var {
        const lit = try self.scalar(dtype, val);
        if (dims.len == 0) return lit;
        return self.broadcast_in_dim(lit, dims, &.{});
    }

    pub fn transpose(self: *FunctionBuilder, operand_var: *Var, permutation: []const i64) BuildError!*Var {
        const a = self.alloc();
        return self.emit(.{ .transpose = .{
            .permutation = try a.dupe(i64, permutation),
        } }, &.{operand_var});
    }

    pub fn reduce_sum(self: *FunctionBuilder, operand_var: *Var, axes: []const i64) BuildError!*Var {
        const a = self.alloc();
        return self.emit(.{ .reduce_sum = .{
            .axes = try a.dupe(i64, axes),
        } }, &.{operand_var});
    }

    pub fn custom_call(self: *FunctionBuilder, target: []const u8, operands: []const *Var, out_like: *Var) BuildError!*Var {
        const a = self.alloc();
        const out_aval = out_like.aval;

        const target_copy = try a.dupe(u8, target);
        const out_avals = try a.alloc(Aval, 1);
        out_avals[0] = out_aval;
        return self.emit(.{ .custom_call = .{
            .target_name = target_copy,
            .has_side_effect = false,
            .out_avals = out_avals,
        } }, operands);
    }

    /// Emit a multi-output custom_call. Used by kernelize pass.
    pub fn custom_call_multi(self: *FunctionBuilder, cc_params: CustomCallParams, inputs: []const *Var) BuildError![]*Var {
        return self.emit_with_outputs(.{ .custom_call = cc_params }, inputs, cc_params.out_avals);
    }

    pub fn call(self: *FunctionBuilder, callee: []const u8, inputs: []const *Var) BuildError![]*Var {
        var callee_func: ?Function = null;
        for (self.program.functions) |func| {
            if (std.mem.eql(u8, func.name, callee)) {
                callee_func = func;
                break;
            }
        }
        const callee_fn = callee_func orelse {
            pr_log.err("call references unknown function '{s}'", .{callee});
            return error.CallUnresolvedCallee;
        };

        if (inputs.len != callee_fn.params.len) {
            pr_log.err("call arity mismatch: '{s}' expects {d} inputs, got {d}", .{ callee, callee_fn.params.len, inputs.len });
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

        const callee_copy = try a.dupe(u8, callee);
        return self.emit_with_outputs(.{ .call = .{ .callee = callee_copy } }, inputs, out_avals);
    }

    // ====================================================================
    // Finish
    // ====================================================================

    /// Finalize the function: flush open regions, validate all ops, and
    /// return an immutable Function. The builder should not be used after this
    /// (call `deinit` to release its working lists).
    pub fn finish(self: *FunctionBuilder, returns: []const *Var) BuildError!Function {
        const a = self.alloc();

        // Flush any remaining open regions
        while (self.region_stack.items.len > 0) {
            try self.pop_region();
        }

        const func = Function{
            .name = self.name,
            .params = try self.params_list.toOwnedSlice(a),
            .returns = try a.dupe(*Var, returns),
            .ops = try self.ops_list.toOwnedSlice(a),
            .regions = try self.completed_regions.toOwnedSlice(a),
            .var_count = self.next_var_id,
        };
        try validate_ops_in_func(func);
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

test "FunctionBuilder reduce_sum basic" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.reduce_sum(x, &.{0});
    const func = try b.finish(&.{y});
    try validate_ops_in_func(func);
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
    try std.testing.expectEqual(@as(u32, 0), func.regions[0].op_start);
    try std.testing.expectEqual(@as(u32, 2), func.regions[0].op_len);
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
    try std.testing.expectEqual(@as(u32, 1), func.regions[0].op_start);
    try std.testing.expectEqual(@as(u32, 1), func.regions[0].op_len);
    // Outer region completed second
    try std.testing.expectEqualStrings("outer", func.regions[1].name);
    try std.testing.expectEqualStrings("tvm", func.regions[1].annotation.kernelize.?);
    try std.testing.expectEqual(@as(u32, 0), func.regions[1].op_start);
    try std.testing.expectEqual(@as(u32, 3), func.regions[1].op_len);
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

test "no external imports" {
    // This test verifies at compile time that this module depends only on std.
    const pr = @This();
    try std.testing.expect(pr.DType.f32 == .f32);
}
