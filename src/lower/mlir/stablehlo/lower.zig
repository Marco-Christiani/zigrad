//! StableHLO Lowering
//!
//! StableHLO-specific op translation for the PR -> MLIR lowering pipeline.
//! Provides the `lower_op` callback that maps each PR op to StableHLO MLIR
//!  ops, plus public entry points that wire this callback into the
//!  dialect-agnostic scaffold from `context.zig`.
//!
//! Provides:
//!  1. `lower_program_to_mlir`: Direct lowering API (PR program -> StableHLO MLIR bytes)
//!  2. `lower_function_to_mlir`: Single-function convenience wrapper
//!  3. `lower_pass` / `lower_pass_with_config`: Pass-based pipeline integration
const std = @import("std");

const pr = @import("../../../pr/pr.zig");
const mlir = @import("../../../c/mlir/mlir.zig");
const stablehlo = @import("../../../c/mlir/dialects/stablehlo.zig");
const pass = @import("../../../pipeline/pass.zig");
const MlirSession = @import("../session.zig").MlirSession;
const context = @import("../context.zig");
const log = std.log.scoped(.@"zg/lower_stablehlo");

const LowerError = context.LowerError;
const LowerContext = context.LowerContext;
const OutputFormat = context.OutputFormat;

const zigrad_kernel_call_op_name = context.zigrad_kernel_call_op_name;

const lower_types = @import("../../types.zig");

// ============================================================================
// Core Lowering Implementation
// ============================================================================

/// Lower a PR program to StableHLO MLIR bytecode or text.
///
/// Creates an MLIR session with StableHLO registered and delegates to the
///  dialect-agnostic scaffold with the StableHLO `lower_op` callback.
pub fn lower_program_to_mlir(
    allocator: std.mem.Allocator,
    program: *const pr.Program,
    entry_name: ?[]const u8,
    out: OutputFormat,
) LowerError![]u8 {
    var session = try MlirSession.init();
    defer session.deinit();
    session.load_dialect("stablehlo");
    return context.lower_program_to_mlir(allocator, session, program, entry_name, out, lower_op);
}

pub fn lower_function_to_mlir(allocator: std.mem.Allocator, func: pr.Function, out: OutputFormat) LowerError![]u8 {
    var program = pr.Program.init(allocator);
    defer program.deinit();

    try program.add_function(func);
    return lower_program_to_mlir(allocator, &program, func.name, out);
}

// ============================================================================
// Per-Op Lowering (switch dispatch)
// ============================================================================

fn lower_op(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    switch (op.prim()) {
        // Binary elementwise
        .add => lower_binary(stablehlo.add, ctx, op),
        .subtract => lower_binary(stablehlo.subtract, ctx, op),
        .multiply => lower_binary(stablehlo.multiply, ctx, op),
        .divide => lower_binary(stablehlo.divide, ctx, op),
        .maximum => lower_binary(stablehlo.maximum, ctx, op),
        // Unary
        .exp => lower_unary(stablehlo.exponential, ctx, op),
        .log => lower_unary(stablehlo.log, ctx, op),
        .rsqrt => lower_unary(stablehlo.rsqrt, ctx, op),
        .logistic => lower_unary(stablehlo.logistic, ctx, op),
        // Type conversion
        .convert => try lower_convert(ctx, op),
        // Constant
        .literal => lower_literal(ctx, op),
        // Shape
        .reshape => try lower_reshape(ctx, op),
        .transpose => try lower_transpose(ctx, op),
        .broadcast_in_dim => try lower_broadcast_in_dim(ctx, op),
        .iota => lower_iota(ctx, op),
        .slice => try lower_slice(ctx, op),
        .concatenate => try lower_concatenate(ctx, op),
        // Reduction
        .reduce_sum => try lower_reduce(ctx, op, .sum),
        .reduce_max => try lower_reduce(ctx, op, .max),
        // Contraction
        .dot => try lower_dot(ctx, op),
        .dot_general => try lower_dot_general(ctx, op),
        // Compare
        .compare => try lower_compare(ctx, op),
        .select => try lower_select(ctx, op),
        // Structured
        .gather => try lower_gather(ctx, op),
        .scatter => try lower_scatter(ctx, op),
        // Special - Dialect-agnostic, delegated.
        .call => try context.lower_call(ctx, op),
        // TODO: see
        //  1. https://openxla.org/stablehlo/spec#custom_call
        //  2. https://openxla.org/stablehlo/spec#xla_gpu_support_special_custom_call_targets
        //  2. https://openxla.org/stablehlo/spec#alias
        .custom_call => try context.lower_custom_call(ctx, op),
    }
}

// --- Binary elementwise ---

fn lower_binary(
    comptime op_fn: fn (mlir.Context, mlir.Value, mlir.Value, mlir.Location) mlir.Operation,
    ctx: LowerContext,
    op: *const pr.Op,
) void {
    const lhs = ctx.get_value(op.operand(0)).?;
    const rhs = ctx.get_value(op.operand(1)).?;
    const mlir_op = op_fn(ctx.mlir_ctx, lhs, rhs, ctx.loc);
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

// --- Unary ---

fn lower_unary(
    comptime op_fn: fn (mlir.Context, mlir.Value, mlir.Location) mlir.Operation,
    ctx: LowerContext,
    op: *const pr.Op,
) void {
    const operand = ctx.get_value(op.operand(0)).?;
    const mlir_op = op_fn(ctx.mlir_ctx, operand, ctx.loc);
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn lower_convert(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const operand = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const out_tensor = op.result(0).aval.as_tensor();
    const out_type = ctx.tensor_to_mlir_type(out_tensor);
    const mlir_op = stablehlo.convert(ctx.mlir_ctx, operand, out_type, ctx.loc);
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

// --- Constant ---

fn lower_literal(ctx: LowerContext, op: *const pr.Op) void {
    const out_var = op.result(0);
    const out_tensor = out_var.aval.as_tensor();
    const lit = op.params.literal;
    const elem_type = dtype_to_dense_elements_type(out_tensor.dtype);
    const raw_bytes = switch (lit) {
        inline else => |v| std.mem.asBytes(&v),
    };
    const mlir_op = stablehlo.constant(ctx.mlir_ctx, &.{}, elem_type, raw_bytes, ctx.loc);
    ctx.block.append_operation(mlir_op);
    ctx.set_value(out_var, mlir_op.result(0));
}

// --- Shape ops ---

fn lower_reshape(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const operand = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const out_tensor = op.result(0).aval.as_tensor();
    const out_type = ctx.tensor_to_mlir_type(out_tensor);
    const mlir_op = stablehlo.reshape(ctx.mlir_ctx, operand, out_type, ctx.loc);
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn lower_transpose(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const perm = op.params.transpose.permutation;
    const operand = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const out_tensor = op.result(0).aval.as_tensor();
    const out_type = ctx.tensor_to_mlir_type(out_tensor);
    const mlir_op = stablehlo.transpose(ctx.mlir_ctx, operand, out_type, ctx.loc, .{ .permutation = perm });
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn lower_broadcast_in_dim(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const bd = op.params.broadcast_in_dim.dimensions;
    const operand = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const out_tensor = op.result(0).aval.as_tensor();
    const out_type = ctx.tensor_to_mlir_type(out_tensor);
    const mlir_op = stablehlo.broadcast_in_dim(ctx.mlir_ctx, operand, bd, out_type, ctx.loc);
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn lower_iota(ctx: LowerContext, op: *const pr.Op) void {
    const iota_dim = op.params.iota.dimension;
    const out_tensor = op.result(0).aval.as_tensor();
    const out_type = ctx.tensor_to_mlir_type(out_tensor);
    const mlir_op = stablehlo.iota(ctx.mlir_ctx, iota_dim, out_type, ctx.loc);
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn lower_slice(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const sparams = op.params.slice;
    const operand = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const out_tensor = op.result(0).aval.as_tensor();
    const out_type = ctx.tensor_to_mlir_type(out_tensor);
    const mlir_op = stablehlo.slice(
        ctx.mlir_ctx,
        operand,
        sparams.start_indices,
        sparams.limit_indices,
        sparams.strides,
        out_type,
        ctx.loc,
    );
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn lower_concatenate(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const axis = op.params.concatenate.axis;
    const values = try ctx.arena.alloc(mlir.Value, op.inputs.len);
    for (op.inputs, 0..) |operand, i| {
        values[i] = ctx.get_value(operand.value) orelse return error.InvalidProgram;
    }
    const mlir_op = stablehlo.concatenate(ctx.mlir_ctx, values, axis, ctx.loc);
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

// --- Reduction ---

const ReduceKind = enum { sum, max };

fn lower_reduce(ctx: LowerContext, op: *const pr.Op, kind: ReduceKind) LowerError!void {
    const axes = switch (op.prim()) {
        .reduce_sum => op.params.reduce_sum.axes,
        .reduce_max => op.params.reduce_max.axes,
        else => return error.InvalidProgram,
    };
    const operand = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const out_tensor = op.result(0).aval.as_tensor();
    const elem_type = dtype_to_dense_elements_type(out_tensor.dtype);

    switch (kind) {
        .sum => {
            const zero_bytes = scalar_zero_bytes(out_tensor.dtype);
            const zero_op = stablehlo.constant(ctx.mlir_ctx, &.{}, elem_type, zero_bytes, ctx.loc);
            ctx.block.append_operation(zero_op);
            const mlir_op = stablehlo.reduce(ctx.mlir_ctx, &.{operand}, &.{zero_op.result(0)}, axes, {}, reduce_add_block, ctx.loc);
            ctx.block.append_operation(mlir_op);
            ctx.set_value(op.result(0), mlir_op.result(0));
        },
        .max => {
            const min_bytes = scalar_min_bytes(out_tensor.dtype);
            const min_op = stablehlo.constant(ctx.mlir_ctx, &.{}, elem_type, min_bytes, ctx.loc);
            ctx.block.append_operation(min_op);
            const mlir_op = stablehlo.reduce(ctx.mlir_ctx, &.{operand}, &.{min_op.result(0)}, axes, {}, reduce_max_block, ctx.loc);
            ctx.block.append_operation(mlir_op);
            ctx.set_value(op.result(0), mlir_op.result(0));
        },
    }
}

fn reduce_add_block(_: anytype, ctx: mlir.Context, ins: []const mlir.Value, accs: []const mlir.Value) mlir.Operation {
    return stablehlo.add(ctx, ins[0], accs[0], mlir.Location.unknown(ctx));
}

fn reduce_max_block(_: anytype, ctx: mlir.Context, ins: []const mlir.Value, accs: []const mlir.Value) mlir.Operation {
    return stablehlo.maximum(ctx, ins[0], accs[0], mlir.Location.unknown(ctx));
}

// --- Contraction ---

fn lower_dot(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const lhs = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const rhs = ctx.get_value(op.operand(1)) orelse return error.InvalidProgram;
    const out_tensor = op.result(0).aval.as_tensor();
    const out_type = ctx.tensor_to_mlir_type(out_tensor);
    const mlir_op = stablehlo.dot_general(ctx.mlir_ctx, lhs, rhs, out_type, ctx.loc, .{
        .lhs_batching_dimensions = &.{},
        .rhs_batching_dimensions = &.{},
        .lhs_contracting_dimensions = &.{1},
        .rhs_contracting_dimensions = &.{0},
        .precision = .fast,
    });
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn lower_dot_general(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const dg = op.params.dot_general;
    const lhs = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const rhs = ctx.get_value(op.operand(1)) orelse return error.InvalidProgram;
    const out_tensor = op.result(0).aval.as_tensor();
    const out_type = ctx.tensor_to_mlir_type(out_tensor);
    const mlir_op = stablehlo.dot_general(ctx.mlir_ctx, lhs, rhs, out_type, ctx.loc, .{
        .lhs_batching_dimensions = dg.lhs_batch_dims,
        .rhs_batching_dimensions = dg.rhs_batch_dims,
        .lhs_contracting_dimensions = dg.lhs_contracting_dims,
        .rhs_contracting_dimensions = dg.rhs_contracting_dims,
        .precision = .fast,
    });
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

// --- Compare ---

fn lower_compare(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const cparams = op.params.compare;
    const lhs = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const rhs = ctx.get_value(op.operand(1)) orelse return error.InvalidProgram;
    const mlir_op = stablehlo.compare(
        ctx.mlir_ctx,
        lhs,
        rhs,
        stablehlo.ComparisonDirection.init(ctx.mlir_ctx, map_compare_direction(cparams.direction)),
        stablehlo.CompareType.init(ctx.mlir_ctx, map_compare_type(cparams.compare_type)),
        ctx.loc,
    );
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn lower_select(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const cond = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const on_true = ctx.get_value(op.operand(1)) orelse return error.InvalidProgram;
    const on_false = ctx.get_value(op.operand(2)) orelse return error.InvalidProgram;
    const mlir_op = stablehlo.select(ctx.mlir_ctx, cond, on_true, on_false, ctx.loc);
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn map_compare_direction(dir: pr.CompareDirection) stablehlo.ComparisonDirection.Direction {
    return switch (dir) {
        .EQ => .EQ,
        .NE => .NE,
        .GE => .GE,
        .GT => .GT,
        .LE => .LE,
        .LT => .LT,
    };
}

fn map_compare_type(ctype: pr.CompareType) stablehlo.CompareType.Type {
    return switch (ctype) {
        .SIGNED => .SIGNED,
        .UNSIGNED => .UNSIGNED,
        .FLOAT => .FLOAT,
        .TOTALORDER => .TOTALORDER,
    };
}

// --- Structured (gather/scatter) ---

fn lower_gather(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const gparams = op.params.gather;
    const operand = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const indices = ctx.get_value(op.operand(1)) orelse return error.InvalidProgram;
    const mlir_op = stablehlo.gather(ctx.mlir_ctx, operand, indices, gparams.slice_sizes, ctx.loc, .{
        .offset_dims = gparams.offset_dims,
        .collapsed_slice_dims = gparams.collapsed_slice_dims,
        .operand_batching_dims = &.{},
        .start_indices_batching_dims = &.{},
        .start_index_map = gparams.start_index_map,
        .index_vector_dim = gparams.index_vector_dim,
    });
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn lower_scatter(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const sparams = op.params.scatter;
    const operand = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const indices = ctx.get_value(op.operand(1)) orelse return error.InvalidProgram;
    const updates = ctx.get_value(op.operand(2)) orelse return error.InvalidProgram;
    const update_block = try make_update_block(ctx.mlir_ctx, operand.get_type(), ctx.loc, sparams.reduction);
    const mlir_op = stablehlo.scatter(
        ctx.mlir_ctx,
        &.{operand},
        &.{indices},
        &.{updates},
        update_block,
        .{
            .update_window_dims = sparams.update_window_dims,
            .inserted_window_dims = sparams.inserted_window_dims,
            .input_batching_dims = &.{},
            .scatter_indices_batching_dims = &.{},
            .scatter_dims_to_operand_dims = sparams.scatter_dims_to_operand_dims,
            .index_vector_dim = sparams.index_vector_dim,
        },
        ctx.loc,
    );
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn make_update_block(ctx: mlir.Context, operand_type: mlir.Type, loc: mlir.Location, reduction: pr.ScatterReduction) error{InvalidMlir}!mlir.Block {
    const elem_type = if (operand_type.as(mlir.RankedTensorType)) |shaped| shaped.get_element_type() else operand_type;
    const arg_type: mlir.Type = .tensor(&.{}, elem_type);
    var block = try mlir.Block.init(&.{ arg_type, arg_type }, &.{ loc, loc });
    const mlir_op = switch (reduction) {
        .add => stablehlo.add(ctx, block.argument(0), block.argument(1), loc),
        .max => stablehlo.maximum(ctx, block.argument(0), block.argument(1), loc),
        .min => stablehlo.minimum(ctx, block.argument(0), block.argument(1), loc),
        .mul => stablehlo.multiply(ctx, block.argument(0), block.argument(1), loc),
    };
    block.append_operation(mlir_op);
    const ret = stablehlo.return_(ctx, mlir_op.result(0), loc);
    block.append_operation(ret);
    return block;
}

// ============================================================================
// StableHLO-specific Type Mapping Helpers
// ============================================================================

// TODO: this doesnt belong here
fn dtype_to_dense_elements_type(dt: pr.DType) mlir.DenseElementsAttributeTypes {
    return switch (dt) {
        .f16 => .f16,
        .bf16 => .bf16,
        .f32 => .f32,
        .f64 => .f64,
        .i8 => .i8,
        .u8 => .i8,
        .i32 => .i32,
        .i64 => .i64,
        .u32 => .i32,
        .u64 => .i64,
        .bool => .bool,
    };
}

fn scalar_zero_bytes(dtype: pr.DType) []const u8 {
    return switch (dtype) {
        .f16, .bf16 => std.mem.asBytes(&@as(u16, 0)),
        .f32 => std.mem.asBytes(&@as(f32, 0.0)),
        .f64 => std.mem.asBytes(&@as(f64, 0.0)),
        .i8 => std.mem.asBytes(&@as(i8, 0)),
        .u8 => std.mem.asBytes(&@as(u8, 0)),
        .i32 => std.mem.asBytes(&@as(i32, 0)),
        .i64 => std.mem.asBytes(&@as(i64, 0)),
        .u32 => std.mem.asBytes(&@as(u32, 0)),
        .u64 => std.mem.asBytes(&@as(u64, 0)),
        .bool => std.mem.asBytes(&@as(bool, false)),
    };
}

fn scalar_min_bytes(dtype: pr.DType) []const u8 {
    return switch (dtype) {
        .f16 => std.mem.asBytes(&@as(u16, 0xFC00)), // -inf in f16
        .bf16 => std.mem.asBytes(&pr.DType.bf16.encode(f32, -std.math.inf(f32))),
        .f32 => std.mem.asBytes(&@as(f32, -std.math.inf(f32))),
        .f64 => std.mem.asBytes(&@as(f64, -std.math.inf(f64))),
        .i8 => std.mem.asBytes(&@as(i8, std.math.minInt(i8))),
        .u8 => std.mem.asBytes(&@as(u8, 0)),
        .i32 => std.mem.asBytes(&@as(i32, std.math.minInt(i32))),
        .i64 => std.mem.asBytes(&@as(i64, std.math.minInt(i64))),
        .u32 => std.mem.asBytes(&@as(u32, 0)),
        .u64 => std.mem.asBytes(&@as(u64, 0)),
        .bool => std.mem.asBytes(&@as(bool, false)),
    };
}

// ============================================================================
// Pass Integration
// ============================================================================

pub const LowerPassConfig = lower_types.LowerPassConfig;

// TODO: we dont need all these variations anymore do we?

pub fn lower_pass(ptr: *anyopaque, artifact: *pass.Artifact, ctx: *pass.PassContext) pass.PassError!void {
    if (artifact.kind() != .pr) return error.ArtifactKindMismatch;

    const cfg: *LowerPassConfig = @ptrCast(@alignCast(ptr));
    const program = artifact.pr;

    const format: OutputFormat = switch (cfg.encoding) {
        .text => .mlir_text,
        .binary => .mlir_bytecode,
    };

    // Domain boundary: remap internal lowering errors to pipeline-level LoweringFailed.
    // The log line carries the specific error name for diagnostics.
    const bytes = lower_program_to_mlir(
        ctx.allocator,
        program,
        cfg.entry_name,
        format,
    ) catch |e| {
        log.err("lowering failed: {s}", .{@errorName(e)});
        return error.LoweringFailed;
    };

    artifact.replace(ctx.allocator, .{
        .stablehlo = .{
            .bytes = bytes,
            .encoding = cfg.encoding,
        },
    });
}

/// Metadata for the lower pass.
pub fn lower_pass_with_config(config: *LowerPassConfig) pass.Pass {
    return .{
        .ptr = @ptrCast(config),
        .run_fn = lower_pass,
        .name = "stablehlo_lower",
        .input_kind = .pr,
        .output_kind = .stablehlo,
    };
}

/// Convenience: lower with encoding preference, returning a fully-formed
///  StableHLO `Artifact`. Caller owns the artifact and must `deinit` it.
pub fn lower(
    allocator: std.mem.Allocator,
    program: *const pr.Program,
    entry_name: ?[]const u8,
    encoding: pass.Encoding,
) !pass.Artifact {
    const format: OutputFormat = switch (encoding) {
        .text => .mlir_text,
        .binary => .mlir_bytecode,
    };

    const bytes = try lower_program_to_mlir(allocator, program, entry_name, format);

    return .{ .stablehlo = .{ .bytes = bytes, .encoding = encoding } };
}

// ============================================================================
// Tests
// ============================================================================

test "lowering produces verified bytecode" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();
    {
        var b = try pr.FunctionBuilder.init(&program, "main");
        defer b.deinit();
        const a = try b.param_tensor(.f32, &.{ 2, 3 });
        const b_id = try b.param_tensor(.f32, &.{ 3, 2 });
        const c = try b.param_tensor(.f32, &.{ 2, 2 });
        const dot_id = try b.dot(a, b_id);
        const add_id = try b.add(dot_id, c);
        const out_id = try b.multiply(add_id, c);
        const func = try b.finish(&.{out_id});
        try program.add_function(func);
    }

    const bc = try lower_program_to_mlir(std.testing.allocator, &program, null, .mlir_bytecode);
    defer std.testing.allocator.free(bc);
    try std.testing.expect(bc.len > 0);
}

test "lowering keeps non-entry functions when entry_name is set" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var fwd_builder = try pr.FunctionBuilder.init(&program, "forward");
    defer fwd_builder.deinit();
    const fwd_x = try fwd_builder.param_tensor(.f32, &.{2});
    const fwd = try fwd_builder.finish(&.{fwd_x});
    try program.add_function(fwd);

    var bwd_builder = try pr.FunctionBuilder.init(&program, "backward");
    defer bwd_builder.deinit();
    const bwd_x = try bwd_builder.param_tensor(.f32, &.{2});
    const bwd = try bwd_builder.finish(&.{bwd_x});
    try program.add_function(bwd);

    const text = try lower_program_to_mlir(testing.allocator, &program, "backward", .mlir_text);
    defer testing.allocator.free(text);

    try testing.expect(std.mem.indexOf(u8, text, "func.func @main") != null);
    try testing.expect(std.mem.indexOf(u8, text, "func.func @forward") != null);
}

test "lowering renames non-entry main when entry_name differs" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var main_builder = try pr.FunctionBuilder.init(&program, "main");
    defer main_builder.deinit();
    const main_x = try main_builder.param_tensor(.f32, &.{2});
    const main_fn = try main_builder.finish(&.{main_x});
    try program.add_function(main_fn);

    var other_builder = try pr.FunctionBuilder.init(&program, "backward");
    defer other_builder.deinit();
    const other_x = try other_builder.param_tensor(.f32, &.{2});
    const other_fn = try other_builder.finish(&.{other_x});
    try program.add_function(other_fn);

    const text = try lower_program_to_mlir(testing.allocator, &program, "backward", .mlir_text);
    defer testing.allocator.free(text);

    try testing.expect(std.mem.indexOf(u8, text, "func.func @main") != null);
    try testing.expect(std.mem.indexOf(u8, text, "func.func @main_non_entry_0") != null);
}

test "lowering supports reshape/broadcast/transpose" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const t = try b.transpose(x, &.{ 1, 0 });
    const r = try b.reshape(t, &.{6});
    const y = try b.broadcast_in_dim(r, &.{ 2, 6 }, &.{1});

    const func = try b.finish(&.{y});
    try program.add_function(func);

    const bc = try lower_program_to_mlir(std.testing.allocator, &program, null, .mlir_bytecode);
    defer std.testing.allocator.free(bc);
    try std.testing.expect(bc.len > 0);
}

test "lowering supports custom_call boundary" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.custom_call("zigrad.test.missing_handler", &.{x}, x);

    const func = try b.finish(&.{y});
    try program.add_function(func);

    const bc = try lower_program_to_mlir(std.testing.allocator, &program, null, .mlir_bytecode);
    defer std.testing.allocator.free(bc);
    try std.testing.expect(bc.len > 0);
}

test "lowering supports multi-output custom_call boundary" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});
    const y = try b.param_tensor(.f32, &.{2});

    try b.push_region("mock_multi", .{ .kernelize = "mock" });
    const ex = try b.emit(.{ .exp = {} }, &.{x});
    const lg = try b.emit(.{ .log = {} }, &.{y});
    try b.pop_region();

    const func = try b.finish(&.{ ex, lg });
    try program.add_function(func);

    const kernel_mod = @import("../../../kernel.zig");
    const kernelize = @import("../../../pipeline/kernelize.zig");

    var store = kernel_mod.KernelStore.init(testing.allocator);
    defer store.deinit();
    try store.put_profitable("exp,f32[2]>f32[2];log,f32[2]>f32[2]", .{
        .provider_name = "mock",
        .data = "mock",
        .target_name = "mock_multi",
    });

    var kp = kernelize.KernelizePass{
        .store = &store,
    };

    var artifact = pass.Artifact{ .pr = &program };
    var pass_ctx = pass.PassContext{ .allocator = testing.allocator, .io = std.testing.io };
    try kp.pass().run(&artifact, &pass_ctx);

    const text = try lower_program_to_mlir(testing.allocator, &program, null, .mlir_text);
    defer testing.allocator.free(text);

    try testing.expect(std.mem.indexOf(u8, text, zigrad_kernel_call_op_name) != null);
    try testing.expect(std.mem.indexOf(u8, text, "tensor<2xf32>, tensor<2xf32>") != null);
}

test "lowering supports vjp matmul demo" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();
    {
        var b = try pr.FunctionBuilder.init(&program, "main");
        defer b.deinit();
        const a = try b.param_tensor(.f32, &.{ 2, 3 });
        const b_id = try b.param_tensor(.f32, &.{ 3, 2 });
        const c = try b.param_tensor(.f32, &.{ 2, 2 });
        const dot_id = try b.dot(a, b_id);
        const add_id = try b.add(dot_id, c);
        const out_id = try b.multiply(add_id, c);
        const func = try b.finish(&.{out_id});
        try program.add_function(func);
    }

    const fwd = program.functions[0];
    const vjp_func = try @import("../../../pr/ad.zig").vjp(std.testing.allocator, &program, fwd, "vjp", .{});

    try program.add_function(vjp_func);
    const bc = try lower_program_to_mlir(std.testing.allocator, &program, "vjp", .mlir_bytecode);
    defer std.testing.allocator.free(bc);
    try std.testing.expect(bc.len > 0);
}

test "lowering can outline via region annotation" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a = try b.param_tensor(.f32, &.{ 2, 3 });
    const c = try b.param_tensor(.f32, &.{ 3, 2 });
    try b.push_region("outlined-dot", .{ .outline = true });
    const d = try b.dot(a, c);
    try b.pop_region();
    const func = try b.finish(&.{d});
    try program.add_function(func);

    const text = try lower_program_to_mlir(std.testing.allocator, &program, null, .mlir_text);
    defer std.testing.allocator.free(text);

    try std.testing.expect(std.mem.indexOf(u8, text, "call @main_outlined_0") != null);
}

test "lowering tags kernelize provider on outlined functions" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a = try b.param_tensor(.f32, &.{ 2, 3 });
    const c = try b.param_tensor(.f32, &.{ 3, 2 });
    try b.push_region("tvm-kernel", .{ .kernelize = "tvm" });
    const d = try b.dot(a, c);
    try b.pop_region();
    const func = try b.finish(&.{d});
    try program.add_function(func);

    const text = try lower_program_to_mlir(std.testing.allocator, &program, null, .mlir_text);
    defer std.testing.allocator.free(text);

    try std.testing.expect(std.mem.indexOf(u8, text, "zigrad.kernelize.provider") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "tvm") != null);
}

test "lower pass outlines kernelize-annotated region" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const lhs = try b.param_tensor(.f32, &.{ 2, 3 });
    const rhs = try b.param_tensor(.f32, &.{ 3, 2 });
    const bias = try b.param_tensor(.f32, &.{ 2, 2 });

    try b.push_region("matmul_region", .{ .kernelize = "mirage" });
    const dot = try b.dot(lhs, rhs);
    const sum = try b.add(dot, bias);
    const out = try b.multiply(sum, bias);
    try b.pop_region();

    const func = try b.finish(&.{out});
    try program.add_function(func);

    var cfg = LowerPassConfig{ .encoding = .text };
    var artifact = pass.Artifact{ .pr = &program };
    var pass_ctx = pass.PassContext{ .allocator = testing.allocator, .io = std.testing.io };

    try lower_pass(@ptrCast(&cfg), &artifact, &pass_ctx);
    defer artifact.deinit(testing.allocator);

    // Kernelize-annotated regions are unconditionally outlined.
    try testing.expect(std.mem.indexOf(u8, artifact.stablehlo.bytes, "main_outlined_0") != null);
    try testing.expect(std.mem.indexOf(u8, artifact.stablehlo.bytes, "call @main_outlined_0") != null);
}

test "lower pass outlines dot-add kernelize region" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const lhs = try b.param_tensor(.f32, &.{ 2, 3 });
    const rhs = try b.param_tensor(.f32, &.{ 3, 2 });
    const bias = try b.param_tensor(.f32, &.{ 2, 2 });

    try b.push_region("dot_add_region", .{ .kernelize = "mirage" });
    const dot = try b.dot(lhs, rhs);
    const out = try b.add(dot, bias);
    try b.pop_region();

    const func = try b.finish(&.{out});
    try program.add_function(func);

    var cfg = LowerPassConfig{ .encoding = .text };
    var artifact = pass.Artifact{ .pr = &program };
    var pass_ctx = pass.PassContext{ .allocator = testing.allocator, .io = std.testing.io };

    try lower_pass(@ptrCast(&cfg), &artifact, &pass_ctx);
    defer artifact.deinit(testing.allocator);

    // Kernelize-annotated region is outlined; ops move to outlined function.
    try testing.expect(std.mem.indexOf(u8, artifact.stablehlo.bytes, "main_outlined_0") != null);
    try testing.expect(std.mem.indexOf(u8, artifact.stablehlo.bytes, "call @main_outlined_0") != null);
}

test "lower pass outlines dot-log kernelize region" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const lhs = try b.param_tensor(.f32, &.{ 2, 3 });
    const rhs = try b.param_tensor(.f32, &.{ 3, 2 });

    try b.push_region("dot_log_region", .{ .kernelize = "mirage" });
    const dot = try b.dot(lhs, rhs);
    const out = try b.log(dot);
    try b.pop_region();

    const func = try b.finish(&.{out});
    try program.add_function(func);

    var cfg = LowerPassConfig{ .encoding = .text };
    var artifact = pass.Artifact{ .pr = &program };
    var pass_ctx = pass.PassContext{ .allocator = testing.allocator, .io = std.testing.io };

    try lower_pass(@ptrCast(&cfg), &artifact, &pass_ctx);
    defer artifact.deinit(testing.allocator);

    // Kernelize-annotated region is outlined; provider attribute set on callee.
    try testing.expect(std.mem.indexOf(u8, artifact.stablehlo.bytes, "zigrad.kernelize.provider") != null);
    try testing.expect(std.mem.indexOf(u8, artifact.stablehlo.bytes, "main_outlined_0") != null);
    try testing.expect(std.mem.indexOf(u8, artifact.stablehlo.bytes, "stablehlo.dot_general") != null);
    try testing.expect(std.mem.indexOf(u8, artifact.stablehlo.bytes, "stablehlo.log") != null);
}

test "lower pass outlines near-miss kernelize region" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const lhs = try b.param_tensor(.f32, &.{ 2, 3 });
    const rhs = try b.param_tensor(.f32, &.{ 3, 2 });
    const bias = try b.param_tensor(.f32, &.{ 2, 2 });

    try b.push_region("near_miss_region", .{ .kernelize = "mirage" });
    const dot = try b.dot(lhs, rhs);
    const shifted = try b.subtract(dot, bias);
    const out = try b.multiply(shifted, bias);
    try b.pop_region();

    const func = try b.finish(&.{out});
    try program.add_function(func);

    var cfg = LowerPassConfig{ .encoding = .text };
    var artifact = pass.Artifact{ .pr = &program };
    var pass_ctx = pass.PassContext{ .allocator = testing.allocator, .io = std.testing.io };

    try lower_pass(@ptrCast(&cfg), &artifact, &pass_ctx);
    defer artifact.deinit(testing.allocator);

    // Kernelize-annotated region is outlined even for non-standard patterns.
    try testing.expect(std.mem.indexOf(u8, artifact.stablehlo.bytes, "main_outlined_0") != null);
    try testing.expect(std.mem.indexOf(u8, artifact.stablehlo.bytes, "call @main_outlined_0") != null);
}

test "lower pass produces MLIR artifact" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.param_tensor(.f32, &.{ 3, 2 });
    const z = try b.dot(x, y);
    const func = try b.finish(&.{z});

    var ctx = pass.PassContext{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };
    var cfg = LowerPassConfig{ .encoding = .binary };

    try program.add_function(func);
    var output = pass.Artifact{ .pr = &program };
    try lower_pass(@ptrCast(&cfg), &output, &ctx);
    defer output.deinit(std.testing.allocator);

    try std.testing.expectEqual(pass.ArtifactKind.stablehlo, output.kind());
    try std.testing.expect(output.stablehlo.bytes.len > 0);
}

test "lower pass emits zigrad.kernel_call for custom_call ops (pre-legalize)" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});
    const y = try b.custom_call("zigrad.test.missing_handler", &.{x}, x);
    const func = try b.finish(&.{y});
    try program.add_function(func);

    // Lower pass is baseline - no legalize.
    var cfg = LowerPassConfig{ .encoding = .text };
    var artifact = pass.Artifact{ .pr = &program };
    var pass_ctx = pass.PassContext{ .allocator = testing.allocator, .io = std.testing.io };
    try lower_pass(@ptrCast(&cfg), &artifact, &pass_ctx);
    defer artifact.deinit(testing.allocator);

    try testing.expectEqual(pass.ArtifactKind.stablehlo, artifact.kind());

    // Custom calls are emitted as zigrad.kernel_call (legalize converts to stablehlo.custom_call).
    try testing.expect(std.mem.indexOf(u8, artifact.stablehlo.bytes, zigrad_kernel_call_op_name) != null);
    try testing.expect(std.mem.indexOf(u8, artifact.stablehlo.bytes, "stablehlo.custom_call") == null);
}
