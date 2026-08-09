//! Shared PR to MLIR function scaffolding.
//!
//! Provides function construction, value mapping, type conversion, outlining,
//!  and serialization for a caller-supplied `LowerOpFn`.
//!
//! The current executable recipe controls entry naming and PR region outlining.
const std = @import("std");

const pr = @import("../pr/pr.zig");
const kernel = @import("../pr/kernel.zig");
const outline = @import("../pr/outline.zig");
const mlir = @import("../c/mlir/mlir.zig");
const MlirSession = @import("session.zig").Session;

/// Serialized MLIR output encoding.
pub const OutputFormat = enum {
    mlir_text,
    mlir_bytecode,
};

const log = std.log.scoped(.@"zg/mlir_lower");

/// Maximum tensor rank representable by PR.
pub const max_rank = pr.max_rank;

/// Failures exposed by PR to MLIR lowering.
///
/// `InvalidProgram` reports a missing value mapping or another structural
///  invariant that lowering requires. Callers validate PR before lowering.
pub const LowerError = mlir.Error || std.Io.Writer.Error || error{InvalidProgram};

/// Callback type for dialect-specific op translation.
///
/// Each dialect module provides an implementation that maps a single PR op
///  to one or more MLIR ops in the target dialect, appending them to
///  `LowerContext.block`.
pub const LowerOpFn = *const fn (LowerContext, *const pr.Op) LowerError!void;

/// Per-function lowering state threaded through op lowering helpers.
///
/// `value_map` carries lowered SSA values from producers to consumers and uses
///  null for values that have not been lowered.
///
/// The caller provides arena-backed storage for this function lowering.
pub const LowerContext = struct {
    /// MLIR context for type/attribute construction.
    mlir_ctx: mlir.Context,
    /// The MLIR block being populated (function body).
    block: mlir.Block,
    /// Location attached to every generated op (currently file-level).
    loc: mlir.Location,
    /// `Var.id`-indexed map sized to `func.var_count`, with null for values not
    ///  yet lowered.
    value_map: []?mlir.Value,
    /// Scratch allocator for transient lowering buffers (e.g., dim arrays).
    arena: std.mem.Allocator,

    pub fn get_value(self: LowerContext, v: *const pr.Var) ?mlir.Value {
        return self.value_map[v.id];
    }

    pub fn set_value(self: LowerContext, v: *const pr.Var, value: mlir.Value) void {
        self.value_map[v.id] = value;
    }

    pub fn tensor_to_mlir_type(self: LowerContext, t: pr.Tensor) mlir.Type {
        return tensor_type(self.mlir_ctx, t);
    }
};

/// Lower a PR program to MLIR bytecode or text.
///
/// The caller provides a configured session and the callback that translates
///  individual PR operations.
///
/// ## Entry function selection and naming
///
/// `entry_name` selects which PR function is the compilation entry point.
///
/// The selected function is emitted as `@main`. Other functions retain their
///  PR names unless that would collide with the entry symbol.
///
/// This operation does not run MLIR passes. Callers compose those operations
///  explicitly.
///
/// TODO(mlir): Make entry naming and region outlining explicit options.
pub fn lower_program_to_mlir(
    allocator: std.mem.Allocator,
    session: MlirSession,
    program: *const pr.Program,
    entry_name: ?[]const u8,
    out: OutputFormat,
    lower_op_fn: LowerOpFn,
) LowerError![]u8 {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const ctx = session.ctx;
    const loc = mlir.Location.unknown(ctx);

    var module = mlir.Module.init(loc);
    defer module.deinit();

    const entry_index = try find_entry_function(program, entry_name);

    for (program.functions, 0..) |func, idx| {
        const sym_name = try choose_symbol_name(arena, program, idx, entry_index, entry_name);
        try lower_function_into_module(arena, ctx, module, func, sym_name, lower_op_fn);
    }

    if (!module.op().verify()) return error.InvalidMlir;

    return try serialize_module(allocator, module, out);
}

/// Lower a single PR function into the MLIR module.
///
/// Annotated operations may be outlined into separate `func.func` operations.
fn lower_function_into_module(
    arena: std.mem.Allocator,
    ctx: mlir.Context,
    module: mlir.Module,
    func: pr.Function,
    sym_name: []const u8,
    lower_op_fn: LowerOpFn,
) LowerError!void {
    const loc = mlir.Location.unknown(ctx);

    const param_types = try arena.alloc(mlir.Type, func.params.len);
    const param_locs = try arena.alloc(mlir.Location, func.params.len);
    for (func.params, 0..) |param_var, i| {
        const tensor = param_var.aval.as_tensor();
        param_types[i] = tensor_type(ctx, tensor);
        param_locs[i] = loc;
    }

    const result_types = try arena.alloc(mlir.Type, func.returns.len);
    for (func.returns, 0..) |ret_var, i| {
        const tensor = ret_var.aval.as_tensor();
        result_types[i] = tensor_type(ctx, tensor);
    }

    const fn_type = mlir.Type.function(ctx, param_types, result_types);

    const entry_block = try mlir.Block.init(param_types, param_locs);

    const value_map = try arena.alloc(?mlir.Value, func.var_count);
    @memset(value_map, null);
    for (func.params, 0..) |param_var, i| {
        value_map[param_var.id] = entry_block.argument(i);
    }

    const lower_ctx = LowerContext{
        .mlir_ctx = ctx,
        .block = entry_block,
        .loc = loc,
        .value_map = value_map,
        .arena = arena,
    };

    const region_map = try build_region_map(arena, func);

    var outlined_index: usize = 0;
    const outlined_prefix = if (sym_name.len == 0) "func" else sym_name;
    for (func.ops, 0..) |op, op_idx| {
        if (region_map[op_idx]) |region| {
            const requests_outline = outline.is_requested(region) catch return error.InvalidProgram;
            const requests_kernel = (kernel.requested_provider(region) catch return error.InvalidProgram) != null;
            if (requests_outline or requests_kernel) {
                try lower_outlined_op(arena, &outlined_index, outlined_prefix, ctx, module, lower_ctx, op, region, lower_op_fn);
                continue;
            }
        }
        try lower_op_fn(lower_ctx, op);
    }

    const ret_values = try arena.alloc(mlir.Value, func.returns.len);
    for (func.returns, 0..) |ret_var, i| {
        ret_values[i] = value_map[ret_var.id] orelse return error.InvalidProgram;
    }

    const return_op = mlir.Operation.make(ctx, "func.return", .{
        .operands = ret_values,
        .verify = false,
        .location = loc,
    });
    entry_block.append_operation(return_op);

    const func_op = mlir.Operation.make(ctx, "func.func", .{
        .results = &.{},
        .blocks = &.{entry_block},
        .attributes = &.{
            .{ "sym_name", mlir.Attribute.string(ctx, sym_name) },
            .{ "function_type", mlir.Attribute.type_(fn_type) },
        },
        .verify = false,
        .location = loc,
    });
    module.get_body().append_operation(func_op);
}

/// Builds a region lookup indexed by op position.
///
/// Entries are null when the op is outside every outlined or kernelized region.
///
/// TODO(mlir): Decide how program-level lowering represents nested functions and regions.
fn build_region_map(arena: std.mem.Allocator, func: pr.Function) std.mem.Allocator.Error![]?pr.Region {
    const map = try arena.alloc(?pr.Region, func.ops.len);
    @memset(map, null);
    for (func.regions) |region| {
        for (region.op_ids) |op_id| {
            const index = func.op_index_by_id(op_id) orelse continue;
            if (index < map.len) map[index] = region;
        }
    }
    return map;
}

fn lower_outlined_op(
    arena: std.mem.Allocator,
    outlined_index: *usize,
    outlined_prefix: []const u8,
    mlir_ctx: mlir.Context,
    module: mlir.Module,
    ctx: LowerContext,
    op: *const pr.Op,
    region: pr.Region,
    lower_op_fn: LowerOpFn,
) LowerError!void {
    // TODO(mlir): Represent multi-operation PR regions as one outlined function.

    // This outlining form accepts one input-bearing, single-result operation.
    if (op.outputs.len != 1) return error.InvalidProgram;
    if (op.inputs.len == 0) return error.InvalidProgram;

    const out_var = op.result(0);
    const out_tensor = out_var.aval.as_tensor();
    const out_type = ctx.tensor_to_mlir_type(out_tensor);

    const callee_name_z = try std.fmt.allocPrintSentinel(arena, "{s}_outlined_{d}", .{ outlined_prefix, outlined_index.* }, 0);
    outlined_index.* += 1;

    const callee_param_types = try arena.alloc(mlir.Type, op.inputs.len);
    const callee_param_locs = try arena.alloc(mlir.Location, op.inputs.len);
    for (op.inputs, 0..) |operand, i| {
        const in_tensor = operand.value.aval.as_tensor();
        callee_param_types[i] = ctx.tensor_to_mlir_type(in_tensor);
        callee_param_locs[i] = ctx.loc;
    }
    const callee_result_types = &[_]mlir.Type{out_type};
    const callee_fn_type = mlir.Type.function(mlir_ctx, callee_param_types, callee_result_types);

    const callee_entry = try mlir.Block.init(callee_param_types, callee_param_locs);

    const callee_value_map = try arena.alloc(?mlir.Value, ctx.value_map.len);
    @memset(callee_value_map, null);
    for (op.inputs, 0..) |operand, i| callee_value_map[operand.value.id] = callee_entry.argument(i);

    const callee_ctx = LowerContext{
        .mlir_ctx = mlir_ctx,
        .block = callee_entry,
        .loc = ctx.loc,
        .value_map = callee_value_map,
        .arena = arena,
    };

    try lower_op_fn(callee_ctx, op);

    const callee_out = callee_value_map[out_var.id] orelse return error.InvalidProgram;
    const callee_ret = mlir.Operation.make(mlir_ctx, "func.return", .{
        .operands = &.{callee_out},
        .verify = false,
        .location = ctx.loc,
    });
    callee_entry.append_operation(callee_ret);

    const callee_op = mlir.Operation.make(mlir_ctx, "func.func", .{
        .results = &.{},
        .blocks = &.{callee_entry},
        .attributes = &.{
            .{ "sym_name", mlir.Attribute.string(mlir_ctx, callee_name_z) },
            .{ "function_type", mlir.Attribute.type_(callee_fn_type) },
            .{ "llvm.noinline", mlir.Attribute.unit(mlir_ctx) },
        },
        .verify = false,
        .location = ctx.loc,
    });

    if (kernel.requested_provider(region) catch return error.InvalidProgram) |provider| {
        callee_op.set_attribute_by_name("zigrad.kernelize.provider", mlir.Attribute.string(mlir_ctx, provider));
    }
    module.get_body().append_operation(callee_op);

    const call_operands = try arena.alloc(mlir.Value, op.inputs.len);
    for (op.inputs, 0..) |operand, i| call_operands[i] = ctx.get_value(operand.value) orelse return error.InvalidProgram;

    const call_op = mlir.Operation.make(mlir_ctx, "func.call", .{
        .results = &.{out_type},
        .operands = call_operands,
        .attributes = &.{
            .{ "callee", mlir.Attribute.symbol(mlir_ctx, callee_name_z) },
        },
        .verify = false,
        .location = ctx.loc,
    });
    ctx.block.append_operation(call_op);
    ctx.set_value(out_var, call_op.result(0));
}

/// Lower a `func.call` op. Uses the `func` dialect only.
pub fn lower_call(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const callee = op.params.call.callee;

    const operand_values = try ctx.arena.alloc(mlir.Value, op.inputs.len);
    for (op.inputs, 0..) |operand, i| {
        operand_values[i] = ctx.get_value(operand.value) orelse return error.InvalidProgram;
    }

    const result_types = try ctx.arena.alloc(mlir.Type, op.outputs.len);
    for (op.outputs, 0..) |out_var, i| {
        const out_tensor = out_var.aval.as_tensor();
        result_types[i] = ctx.tensor_to_mlir_type(out_tensor);
    }

    const callee_z = try ctx.arena.dupeZ(u8, callee);

    const mlir_op = mlir.Operation.make(ctx.mlir_ctx, "func.call", .{
        .results = result_types,
        .operands = operand_values,
        .attributes = &.{
            .{ "callee", mlir.Attribute.symbol(ctx.mlir_ctx, callee_z) },
        },
        .verify = false,
        .location = ctx.loc,
    });

    ctx.block.append_operation(mlir_op);
    for (op.outputs, 0..) |out_var, i| {
        ctx.set_value(out_var, mlir_op.result(i));
    }
}

/// Convert a PR element type to its builtin MLIR scalar type.
pub fn dtype_to_mlir_type(ctx: mlir.Context, dt: pr.DType) mlir.Type {
    return switch (dt) {
        .f16 => mlir.Type.float(ctx, .f16),
        .bf16 => mlir.Type.float(ctx, .bf16),
        .f32 => mlir.Type.float(ctx, .f32),
        .f64 => mlir.Type.float(ctx, .f64),
        .i8 => mlir.Type.int(ctx, .i8),
        .u8 => mlir.Type.int(ctx, .i8),
        .i32 => mlir.Type.int(ctx, .i32),
        .i64 => mlir.Type.int(ctx, .i64),
        .u32 => mlir.Type.int(ctx, .i32),
        .u64 => mlir.Type.int(ctx, .i64),
        .bool => mlir.Type.int(ctx, .i1),
    };
}

/// Convert a PR tensor type to a ranked MLIR tensor type.
pub fn tensor_type(ctx: mlir.Context, t: pr.Tensor) mlir.Type {
    var buf: [max_rank]i64 = undefined;
    const dims_i64 = buf[0..t.shape.dims.len];
    for (t.shape.dims, 0..) |d, i| dims_i64[i] = @intCast(d);
    return mlir.Type.tensor(dims_i64, dtype_to_mlir_type(ctx, t.dtype));
}

const SerializeError = std.mem.Allocator.Error || std.Io.Writer.Error;

fn serialize_module(allocator: std.mem.Allocator, module: mlir.Module, out: OutputFormat) SerializeError![]u8 {
    var writer_state = std.Io.Writer.Allocating.init(allocator);
    defer writer_state.deinit();

    switch (out) {
        .mlir_bytecode => try module.op().write_bytecode(&writer_state.writer),
        .mlir_text => try module.op().print(&writer_state.writer, .{}),
    }

    return try writer_state.toOwnedSlice();
}

/// Resolve the PR function selected as the executable entry point.
pub fn find_entry_function(program: *const pr.Program, entry_name: ?[]const u8) LowerError!usize {
    if (program.functions.len == 0) return error.InvalidProgram;

    if (entry_name) |name| {
        for (program.functions, 0..) |func, idx| {
            if (std.mem.eql(u8, func.name, name)) return idx;
        }
        return error.InvalidProgram;
    }

    for (program.functions, 0..) |func, idx| {
        if (std.mem.eql(u8, func.name, "main")) return idx;
    }

    if (program.functions.len == 1) return 0;
    return error.InvalidProgram;
}

/// Choose an MLIR symbol without colliding with the emitted `@main` entry.
pub fn choose_symbol_name(
    arena: std.mem.Allocator,
    program: *const pr.Program,
    idx: usize,
    entry_index: usize,
    entry_name: ?[]const u8,
) error{OutOfMemory}![]const u8 {
    if (idx == entry_index) return "main";

    const func = program.functions[idx];
    if (entry_name == null or !std.mem.eql(u8, func.name, "main")) return func.name;

    var suffix: usize = 0;
    while (true) : (suffix += 1) {
        const candidate = try std.fmt.allocPrint(arena, "main_non_entry_{d}", .{suffix});
        if (!is_symbol_name_used(program, entry_index, candidate)) {
            if (!@import("builtin").is_test) log.warn("renaming non-entry function 'main' to '{s}' to avoid entry collision", .{candidate});
            return candidate;
        }
    }
}

fn is_symbol_name_used(program: *const pr.Program, entry_index: usize, name: []const u8) bool {
    for (program.functions, 0..) |func, idx| {
        if (idx == entry_index) continue;
        if (std.mem.eql(u8, func.name, name)) return true;
    }
    return false;
}
