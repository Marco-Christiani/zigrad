/// StableHLO Lowering
///
/// Lowers PR (Program Representation) to StableHLO MLIR.
/// This is the core PR -> MLIR boundary in the pass-based pipeline.
///
/// All MLIR/StableHLO concerns are contained here. PR is read-only input.
///
/// Provides:
/// 1. `lower_program_to_mlir`: Direct lowering API (PR program → MLIR bytes)
/// 2. `lower_function_to_mlir`: Single-function convenience wrapper
/// 3. `lower_pass` / `lower_pass_with_config`: Pass-based pipeline integration
const std = @import("std");

const pr = @import("../pr/pr.zig");
const mlir = @import("../c/mlir/mlir.zig");
const stablehlo = @import("../c/mlir/dialects/stablehlo.zig");
const pass = @import("../pipeline/pass.zig");
const log = std.log.scoped(.@"zg/lower_stablehlo");

pub const LowerError = error{ InvalidProgram, InvalidMlir, OutOfMemory };

const zigrad_kernel_call_op_name = "zigrad.kernel_call";

pub const OutputFormat = enum {
    mlir_text,
    mlir_bytecode,
};

pub const KernelizationLane = enum {
    pr,
    mlir,
};

// ============================================================================
// Lowering Context
// ============================================================================

/// State threaded through per-equation lowering. Owns the VarId → MLIR Value mapping.
const LowerContext = struct {
    mlir_ctx: mlir.Context,
    block: mlir.Block,
    loc: mlir.Location,
    value_map: []?mlir.Value,
    func: pr.Function,
    arena: std.mem.Allocator,

    fn inputs(self: LowerContext, eqn: pr.Eqn) []const pr.VarId {
        return eqn.inputs.slice(pr.VarId, self.func.varids_store);
    }

    fn outputs(self: LowerContext, eqn: pr.Eqn) []const pr.VarId {
        return eqn.outputs.slice(pr.VarId, self.func.varids_store);
    }

    fn params(self: LowerContext, eqn: pr.Eqn) []const pr.Param {
        return eqn.params.slice(pr.Param, self.func.params_store);
    }

    fn get_value(self: LowerContext, id: pr.VarId) ?mlir.Value {
        return self.value_map[@intCast(id)];
    }

    fn set_value(self: LowerContext, id: pr.VarId, value: mlir.Value) void {
        self.value_map[@intCast(id)] = value;
    }

    fn tensor_of(self: LowerContext, id: pr.VarId) LowerError!pr.Tensor {
        return self.func.avals[@intCast(id)].as_tensor() orelse error.InvalidProgram;
    }

    fn tensor_to_mlir_type(self: LowerContext, t: pr.Tensor) LowerError!mlir.Type {
        const dims_i64 = self.arena.alloc(i64, t.shape.dims.len) catch return error.OutOfMemory;
        for (t.shape.dims, 0..) |d, i| dims_i64[i] = @intCast(d);
        return mlir.Type.tensor(dims_i64, dtype_to_mlir_type(self.mlir_ctx, t.dtype));
    }
};

// ============================================================================
// Core Lowering Implementation
// ============================================================================

/// Lower a PR program to StableHLO MLIR bytecode or text.
///
/// Entry Function Selection and Naming:
/// - `entry_name`: Selects which PR function is the compilation entry point.
///   - If provided: That function is renamed to "@main" in MLIR.
///   - If null: "main" (or the only function) is used as entry.
/// - The entry function is always renamed to "@main" in the MLIR output (XLA/PJRT requirement).
/// - Non-entry functions retain their PR names, except when a non-entry function
///   is already named "main" and a different entry is selected; that symbol is
///   renamed to avoid collisions.
/// Lower a PR program to StableHLO MLIR.
///
/// This is baseline lowering only: PR ops become MLIR ops. No MLIR-stage
/// passes (select, legalize) are executed — those are separate pipeline
/// passes composed explicitly by the caller.
pub fn lower_program_to_mlir(
    allocator: std.mem.Allocator,
    program: *const pr.Program,
    entry_name: ?[]const u8,
    out: OutputFormat,
) LowerError![]u8 {
    return lower_program_impl(allocator, program, entry_name, out, .pr);
}

fn lower_program_impl(
    allocator: std.mem.Allocator,
    program: *const pr.Program,
    entry_name: ?[]const u8,
    out: OutputFormat,
    kernelization_lane: KernelizationLane,
) LowerError![]u8 {
    pr.validate_program(program) catch return error.InvalidProgram;

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var registry = mlir.Registry.init() catch return error.OutOfMemory;
    defer registry.deinit();

    mlir.DialectHandle.from_string("func").insert_dialect(registry);
    mlir.DialectHandle.from_string("stablehlo").insert_dialect(registry);

    var ctx = mlir.Context.init_with_registry(registry, false) catch return error.OutOfMemory;
    defer ctx.deinit();
    ctx.allow_unregistered_dialects(false);

    mlir.register_zigrad_extensions(ctx) catch {
        log.err("missing MLIR extension shim; set ZG_MLIR_SHIM_PATH or provide ZG_EXTERNAL_SDK_ROOT with lib/libzigrad_mlir_ext.so", .{});
        return error.InvalidMlir;
    };

    const func_handle = mlir.DialectHandle.from_string("func");
    func_handle.register_dialect(ctx);
    _ = func_handle.load_dialect(ctx);

    const stablehlo_handle = mlir.DialectHandle.from_string("stablehlo");
    stablehlo_handle.register_dialect(ctx);
    _ = stablehlo_handle.load_dialect(ctx);

    const loc = mlir.Location.unknown(ctx);

    var module = mlir.Module.init(loc);
    defer module.deinit();

    const entry_index = try find_entry_function(program, entry_name);

    for (program.functions, 0..) |func, idx| {
        const sym_name = choose_symbol_name(arena, program, idx, entry_index, entry_name) catch return error.OutOfMemory;
        try lower_function_into_module(arena, ctx, module, func, sym_name, kernelization_lane);
    }

    if (!module.op().verify()) return error.InvalidMlir;

    return serialize_module(allocator, module, out);
}

fn serialize_module(allocator: std.mem.Allocator, module: mlir.Module, out: OutputFormat) LowerError![]u8 {
    var writer_state = std.Io.Writer.Allocating.init(allocator);
    defer writer_state.deinit();

    switch (out) {
        .mlir_bytecode => module.op().write_bytecode(&writer_state.writer) catch return error.OutOfMemory,
        .mlir_text => module.op().print(&writer_state.writer, .{}) catch return error.OutOfMemory,
    }

    return writer_state.toOwnedSlice() catch return error.OutOfMemory;
}

pub fn lower_function_to_mlir(allocator: std.mem.Allocator, func: pr.Function, out: OutputFormat) LowerError![]u8 {
    var program = pr.Program.init(allocator);
    defer program.deinit();

    program.add_function(func) catch return error.OutOfMemory;
    return lower_program_to_mlir(allocator, &program, func.name, out);
}

fn lower_function_into_module(
    arena: std.mem.Allocator,
    ctx: mlir.Context,
    module: mlir.Module,
    func: pr.Function,
    sym_name: []const u8,
    kernelization_lane: KernelizationLane,
) LowerError!void {
    const loc = mlir.Location.unknown(ctx);

    const param_types = arena.alloc(mlir.Type, func.params.len) catch return error.OutOfMemory;
    const param_locs = arena.alloc(mlir.Location, func.params.len) catch return error.OutOfMemory;
    for (func.params, 0..) |param_id, i| {
        const tensor = func.avals[@intCast(param_id)].as_tensor() orelse return error.InvalidProgram;
        param_types[i] = try tensor_to_mlir_type_standalone(ctx, tensor, arena);
        param_locs[i] = loc;
    }

    const result_types = arena.alloc(mlir.Type, func.returns.len) catch return error.OutOfMemory;
    for (func.returns, 0..) |ret_id, i| {
        const tensor = func.avals[@intCast(ret_id)].as_tensor() orelse return error.InvalidProgram;
        result_types[i] = try tensor_to_mlir_type_standalone(ctx, tensor, arena);
    }

    const fn_type = mlir.Type.function(ctx, param_types, result_types);

    const entry_block = mlir.Block.init(param_types, param_locs) catch return error.OutOfMemory;

    const value_map = arena.alloc(?mlir.Value, func.avals.len) catch return error.OutOfMemory;
    @memset(value_map, null);
    for (func.params, 0..) |param_id, i| {
        value_map[@intCast(param_id)] = entry_block.argument(i);
    }

    const lower_ctx = LowerContext{
        .mlir_ctx = ctx,
        .block = entry_block,
        .loc = loc,
        .value_map = value_map,
        .func = func,
        .arena = arena,
    };

    // Build eqn-index → region lookup for outlining decisions.
    const region_map = build_region_map(arena, func) catch return error.OutOfMemory;

    var outlined_index: usize = 0;
    const outlined_prefix = if (sym_name.len == 0) "func" else sym_name;
    for (func.eqns, 0..) |eqn, eqn_idx| {
        if (region_map[eqn_idx]) |region| {
            const outline_for_kernelize = region.annotation.kernelize != null and kernelization_lane == .pr;
            if (region.annotation.outline or outline_for_kernelize) {
                try lower_outlined_eqn(arena, &outlined_index, outlined_prefix, ctx, module, lower_ctx, eqn, region);
                continue;
            }

            try lower_eqn(lower_ctx, eqn);

            if (kernelization_lane == .mlir and region.annotation.kernelize != null) {
                try tag_kernelize_marker_attrs(lower_ctx, eqn, region);
            }
            continue;
        }
        try lower_eqn(lower_ctx, eqn);
    }

    const ret_values = arena.alloc(mlir.Value, func.returns.len) catch return error.OutOfMemory;
    for (func.returns, 0..) |ret_id, i| {
        ret_values[i] = value_map[@intCast(ret_id)] orelse return error.InvalidProgram;
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

fn tag_kernelize_marker_attrs(ctx: LowerContext, eqn: pr.Eqn, region: pr.Region) LowerError!void {
    const provider = region.annotation.kernelize orelse return;
    const outs = ctx.outputs(eqn);
    if (outs.len == 0) return;

    const out_value = ctx.get_value(outs[0]) orelse return error.InvalidProgram;
    if (!out_value.is_a_op_result()) return;

    const owner = out_value.owner();
    owner.set_attribute_by_name("zigrad.kernelize.provider", mlir.Attribute.string(ctx.mlir_ctx, provider));
    owner.set_attribute_by_name("zigrad.kernelize.region", mlir.Attribute.string(ctx.mlir_ctx, region.name));
}

/// Build per-eqn region lookup. Returns a slice indexed by eqn position;
/// null if the eqn is not inside any outline/kernelize region.
fn build_region_map(arena: std.mem.Allocator, func: pr.Function) error{OutOfMemory}![]?pr.Region {
    const map = arena.alloc(?pr.Region, func.eqns.len) catch return error.OutOfMemory;
    @memset(map, null);
    for (func.regions) |region| {
        const start: usize = region.eqn_start;
        const end: usize = start + region.eqn_len;
        for (start..end) |i| {
            if (i < map.len) map[i] = region;
        }
    }
    return map;
}

fn lower_outlined_eqn(
    arena: std.mem.Allocator,
    outlined_index: *usize,
    outlined_prefix: []const u8,
    mlir_ctx: mlir.Context,
    module: mlir.Module,
    ctx: LowerContext,
    eqn: pr.Eqn,
    region: pr.Region,
) LowerError!void {
    const eqn_inputs = ctx.inputs(eqn);
    const eqn_outputs = ctx.outputs(eqn);

    // All current PR ops are single-output; keep outlining strict.
    if (eqn_outputs.len != 1) return error.InvalidProgram;
    if (eqn_inputs.len == 0) return error.InvalidProgram;

    const out_id = eqn_outputs[0];
    const out_tensor = try ctx.tensor_of(out_id);
    const out_type = try ctx.tensor_to_mlir_type(out_tensor);

    const callee_name = std.fmt.allocPrint(arena, "{s}_outlined_{d}", .{ outlined_prefix, outlined_index.* }) catch return error.OutOfMemory;
    outlined_index.* += 1;
    const callee_name_z = arena.allocSentinel(u8, callee_name.len, 0) catch return error.OutOfMemory;
    @memcpy(callee_name_z, callee_name);

    // Build callee signature (inputs -> output).
    const callee_param_types = arena.alloc(mlir.Type, eqn_inputs.len) catch return error.OutOfMemory;
    const callee_param_locs = arena.alloc(mlir.Location, eqn_inputs.len) catch return error.OutOfMemory;
    for (eqn_inputs, 0..) |in_id, i| {
        const in_tensor = try ctx.tensor_of(in_id);
        callee_param_types[i] = try ctx.tensor_to_mlir_type(in_tensor);
        callee_param_locs[i] = ctx.loc;
    }
    const callee_result_types = &[_]mlir.Type{out_type};
    const callee_fn_type = mlir.Type.function(mlir_ctx, callee_param_types, callee_result_types);

    // Build callee body.
    const callee_entry = mlir.Block.init(callee_param_types, callee_param_locs) catch return error.OutOfMemory;

    const callee_value_map = arena.alloc(?mlir.Value, ctx.func.avals.len) catch return error.OutOfMemory;
    @memset(callee_value_map, null);
    for (eqn_inputs, 0..) |in_id, i| callee_value_map[@intCast(in_id)] = callee_entry.argument(i);

    const callee_ctx = LowerContext{
        .mlir_ctx = mlir_ctx,
        .block = callee_entry,
        .loc = ctx.loc,
        .value_map = callee_value_map,
        .func = ctx.func,
        .arena = arena,
    };

    try lower_eqn(callee_ctx, eqn);

    const callee_out = callee_value_map[@intCast(out_id)] orelse return error.InvalidProgram;
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
            .{ "sym_name", mlir.Attribute.string(mlir_ctx, callee_name) },
            .{ "function_type", mlir.Attribute.type_(callee_fn_type) },
            .{ "llvm.noinline", mlir.Attribute.unit(mlir_ctx) },
        },
        .verify = false,
        .location = ctx.loc,
    });

    if (region.annotation.kernelize) |provider| {
        callee_op.set_attribute_by_name("zigrad.kernelize.provider", mlir.Attribute.string(mlir_ctx, provider));
    }
    module.get_body().append_operation(callee_op);

    // Emit a call in the original block.
    const call_operands = arena.alloc(mlir.Value, eqn_inputs.len) catch return error.OutOfMemory;
    for (eqn_inputs, 0..) |in_id, i| call_operands[i] = ctx.get_value(in_id) orelse return error.InvalidProgram;

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
    ctx.set_value(out_id, call_op.result(0));
}

// ============================================================================
// Per-Op Lowering (switch dispatch)
// ============================================================================

fn lower_eqn(ctx: LowerContext, eqn: pr.Eqn) LowerError!void {
    switch (eqn.prim) {
        // Binary elementwise
        .add => lower_binary(stablehlo.add, ctx, eqn),
        .subtract => lower_binary(stablehlo.subtract, ctx, eqn),
        .multiply => lower_binary(stablehlo.multiply, ctx, eqn),
        .divide => lower_binary(stablehlo.divide, ctx, eqn),
        .maximum => lower_binary(stablehlo.maximum, ctx, eqn),
        // Unary
        .exp => lower_unary(stablehlo.exponential, ctx, eqn),
        .log => lower_unary(stablehlo.log, ctx, eqn),
        .rsqrt => lower_unary(stablehlo.rsqrt, ctx, eqn),
        .logistic => lower_unary(stablehlo.logistic, ctx, eqn),
        // Type conversion
        .convert => try lower_convert(ctx, eqn),
        // Constant
        .literal => try lower_literal(ctx, eqn),
        // Shape
        .reshape => try lower_reshape(ctx, eqn),
        .transpose => try lower_transpose(ctx, eqn),
        .broadcast_in_dim => try lower_broadcast_in_dim(ctx, eqn),
        .iota => try lower_iota(ctx, eqn),
        .slice => try lower_slice(ctx, eqn),
        .concatenate => try lower_concatenate(ctx, eqn),
        // Reduction
        .reduce_sum => try lower_reduce_sum(ctx, eqn),
        .reduce_max => try lower_reduce_max(ctx, eqn),
        // Contraction
        .dot => try lower_dot(ctx, eqn),
        .dot_general => try lower_dot_general(ctx, eqn),
        // Compare
        .compare => try lower_compare(ctx, eqn),
        .select => try lower_select(ctx, eqn),
        // Structured
        .gather => try lower_gather(ctx, eqn),
        .scatter => try lower_scatter(ctx, eqn),
        // Special
        .call => try lower_call(ctx, eqn),
        .custom_call => try lower_custom_call(ctx, eqn),
    }
}

// --- Binary elementwise ---

fn lower_binary(
    comptime op_fn: fn (mlir.Context, mlir.Value, mlir.Value, mlir.Location) mlir.Operation,
    ctx: LowerContext,
    eqn: pr.Eqn,
) void {
    const ins = ctx.inputs(eqn);
    const outs = ctx.outputs(eqn);
    const lhs = ctx.get_value(ins[0]).?;
    const rhs = ctx.get_value(ins[1]).?;
    const op = op_fn(ctx.mlir_ctx, lhs, rhs, ctx.loc);
    ctx.block.append_operation(op);
    ctx.set_value(outs[0], op.result(0));
}

// --- Unary ---

fn lower_unary(
    comptime op_fn: fn (mlir.Context, mlir.Value, mlir.Location) mlir.Operation,
    ctx: LowerContext,
    eqn: pr.Eqn,
) void {
    const ins = ctx.inputs(eqn);
    const outs = ctx.outputs(eqn);
    const operand = ctx.get_value(ins[0]).?;
    const op = op_fn(ctx.mlir_ctx, operand, ctx.loc);
    ctx.block.append_operation(op);
    ctx.set_value(outs[0], op.result(0));
}

fn lower_convert(ctx: LowerContext, eqn: pr.Eqn) LowerError!void {
    const ins = ctx.inputs(eqn);
    const outs = ctx.outputs(eqn);
    const operand = ctx.get_value(ins[0]) orelse return error.InvalidProgram;
    const out_tensor = try ctx.tensor_of(outs[0]);
    const out_type = try ctx.tensor_to_mlir_type(out_tensor);
    const op = stablehlo.convert(ctx.mlir_ctx, operand, out_type, ctx.loc);
    ctx.block.append_operation(op);
    ctx.set_value(outs[0], op.result(0));
}

// --- Constant ---

fn lower_literal(ctx: LowerContext, eqn: pr.Eqn) LowerError!void {
    const outs = ctx.outputs(eqn);
    const eqn_params = ctx.params(eqn);
    const out_id = outs[0];
    const out_tensor = try ctx.tensor_of(out_id);
    const lit = pr.param_literal(eqn_params) orelse return error.InvalidProgram;
    const elem_type = dtype_to_dense_elements_type(out_tensor.dtype);
    const raw_bytes = switch (lit) {
        inline else => |v| std.mem.asBytes(&v),
    };
    const op = stablehlo.constant(ctx.mlir_ctx, &.{}, elem_type, raw_bytes, ctx.loc);
    ctx.block.append_operation(op);
    ctx.set_value(out_id, op.result(0));
}

// --- Shape ops ---

fn lower_reshape(ctx: LowerContext, eqn: pr.Eqn) LowerError!void {
    const ins = ctx.inputs(eqn);
    const outs = ctx.outputs(eqn);
    const operand = ctx.get_value(ins[0]) orelse return error.InvalidProgram;
    const out_tensor = try ctx.tensor_of(outs[0]);
    const out_type = try ctx.tensor_to_mlir_type(out_tensor);
    const op = stablehlo.reshape(ctx.mlir_ctx, operand, out_type, ctx.loc);
    ctx.block.append_operation(op);
    ctx.set_value(outs[0], op.result(0));
}

fn lower_transpose(ctx: LowerContext, eqn: pr.Eqn) LowerError!void {
    const ins = ctx.inputs(eqn);
    const outs = ctx.outputs(eqn);
    const eqn_params = ctx.params(eqn);
    const perm = pr.param_permutation(eqn_params) orelse return error.InvalidProgram;
    const operand = ctx.get_value(ins[0]) orelse return error.InvalidProgram;
    const out_tensor = try ctx.tensor_of(outs[0]);
    const out_type = try ctx.tensor_to_mlir_type(out_tensor);
    const op = stablehlo.transpose(ctx.mlir_ctx, operand, out_type, ctx.loc, .{ .permutation = perm });
    ctx.block.append_operation(op);
    ctx.set_value(outs[0], op.result(0));
}

fn lower_broadcast_in_dim(ctx: LowerContext, eqn: pr.Eqn) LowerError!void {
    const ins = ctx.inputs(eqn);
    const outs = ctx.outputs(eqn);
    const eqn_params = ctx.params(eqn);
    const bd = pr.param_broadcast_dims(eqn_params) orelse return error.InvalidProgram;
    const operand = ctx.get_value(ins[0]) orelse return error.InvalidProgram;
    const out_tensor = try ctx.tensor_of(outs[0]);
    const out_type = try ctx.tensor_to_mlir_type(out_tensor);
    const op = stablehlo.broadcast_in_dim(ctx.mlir_ctx, operand, bd, out_type, ctx.loc);
    ctx.block.append_operation(op);
    ctx.set_value(outs[0], op.result(0));
}

fn lower_iota(ctx: LowerContext, eqn: pr.Eqn) LowerError!void {
    const outs = ctx.outputs(eqn);
    const eqn_params = ctx.params(eqn);
    const iota_dim = pr.param_iota_dimension(eqn_params) orelse return error.InvalidProgram;
    const out_tensor = try ctx.tensor_of(outs[0]);
    const out_type = try ctx.tensor_to_mlir_type(out_tensor);
    const op = stablehlo.iota(ctx.mlir_ctx, iota_dim, out_type, ctx.loc);
    ctx.block.append_operation(op);
    ctx.set_value(outs[0], op.result(0));
}

fn lower_slice(ctx: LowerContext, eqn: pr.Eqn) LowerError!void {
    const ins = ctx.inputs(eqn);
    const outs = ctx.outputs(eqn);
    const eqn_params = ctx.params(eqn);
    const sparams = pr.param_slice(eqn_params) orelse return error.InvalidProgram;
    const operand = ctx.get_value(ins[0]) orelse return error.InvalidProgram;
    const out_tensor = try ctx.tensor_of(outs[0]);
    const out_type = try ctx.tensor_to_mlir_type(out_tensor);
    const op = stablehlo.slice(
        ctx.mlir_ctx,
        operand,
        sparams.start_indices,
        sparams.limit_indices,
        sparams.strides,
        out_type,
        ctx.loc,
    );
    ctx.block.append_operation(op);
    ctx.set_value(outs[0], op.result(0));
}

fn lower_concatenate(ctx: LowerContext, eqn: pr.Eqn) LowerError!void {
    const ins = ctx.inputs(eqn);
    const outs = ctx.outputs(eqn);
    const eqn_params = ctx.params(eqn);
    const axis = pr.param_concat_axis(eqn_params) orelse return error.InvalidProgram;
    const values = ctx.arena.alloc(mlir.Value, ins.len) catch return error.OutOfMemory;
    for (ins, 0..) |id, i| {
        values[i] = ctx.get_value(id) orelse return error.InvalidProgram;
    }
    const op = stablehlo.concatenate(ctx.mlir_ctx, values, axis, ctx.loc);
    ctx.block.append_operation(op);
    ctx.set_value(outs[0], op.result(0));
}

// --- Reduction ---

fn lower_reduce_sum(ctx: LowerContext, eqn: pr.Eqn) LowerError!void {
    const ins = ctx.inputs(eqn);
    const outs = ctx.outputs(eqn);
    const eqn_params = ctx.params(eqn);
    const axes = pr.param_reduce_axes(eqn_params) orelse return error.InvalidProgram;
    const operand = ctx.get_value(ins[0]) orelse return error.InvalidProgram;
    const out_tensor = try ctx.tensor_of(outs[0]);
    const elem_type = dtype_to_dense_elements_type(out_tensor.dtype);
    const zero_bytes = scalar_zero_bytes(out_tensor.dtype);
    const zero_op = stablehlo.constant(ctx.mlir_ctx, &.{}, elem_type, zero_bytes, ctx.loc);
    ctx.block.append_operation(zero_op);
    const op = stablehlo.reduce(ctx.mlir_ctx, &.{operand}, &.{zero_op.result(0)}, axes, {}, reduce_add_block, ctx.loc);
    ctx.block.append_operation(op);
    ctx.set_value(outs[0], op.result(0));
}

fn lower_reduce_max(ctx: LowerContext, eqn: pr.Eqn) LowerError!void {
    const ins = ctx.inputs(eqn);
    const outs = ctx.outputs(eqn);
    const eqn_params = ctx.params(eqn);
    const axes = pr.param_reduce_axes(eqn_params) orelse return error.InvalidProgram;
    const operand = ctx.get_value(ins[0]) orelse return error.InvalidProgram;
    const out_tensor = try ctx.tensor_of(outs[0]);
    const elem_type = dtype_to_dense_elements_type(out_tensor.dtype);
    const min_bytes = scalar_min_bytes(out_tensor.dtype);
    const min_op = stablehlo.constant(ctx.mlir_ctx, &.{}, elem_type, min_bytes, ctx.loc);
    ctx.block.append_operation(min_op);
    const op = stablehlo.reduce(ctx.mlir_ctx, &.{operand}, &.{min_op.result(0)}, axes, {}, reduce_max_block, ctx.loc);
    ctx.block.append_operation(op);
    ctx.set_value(outs[0], op.result(0));
}

fn reduce_add_block(_: anytype, ctx: mlir.Context, ins: []const mlir.Value, accs: []const mlir.Value) mlir.Operation {
    return stablehlo.add(ctx, ins[0], accs[0], mlir.Location.unknown(ctx));
}

fn reduce_max_block(_: anytype, ctx: mlir.Context, ins: []const mlir.Value, accs: []const mlir.Value) mlir.Operation {
    return stablehlo.maximum(ctx, ins[0], accs[0], mlir.Location.unknown(ctx));
}

// --- Contraction ---

fn lower_dot(ctx: LowerContext, eqn: pr.Eqn) LowerError!void {
    const ins = ctx.inputs(eqn);
    const outs = ctx.outputs(eqn);
    const lhs = ctx.get_value(ins[0]) orelse return error.InvalidProgram;
    const rhs = ctx.get_value(ins[1]) orelse return error.InvalidProgram;
    const out_tensor = try ctx.tensor_of(outs[0]);
    const out_type = try ctx.tensor_to_mlir_type(out_tensor);
    const op = stablehlo.dot_general(ctx.mlir_ctx, lhs, rhs, out_type, ctx.loc, .{
        .lhs_batching_dimensions = &.{},
        .rhs_batching_dimensions = &.{},
        .lhs_contracting_dimensions = &.{1},
        .rhs_contracting_dimensions = &.{0},
        .precision = .fast,
    });
    ctx.block.append_operation(op);
    ctx.set_value(outs[0], op.result(0));
}

fn lower_dot_general(ctx: LowerContext, eqn: pr.Eqn) LowerError!void {
    const ins = ctx.inputs(eqn);
    const outs = ctx.outputs(eqn);
    const eqn_params = ctx.params(eqn);
    const dg = pr.param_dot_general(eqn_params) orelse return error.InvalidProgram;
    const lhs = ctx.get_value(ins[0]) orelse return error.InvalidProgram;
    const rhs = ctx.get_value(ins[1]) orelse return error.InvalidProgram;
    const out_tensor = try ctx.tensor_of(outs[0]);
    const out_type = try ctx.tensor_to_mlir_type(out_tensor);
    const op = stablehlo.dot_general(ctx.mlir_ctx, lhs, rhs, out_type, ctx.loc, .{
        .lhs_batching_dimensions = dg.lhs_batch_dims,
        .rhs_batching_dimensions = dg.rhs_batch_dims,
        .lhs_contracting_dimensions = dg.lhs_contracting_dims,
        .rhs_contracting_dimensions = dg.rhs_contracting_dims,
        .precision = .fast,
    });
    ctx.block.append_operation(op);
    ctx.set_value(outs[0], op.result(0));
}

// --- Compare ---

fn lower_compare(ctx: LowerContext, eqn: pr.Eqn) LowerError!void {
    const ins = ctx.inputs(eqn);
    const outs = ctx.outputs(eqn);
    const eqn_params = ctx.params(eqn);
    const cparams = pr.param_compare(eqn_params) orelse return error.InvalidProgram;
    const lhs = ctx.get_value(ins[0]) orelse return error.InvalidProgram;
    const rhs = ctx.get_value(ins[1]) orelse return error.InvalidProgram;
    const op = stablehlo.compare(
        ctx.mlir_ctx,
        lhs,
        rhs,
        stablehlo.ComparisonDirection.init(ctx.mlir_ctx, map_compare_direction(cparams.direction)),
        stablehlo.CompareType.init(ctx.mlir_ctx, map_compare_type(cparams.compare_type)),
        ctx.loc,
    );
    ctx.block.append_operation(op);
    ctx.set_value(outs[0], op.result(0));
}

fn lower_select(ctx: LowerContext, eqn: pr.Eqn) LowerError!void {
    const ins = ctx.inputs(eqn);
    const outs = ctx.outputs(eqn);
    const cond = ctx.get_value(ins[0]) orelse return error.InvalidProgram;
    const on_true = ctx.get_value(ins[1]) orelse return error.InvalidProgram;
    const on_false = ctx.get_value(ins[2]) orelse return error.InvalidProgram;
    const op = stablehlo.select(ctx.mlir_ctx, cond, on_true, on_false, ctx.loc);
    ctx.block.append_operation(op);
    ctx.set_value(outs[0], op.result(0));
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

fn lower_gather(ctx: LowerContext, eqn: pr.Eqn) LowerError!void {
    const ins = ctx.inputs(eqn);
    const outs = ctx.outputs(eqn);
    const eqn_params = ctx.params(eqn);
    const gparams = pr.param_gather(eqn_params) orelse return error.InvalidProgram;
    const operand = ctx.get_value(ins[0]) orelse return error.InvalidProgram;
    const indices = ctx.get_value(ins[1]) orelse return error.InvalidProgram;
    const op = stablehlo.gather(ctx.mlir_ctx, operand, indices, gparams.slice_sizes, ctx.loc, .{
        .offset_dims = gparams.offset_dims,
        .collapsed_slice_dims = gparams.collapsed_slice_dims,
        .operand_batching_dims = &.{},
        .start_indices_batching_dims = &.{},
        .start_index_map = gparams.start_index_map,
        .index_vector_dim = gparams.index_vector_dim,
    });
    ctx.block.append_operation(op);
    ctx.set_value(outs[0], op.result(0));
}

fn lower_scatter(ctx: LowerContext, eqn: pr.Eqn) LowerError!void {
    const ins = ctx.inputs(eqn);
    const outs = ctx.outputs(eqn);
    const eqn_params = ctx.params(eqn);
    const sparams = pr.param_scatter(eqn_params) orelse return error.InvalidProgram;
    const operand = ctx.get_value(ins[0]) orelse return error.InvalidProgram;
    const indices = ctx.get_value(ins[1]) orelse return error.InvalidProgram;
    const updates = ctx.get_value(ins[2]) orelse return error.InvalidProgram;
    const update_block = make_update_block(ctx.mlir_ctx, operand.get_type(), ctx.loc, sparams.reduction);
    const op = stablehlo.scatter(
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
    ctx.block.append_operation(op);
    ctx.set_value(outs[0], op.result(0));
}

fn make_update_block(ctx: mlir.Context, operand_type: mlir.Type, loc: mlir.Location, reduction: pr.ScatterReduction) mlir.Block {
    const elem_type = if (operand_type.as(mlir.RankedTensorType)) |shaped| shaped.get_element_type() else operand_type;
    const arg_type: mlir.Type = .tensor(&.{}, elem_type);
    var block = mlir.Block.init(&.{ arg_type, arg_type }, &.{ loc, loc }) catch unreachable;
    const op = switch (reduction) {
        .add => stablehlo.add(ctx, block.argument(0), block.argument(1), loc),
        .max => stablehlo.maximum(ctx, block.argument(0), block.argument(1), loc),
        .min => stablehlo.minimum(ctx, block.argument(0), block.argument(1), loc),
        .mul => stablehlo.multiply(ctx, block.argument(0), block.argument(1), loc),
    };
    block.append_operation(op);
    const ret = stablehlo.return_(ctx, op.result(0), loc);
    block.append_operation(ret);
    return block;
}

// --- Special (call / custom_call) ---

fn lower_call(ctx: LowerContext, eqn: pr.Eqn) LowerError!void {
    const ins = ctx.inputs(eqn);
    const outs = ctx.outputs(eqn);
    const eqn_params = ctx.params(eqn);
    const callee = pr.param_call_callee(eqn_params) orelse return error.InvalidProgram;

    const operand_values = ctx.arena.alloc(mlir.Value, ins.len) catch return error.OutOfMemory;
    for (ins, 0..) |id, i| {
        operand_values[i] = ctx.get_value(id) orelse return error.InvalidProgram;
    }

    const result_types = ctx.arena.alloc(mlir.Type, outs.len) catch return error.OutOfMemory;
    for (outs, 0..) |out_id, i| {
        const out_tensor = try ctx.tensor_of(out_id);
        result_types[i] = try ctx.tensor_to_mlir_type(out_tensor);
    }

    const callee_z = ctx.arena.allocSentinel(u8, callee.len, 0) catch return error.OutOfMemory;
    @memcpy(callee_z, callee);

    const op = mlir.Operation.make(ctx.mlir_ctx, "func.call", .{
        .results = result_types,
        .operands = operand_values,
        .attributes = &.{
            .{ "callee", mlir.Attribute.symbol(ctx.mlir_ctx, callee_z) },
        },
        .verify = false,
        .location = ctx.loc,
    });

    ctx.block.append_operation(op);
    for (outs, 0..) |out_id, i| {
        ctx.set_value(out_id, op.result(i));
    }
}

fn lower_custom_call(ctx: LowerContext, eqn: pr.Eqn) LowerError!void {
    const ins = ctx.inputs(eqn);
    const outs = ctx.outputs(eqn);
    const eqn_params = ctx.params(eqn);
    if (outs.len == 0) return error.InvalidProgram;

    const target = pr.param_call_target_name(eqn_params) orelse return error.InvalidProgram;
    const has_side_effect = pr.param_has_side_effect(eqn_params) orelse return error.InvalidProgram;
    const kernel_key = pr.param_call_kernel_key(eqn_params);
    const kernel_id = pr.param_call_kernel_id(eqn_params);
    const provider_name = pr.param_call_provider_name(eqn_params);
    const carrier_hint = pr.param_call_carrier_hint(eqn_params);

    const result_types = ctx.arena.alloc(mlir.Type, outs.len) catch return error.OutOfMemory;
    for (outs, 0..) |out_id, i| {
        const out_tensor = try ctx.tensor_of(out_id);
        result_types[i] = try ctx.tensor_to_mlir_type(out_tensor);
    }

    const operand_values = ctx.arena.alloc(mlir.Value, ins.len) catch return error.OutOfMemory;
    for (ins, 0..) |id, i| {
        operand_values[i] = ctx.get_value(id) orelse return error.InvalidProgram;
    }

    // typed_ffi custom calls expect dictionary backend_config. Keep keys stable
    // to match backend dispatcher parsing.
    var backend_fields: [4]mlir.AttrTuple = undefined;
    var backend_field_count: usize = 0;
    if (kernel_key) |value| {
        backend_fields[backend_field_count] = .{ "zigrad.kernel_key", mlir.Attribute.string(ctx.mlir_ctx, value) };
        backend_field_count += 1;
    }
    if (kernel_id) |value| {
        backend_fields[backend_field_count] = .{ "zigrad.kernel_id", mlir.Attribute.int(ctx.mlir_ctx, .i64, @intCast(value)) };
        backend_field_count += 1;
    }
    if (provider_name) |value| {
        backend_fields[backend_field_count] = .{ "zigrad.provider", mlir.Attribute.string(ctx.mlir_ctx, value) };
        backend_field_count += 1;
    }
    if (carrier_hint) |value| {
        backend_fields[backend_field_count] = .{ "zigrad.carrier_hint", mlir.Attribute.string(ctx.mlir_ctx, value) };
        backend_field_count += 1;
    }
    const backend_config = mlir.Attribute.dict(ctx.mlir_ctx, backend_fields[0..backend_field_count]);

    const op = mlir.Operation.make(ctx.mlir_ctx, zigrad_kernel_call_op_name, .{
        .results = result_types,
        .operands = operand_values,
        .attributes = &.{
            .{ "api_version", mlir.Attribute.int(ctx.mlir_ctx, .i32, @intFromEnum(stablehlo.CustomCallOpts.ApiVersion.typed_ffi)) },
            .{ "call_target_name", mlir.Attribute.string(ctx.mlir_ctx, target) },
            .{ "has_side_effect", mlir.Attribute.boolean(ctx.mlir_ctx, has_side_effect) },
            .{ "backend_config", backend_config },
        },
        .verify = false,
        .location = ctx.loc,
    });

    ctx.block.append_operation(op);
    for (outs, 0..) |out_id, i| {
        ctx.set_value(out_id, op.result(i));
    }
}

// ============================================================================
// Type Mapping Helpers
// ============================================================================

fn dtype_to_mlir_type(ctx: mlir.Context, dt: pr.DType) mlir.Type {
    return switch (dt) {
        .bf16 => mlir.Type.float(ctx, .bf16),
        .f32 => mlir.Type.float(ctx, .f32),
        .f64 => mlir.Type.float(ctx, .f64),
        .i32 => mlir.Type.int(ctx, .i32),
        .i64 => mlir.Type.int(ctx, .i64),
        .u32 => mlir.Type.int(ctx, .i32),
        .u64 => mlir.Type.int(ctx, .i64),
        .bool => mlir.Type.int(ctx, .i1),
    };
}

fn dtype_to_dense_elements_type(dt: pr.DType) mlir.DenseElementsAttributeTypes {
    return switch (dt) {
        .bf16 => .bf16,
        .f32 => .f32,
        .f64 => .f64,
        .i32 => .i32,
        .i64 => .i64,
        .u32 => .i32,
        .u64 => .i64,
        .bool => .bool,
    };
}

fn tensor_to_mlir_type_standalone(ctx: mlir.Context, t: pr.Tensor, arena: std.mem.Allocator) LowerError!mlir.Type {
    const dims_i64 = arena.alloc(i64, t.shape.dims.len) catch return error.OutOfMemory;
    for (t.shape.dims, 0..) |d, i| dims_i64[i] = @intCast(d);
    return mlir.Type.tensor(dims_i64, dtype_to_mlir_type(ctx, t.dtype));
}

fn scalar_zero_bytes(dtype: pr.DType) []const u8 {
    return switch (dtype) {
        .bf16 => std.mem.asBytes(&@as(u16, 0)),
        .f32 => std.mem.asBytes(&@as(f32, 0.0)),
        .f64 => std.mem.asBytes(&@as(f64, 0.0)),
        .i32 => std.mem.asBytes(&@as(i32, 0)),
        .i64 => std.mem.asBytes(&@as(i64, 0)),
        .u32 => std.mem.asBytes(&@as(u32, 0)),
        .u64 => std.mem.asBytes(&@as(u64, 0)),
        .bool => std.mem.asBytes(&@as(bool, false)),
    };
}

fn scalar_min_bytes(dtype: pr.DType) []const u8 {
    return switch (dtype) {
        .bf16 => std.mem.asBytes(&f32_to_bf16_bits(-std.math.inf(f32))),
        .f32 => std.mem.asBytes(&@as(f32, -std.math.inf(f32))),
        .f64 => std.mem.asBytes(&@as(f64, -std.math.inf(f64))),
        .i32 => std.mem.asBytes(&std.math.minInt(i32)),
        .i64 => std.mem.asBytes(&std.math.minInt(i64)),
        .u32 => std.mem.asBytes(&@as(u32, 0)),
        .u64 => std.mem.asBytes(&@as(u64, 0)),
        .bool => std.mem.asBytes(&@as(bool, false)),
    };
}

fn f32_to_bf16_bits(val: f32) u16 {
    const bits: u32 = @bitCast(val);
    return @intCast(bits >> 16);
}

// ============================================================================
// Pass Integration
// ============================================================================

/// Lower pass: PR artifact -> MLIR artifact.
pub const LowerPassConfig = struct {
    encoding: pass.MlirEncoding = .bytecode,

    /// Selects which PR function is the compilation entry point.
    /// The selected function is always renamed to "@main" in MLIR output (XLA requirement).
    entry_name: ?[]const u8 = null,

    kernelization_lane: KernelizationLane = .pr,
};

pub fn lower_pass(ptr: *anyopaque, artifact: *pass.Artifact, ctx: *pass.PassContext) pass.PassError!void {
    if (artifact.kind() != .pr) return error.ArtifactKindMismatch;

    const cfg: *LowerPassConfig = @ptrCast(@alignCast(ptr));
    const program = artifact.pr;

    const format: OutputFormat = switch (cfg.encoding) {
        .text => .mlir_text,
        .bytecode => .mlir_bytecode,
    };

    const bytes = try lower_program_impl(
        ctx.allocator,
        program,
        cfg.entry_name,
        format,
        cfg.kernelization_lane,
    );

    artifact.replace(ctx.allocator, .{
        .mlir = .{
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
        .output_kind = .mlir,
    };
}

/// Validate pass: PR artifact -> PR artifact.
fn validate_pass_run(_: *anyopaque, artifact: *pass.Artifact, ctx: *pass.PassContext) pass.PassError!void {
    _ = ctx;

    if (artifact.kind() != .pr) return error.ArtifactKindMismatch;

    const program = artifact.pr;
    pr.validate_program(program) catch return error.ValidationFailed;
}

/// Metadata for the validate pass.
pub const validate_pass = pass.Pass{
    .ptr = undefined,
    .run_fn = validate_pass_run,
    .name = "pr_validate",
    .input_kind = .pr,
    .output_kind = .pr,
};

/// Convenience: lower with encoding preference.
pub fn lower(
    allocator: std.mem.Allocator,
    program: *const pr.Program,
    entry_name: ?[]const u8,
    encoding: pass.MlirEncoding,
) !pass.MlirArtifact {
    const format: OutputFormat = switch (encoding) {
        .text => .mlir_text,
        .bytecode => .mlir_bytecode,
    };

    const bytes = try lower_program_to_mlir(allocator, program, entry_name, format);

    return .{
        .bytes = bytes,
        .encoding = encoding,
    };
}

// ============================================================================
// Utility
// ============================================================================

fn find_entry_function(program: *const pr.Program, entry_name: ?[]const u8) LowerError!usize {
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

fn choose_symbol_name(
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
        const candidate = std.fmt.allocPrint(arena, "main_non_entry_{d}", .{suffix}) catch return error.OutOfMemory;
        if (!is_symbol_name_used(program, entry_index, candidate)) {
            log.warn("renaming non-entry function 'main' to '{s}' to avoid entry collision", .{candidate});
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

// ============================================================================
// Tests
// ============================================================================

test "lowering produces verified bytecode" {
    var program = try @import("../frontend/frontend.zig").build_demo_program(std.testing.allocator);
    defer program.deinit();

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
    const ex = try b.emit(.exp, &.{x}, &.{});
    const lg = try b.emit(.log, &.{y}, &.{});
    try b.pop_region();

    const func = try b.finish(&.{ ex, lg });
    try program.add_function(func);

    const kernel = @import("../kernel.zig");
    const kernelize = @import("../pipeline/kernelize.zig");

    const MockProvider = struct {
        fn compile(_: *anyopaque, desc: kernel.RegionDescriptor, allocator: std.mem.Allocator) kernel.CompileError!kernel.KernelArtifact {
            return .{
                .provider_name = "mock",
                .data = try allocator.dupe(u8, "mock"),
                .target_name = try allocator.dupe(u8, desc.name),
            };
        }
    };

    const provider = kernel.KernelProvider{
        .name = "mock",
        .ptr = undefined,
        .compile_fn = MockProvider.compile,
    };

    var registry = kernel.KernelRegistry.init(testing.allocator);
    defer registry.deinit();

    var kp = kernelize.KernelizePass{
        .registry = &registry,
        .providers = &.{provider},
    };

    var artifact = pass.Artifact{ .pr = &program };
    var pass_ctx = pass.PassContext{ .allocator = testing.allocator };
    try kp.pass().run(&artifact, &pass_ctx);

    const text = try lower_program_to_mlir(testing.allocator, &program, null, .mlir_text);
    defer testing.allocator.free(text);

    try testing.expect(std.mem.indexOf(u8, text, "stablehlo.custom_call") != null);
    try testing.expect(std.mem.indexOf(u8, text, "tensor<2xf32>, tensor<2xf32>") != null);
}

test "lowering emits kernel_id and carrier_hint in custom_call backend_config" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const lhs = try b.param_tensor(.f32, &.{ 2, 3 });
    const rhs = try b.param_tensor(.f32, &.{ 3, 2 });
    const bias = try b.param_tensor(.f32, &.{ 2, 2 });
    const scale = try b.param_tensor(.f32, &.{ 2, 2 });

    try b.push_region("attention_like", .{ .kernelize = "mock" });
    const dot = try b.emit(.dot, &.{ lhs, rhs }, &.{});
    const sum = try b.emit(.add, &.{ dot, bias }, &.{});
    const ex = try b.emit(.exp, &.{sum}, &.{});
    const mul = try b.emit(.multiply, &.{ ex, scale }, &.{});
    const out = try b.emit(.log, &.{mul}, &.{});
    try b.pop_region();

    const func = try b.finish(&.{out});
    try program.add_function(func);

    const kernel = @import("../kernel.zig");
    const kernelize = @import("../pipeline/kernelize.zig");

    const MockProvider = struct {
        fn compile(_: *anyopaque, desc: kernel.RegionDescriptor, allocator: std.mem.Allocator) kernel.CompileError!kernel.KernelArtifact {
            return .{
                .provider_name = "mock",
                .data = try allocator.dupe(u8, "mock"),
                .target_name = try allocator.dupe(u8, desc.name),
            };
        }
    };

    const provider = kernel.KernelProvider{
        .name = "mock",
        .ptr = undefined,
        .compile_fn = MockProvider.compile,
    };

    var registry = kernel.KernelRegistry.init(testing.allocator);
    defer registry.deinit();

    var kp = kernelize.KernelizePass{
        .registry = &registry,
        .providers = &.{provider},
    };

    var artifact = pass.Artifact{ .pr = &program };
    var pass_ctx = pass.PassContext{ .allocator = testing.allocator };
    try kp.pass().run(&artifact, &pass_ctx);

    const text = try lower_program_to_mlir(testing.allocator, &program, null, .mlir_text);
    defer testing.allocator.free(text);

    try testing.expect(std.mem.indexOf(u8, text, "zigrad.kernel_id") != null);
    try testing.expect(std.mem.indexOf(u8, text, "zigrad.carrier_hint") != null);
}

test "lowering supports vjp matmul demo" {
    var program = try @import("../frontend/frontend.zig").build_demo_program(std.testing.allocator);
    defer program.deinit();

    const fwd = program.functions[0];
    const vjp_func = try @import("../pr/ad.zig").vjp(std.testing.allocator, &program, fwd, "vjp");

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

    try std.testing.expect(std.mem.indexOf(u8, text, "func.call") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "main_outlined_0") != null);
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

test "lower pass mlir lane tags markers without outlining" {
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

    // Lower pass is baseline only — no select/legalize.
    var cfg = LowerPassConfig{ .encoding = .text, .kernelization_lane = .mlir };
    var artifact = pass.Artifact{ .pr = &program };
    var pass_ctx = pass.PassContext{ .allocator = testing.allocator };

    try lower_pass(@ptrCast(&cfg), &artifact, &pass_ctx);
    defer artifact.deinit(testing.allocator);

    // Marker attributes present on ops (for select pass to consume later).
    try testing.expect(std.mem.indexOf(u8, artifact.mlir.bytes, "zigrad.kernelize.provider") != null);
    try testing.expect(std.mem.indexOf(u8, artifact.mlir.bytes, "zigrad.kernelize.region") != null);

    // No outlining (MLIR lane keeps ops inline).
    try testing.expect(std.mem.indexOf(u8, artifact.mlir.bytes, "main_outlined_0") == null);
    try testing.expect(std.mem.indexOf(u8, artifact.mlir.bytes, "func.call") == null);

    // Original ops preserved — select/legalize are separate passes.
    try testing.expect(std.mem.indexOf(u8, artifact.mlir.bytes, "stablehlo.dot_general") != null);
    try testing.expect(std.mem.indexOf(u8, artifact.mlir.bytes, "stablehlo.add") != null);
    try testing.expect(std.mem.indexOf(u8, artifact.mlir.bytes, "stablehlo.multiply") != null);
}

test "lower pass mlir lane tags dot-add chain markers" {
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

    var cfg = LowerPassConfig{ .encoding = .text, .kernelization_lane = .mlir };
    var artifact = pass.Artifact{ .pr = &program };
    var pass_ctx = pass.PassContext{ .allocator = testing.allocator };

    try lower_pass(@ptrCast(&cfg), &artifact, &pass_ctx);
    defer artifact.deinit(testing.allocator);

    // Markers present; original ops preserved (select/legalize are separate).
    try testing.expect(std.mem.indexOf(u8, artifact.mlir.bytes, "zigrad.kernelize.provider") != null);
    try testing.expect(std.mem.indexOf(u8, artifact.mlir.bytes, "zigrad.kernelize.region") != null);
    try testing.expect(std.mem.indexOf(u8, artifact.mlir.bytes, "stablehlo.dot_general") != null);
    try testing.expect(std.mem.indexOf(u8, artifact.mlir.bytes, "stablehlo.add") != null);
}

test "lower pass mlir lane tags dot-log chain markers" {
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

    var cfg = LowerPassConfig{ .encoding = .text, .kernelization_lane = .mlir };
    var artifact = pass.Artifact{ .pr = &program };
    var pass_ctx = pass.PassContext{ .allocator = testing.allocator };

    try lower_pass(@ptrCast(&cfg), &artifact, &pass_ctx);
    defer artifact.deinit(testing.allocator);

    // Markers present; original ops preserved.
    try testing.expect(std.mem.indexOf(u8, artifact.mlir.bytes, "zigrad.kernelize.provider") != null);
    try testing.expect(std.mem.indexOf(u8, artifact.mlir.bytes, "stablehlo.dot_general") != null);
    try testing.expect(std.mem.indexOf(u8, artifact.mlir.bytes, "stablehlo.log") != null);
}

test "lower pass mlir lane tags near-miss region markers" {
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

    var cfg = LowerPassConfig{ .encoding = .text, .kernelization_lane = .mlir };
    var artifact = pass.Artifact{ .pr = &program };
    var pass_ctx = pass.PassContext{ .allocator = testing.allocator };

    try lower_pass(@ptrCast(&cfg), &artifact, &pass_ctx);
    defer artifact.deinit(testing.allocator);

    // Markers present; all original ops preserved (near-miss won't match select).
    try testing.expect(std.mem.indexOf(u8, artifact.mlir.bytes, "zigrad.kernelize.provider") != null);
    try testing.expect(std.mem.indexOf(u8, artifact.mlir.bytes, "zigrad.kernelize.region") != null);
    try testing.expect(std.mem.indexOf(u8, artifact.mlir.bytes, "stablehlo.dot_general") != null);
    try testing.expect(std.mem.indexOf(u8, artifact.mlir.bytes, "stablehlo.subtract") != null);
    try testing.expect(std.mem.indexOf(u8, artifact.mlir.bytes, "stablehlo.multiply") != null);
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
    };
    var cfg = LowerPassConfig{ .encoding = .bytecode };

    try program.add_function(func);
    var output = pass.Artifact{ .pr = &program };
    try lower_pass(@ptrCast(&cfg), &output, &ctx);
    defer output.deinit(std.testing.allocator);

    try std.testing.expectEqual(pass.ArtifactKind.mlir, output.kind());
    try std.testing.expect(output.mlir.bytes.len > 0);
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

    // Lower pass is baseline — no legalize.
    var cfg = LowerPassConfig{ .encoding = .text };
    var artifact = pass.Artifact{ .pr = &program };
    var pass_ctx = pass.PassContext{ .allocator = testing.allocator };
    try lower_pass(@ptrCast(&cfg), &artifact, &pass_ctx);
    defer artifact.deinit(testing.allocator);

    try testing.expectEqual(pass.ArtifactKind.mlir, artifact.kind());

    // Custom calls are emitted as zigrad.kernel_call (legalize converts to stablehlo.custom_call).
    try testing.expect(std.mem.indexOf(u8, artifact.mlir.bytes, zigrad_kernel_call_op_name) != null);
    try testing.expect(std.mem.indexOf(u8, artifact.mlir.bytes, "stablehlo.custom_call") == null);
}
