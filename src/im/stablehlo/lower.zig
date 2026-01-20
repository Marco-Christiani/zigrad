const std = @import("std");

const pr = @import("../../pr/pr.zig");
const ops = @import("../../pr/ops/ops.zig");
const mlir = @import("../../ffi/mlir/mlir.zig");

pub const LowerError = ops.types.LowerError;

pub const OutputFormat = enum {
    mlir_text,
    mlir_bytecode,
};

fn lowerFunctionToMlirInternal(allocator: std.mem.Allocator, func: pr.Function, comptime out: OutputFormat) ![]u8 {
    pr.validateFunction(func) catch return error.InvalidProgram;

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var registry = try mlir.Registry.init();
    defer registry.deinit();

    mlir.DialectHandle.fromString("func").insertDialect(registry);
    mlir.DialectHandle.fromString("stablehlo").insertDialect(registry);

    var ctx = try mlir.Context.initWithRegistry(registry, false);
    defer ctx.deinit();
    ctx.allowUnregisteredDialects(false);

    const func_handle = mlir.DialectHandle.fromString("func");
    func_handle.registerDialect(ctx);
    _ = func_handle.loadDialect(ctx);

    const stablehlo_handle = mlir.DialectHandle.fromString("stablehlo");
    stablehlo_handle.registerDialect(ctx);
    _ = stablehlo_handle.loadDialect(ctx);

    const loc = mlir.Location.unknown(ctx);

    var module = mlir.Module.init(loc);
    defer module.deinit();

    const param_types = try arena.alloc(mlir.Type, func.params.len);
    const param_locs = try arena.alloc(mlir.Location, func.params.len);
    for (func.params, 0..) |param_id, i| {
        const tensor = func.avals[@intCast(param_id)].asTensor() orelse return error.InvalidProgram;
        param_types[i] = try tensorToMlirType(ctx, tensor, arena);
        param_locs[i] = loc;
    }

    const result_types = try arena.alloc(mlir.Type, func.returns.len);
    for (func.returns, 0..) |ret_id, i| {
        const tensor = func.avals[@intCast(ret_id)].asTensor() orelse return error.InvalidProgram;
        result_types[i] = try tensorToMlirType(ctx, tensor, arena);
    }

    const fn_type = mlir.Type.function(ctx, param_types, result_types);

    const entry_block = try mlir.Block.init(param_types, param_locs);

    const value_map = try arena.alloc(?mlir.Value, func.avals.len);
    @memset(value_map, null);
    for (func.params, 0..) |param_id, i| {
        value_map[@intCast(param_id)] = entry_block.argument(i);
    }

    // Use ops dispatch for lowering
    const lower_ctx = ops.types.LowerContext{
        .mlir_ctx = ctx,
        .block = entry_block,
        .loc = loc,
        .value_map = value_map,
        .func = func,
        .arena = arena,
    };

    var outlined_index: usize = 0;
    for (func.eqns) |eqn| {
        if (shouldOutlineEqn(lower_ctx, eqn)) {
            try lowerOutlinedEqn(arena, &outlined_index, ctx, module, lower_ctx, eqn);
        } else {
            try ops.lower(lower_ctx, eqn);
        }
    }

    const ret_values = try arena.alloc(mlir.Value, func.returns.len);
    for (func.returns, 0..) |ret_id, i| {
        ret_values[i] = value_map[@intCast(ret_id)] orelse return error.InvalidProgram;
    }

    const return_op = mlir.Operation.make(ctx, "func.return", .{
        .operands = ret_values,
        .verify = false,
        .location = loc,
    });
    entry_block.appendOperation(return_op);

    const func_op = mlir.Operation.make(ctx, "func.func", .{
        .results = &.{},
        .blocks = &.{entry_block},
        .attributes = &.{
            .{ "sym_name", mlir.Attribute.string(ctx, "main") },
            .{ "function_type", mlir.Attribute.type_(fn_type) },
        },
        .verify = false,
        .location = loc,
    });
    module.getBody().appendOperation(func_op);

    if (!module.op().verify()) return error.InvalidMlir;

    var writer_state = std.Io.Writer.Allocating.init(allocator);
    defer writer_state.deinit();

    switch (out) {
        .mlir_bytecode => try module.op().writeBytecode(&writer_state.writer),
        .mlir_text => try module.op().print(&writer_state.writer, .{}),
    }

    return try writer_state.toOwnedSlice();
}

pub fn lowerFunctionToMlir(allocator: std.mem.Allocator, func: pr.Function, comptime out: OutputFormat) ![]u8 {
    return lowerFunctionToMlirInternal(allocator, func, out);
}

fn shouldOutlineEqn(ctx: ops.types.LowerContext, eqn: pr.Eqn) bool {
    const params = ctx.params(eqn);
    return pr.paramOutline(params) orelse false;
}

fn lowerOutlinedEqn(
    arena: std.mem.Allocator,
    outlined_index: *usize,
    mlir_ctx: mlir.Context,
    module: mlir.Module,
    ctx: ops.types.LowerContext,
    eqn: pr.Eqn,
) !void {
    const inputs = ctx.inputs(eqn);
    const outputs = ctx.outputs(eqn);

    // v0: all current PR ops are single-output; keep outlining strict.
    if (outputs.len != 1) return error.InvalidProgram;
    if (inputs.len == 0) return error.InvalidProgram;

    const out_id = outputs[0];
    const out_tensor = try ctx.tensorOf(out_id);
    const out_type = try ctx.tensorToMlirType(out_tensor);

    const callee_name = try std.fmt.allocPrint(arena, "outlined_{d}", .{outlined_index.*});
    outlined_index.* += 1;
    const callee_name_z = try arena.allocSentinel(u8, callee_name.len, 0);
    @memcpy(callee_name_z, callee_name);

    // Build callee signature (inputs -> output).
    const callee_param_types = try arena.alloc(mlir.Type, inputs.len);
    const callee_param_locs = try arena.alloc(mlir.Location, inputs.len);
    for (inputs, 0..) |in_id, i| {
        const in_tensor = try ctx.tensorOf(in_id);
        callee_param_types[i] = try ctx.tensorToMlirType(in_tensor);
        callee_param_locs[i] = ctx.loc;
    }
    const callee_result_types = &[_]mlir.Type{out_type};
    const callee_fn_type = mlir.Type.function(mlir_ctx, callee_param_types, callee_result_types);

    // Build callee body.
    const callee_entry = try mlir.Block.init(callee_param_types, callee_param_locs);

    const callee_value_map = try arena.alloc(?mlir.Value, ctx.func.avals.len);
    @memset(callee_value_map, null);
    for (inputs, 0..) |in_id, i| callee_value_map[@intCast(in_id)] = callee_entry.argument(i);

    const callee_ctx = ops.types.LowerContext{
        .mlir_ctx = mlir_ctx,
        .block = callee_entry,
        .loc = ctx.loc,
        .value_map = callee_value_map,
        .func = ctx.func,
        .arena = arena,
    };

    try ops.lower(callee_ctx, eqn);

    const callee_out = callee_value_map[@intCast(out_id)] orelse return error.InvalidProgram;
    const callee_ret = mlir.Operation.make(mlir_ctx, "func.return", .{
        .operands = &.{callee_out},
        .verify = false,
        .location = ctx.loc,
    });
    callee_entry.appendOperation(callee_ret);

    const callee_op = mlir.Operation.make(mlir_ctx, "func.func", .{
        .results = &.{},
        .blocks = &.{callee_entry},
        .attributes = &.{
            .{ "sym_name", mlir.Attribute.string(mlir_ctx, callee_name) },
            .{ "function_type", mlir.Attribute.type_(callee_fn_type) },
            // Best-effort: discourage inlining to preserve a visible boundary.
            .{ "llvm.noinline", mlir.Attribute.unit(mlir_ctx) },
        },
        .verify = false,
        .location = ctx.loc,
    });
    module.getBody().appendOperation(callee_op);

    // Emit a call in the original block.
    const call_operands = try arena.alloc(mlir.Value, inputs.len);
    for (inputs, 0..) |in_id, i| call_operands[i] = ctx.getValue(in_id) orelse return error.InvalidProgram;

    const call_op = mlir.Operation.make(mlir_ctx, "func.call", .{
        .results = &.{out_type},
        .operands = call_operands,
        .attributes = &.{
            .{ "callee", mlir.Attribute.symbol(mlir_ctx, callee_name_z) },
        },
        .verify = false,
        .location = ctx.loc,
    });
    ctx.block.appendOperation(call_op);
    ctx.setValue(out_id, call_op.result(0));
}

fn tensorToMlirType(ctx: mlir.Context, t: pr.Tensor, arena: std.mem.Allocator) !mlir.Type {
    const dims_i64 = try arena.alloc(i64, t.shape.dims.len);
    for (t.shape.dims, 0..) |d, i| dims_i64[i] = @intCast(d);
    return mlir.Type.tensor(dims_i64, ops.types.dtypeToMlirType(ctx, t.dtype));
}

test "lowering produces verified bytecode" {
    var program = try @import("../../frontend/frontend.zig").buildDemoProgram(std.testing.allocator);
    defer program.deinit();

    const func = program.functions[0];
    const bc = try lowerFunctionToMlir(std.testing.allocator, func, .mlir_bytecode);
    defer std.testing.allocator.free(bc);
    try std.testing.expect(bc.len > 0);
}

test "lowering supports reshape/broadcast/transpose" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.paramTensor(.f32, &.{ 2, 3 });
    const t = try b.transpose(x, &.{ 1, 0 });
    const r = try b.reshape(t, &.{6});
    const y = try b.broadcastInDim(r, &.{ 2, 6 }, &.{1});

    const func = try b.finish(&.{y});
    try program.addFunction(func);

    const bc = try lowerFunctionToMlir(std.testing.allocator, func, .mlir_bytecode);
    defer std.testing.allocator.free(bc);
    try std.testing.expect(bc.len > 0);
}

test "lowering supports custom_call boundary" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.paramTensor(.f32, &.{ 2, 3 });
    const y = try b.customCall("zigrad.test.missing_handler", &.{x}, x);

    const func = try b.finish(&.{y});
    try program.addFunction(func);

    const bc = try lowerFunctionToMlir(std.testing.allocator, func, .mlir_bytecode);
    defer std.testing.allocator.free(bc);
    try std.testing.expect(bc.len > 0);
}

test "lowering supports vjp matmul demo" {
    var program = try @import("../../frontend/frontend.zig").buildDemoProgram(std.testing.allocator);
    defer program.deinit();

    const fwd = program.functions[0];
    const vjp_func = try @import("../../pr/ad.zig").vjp(std.testing.allocator, &program, fwd, "vjp");

    const bc = try lowerFunctionToMlir(std.testing.allocator, vjp_func, .mlir_bytecode);
    defer std.testing.allocator.free(bc);
    try std.testing.expect(bc.len > 0);
}

test "lowering can outline an equation into a call boundary" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a = try b.paramTensor(.f32, &.{ 2, 3 });
    const c = try b.paramTensor(.f32, &.{ 3, 2 });
    const d = try b.emit(.dot, &.{ a, c }, &.{.{ .outline = true }});
    const func = try b.finish(&.{d});
    try program.addFunction(func);

    const text = try lowerFunctionToMlir(std.testing.allocator, func, .mlir_text);
    defer std.testing.allocator.free(text);

    try std.testing.expect(std.mem.indexOf(u8, text, "func.call") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "outlined_0") != null);
}
