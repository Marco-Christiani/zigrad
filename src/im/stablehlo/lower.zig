const std = @import("std");

const pr = @import("../../pr/pr.zig");
const ops = @import("../../pr/ops/ops.zig");
const mlir = @import("../../ffi/mlir/mlir.zig");

pub const LowerError = ops.types.LowerError;

pub fn lowerFunctionToMlirBytecode(allocator: std.mem.Allocator, func: pr.Function) ![]u8 {
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

    for (func.eqns) |eqn| {
        try ops.lower(lower_ctx, eqn);
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

    var bytecode_buffer: [1024 * 1024]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&bytecode_buffer);
    try module.op().writeBytecode(&writer);
    return allocator.dupe(u8, writer.buffered());
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
    const bc = try lowerFunctionToMlirBytecode(std.testing.allocator, func);
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

    const bc = try lowerFunctionToMlirBytecode(std.testing.allocator, func);
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

    const bc = try lowerFunctionToMlirBytecode(std.testing.allocator, func);
    defer std.testing.allocator.free(bc);
    try std.testing.expect(bc.len > 0);
}

test "lowering supports vjp matmul demo" {
    var program = try @import("../../frontend/frontend.zig").buildDemoProgram(std.testing.allocator);
    defer program.deinit();

    const fwd = program.functions[0];
    const vjp_func = try @import("../../pr/ad.zig").vjp(std.testing.allocator, &program, fwd, "vjp");

    const bc = try lowerFunctionToMlirBytecode(std.testing.allocator, vjp_func);
    defer std.testing.allocator.free(bc);
    try std.testing.expect(bc.len > 0);
}
