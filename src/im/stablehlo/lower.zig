const std = @import("std");

const pr = @import("../../pr/pr.zig");
const mlir = @import("../../bridge/mlir/mlir.zig");
const stablehlo = @import("../../bridge/mlir/dialects/stablehlo.zig");

pub const LowerError = error{
    InvalidProgram,
    InvalidMlir,
    OutOfMemory,
};

fn dtypeToMlirType(ctx: mlir.Context, dt: pr.DType) mlir.Type {
    return switch (dt) {
        .f32 => mlir.Type.float(ctx, .f32),
        .f64 => mlir.Type.float(ctx, .f64),
        .i32 => mlir.Type.int(ctx, .i32),
        .i64 => mlir.Type.int(ctx, .i64),
        .u32 => mlir.Type.int(ctx, .i32),
        .u64 => mlir.Type.int(ctx, .i64),
    };
}

fn tensorToMlirType(ctx: mlir.Context, t: pr.Tensor, arena: std.mem.Allocator) !mlir.Type {
    const dims_i64 = try arena.alloc(i64, t.shape.dims.len);
    for (t.shape.dims, 0..) |d, i| dims_i64[i] = @intCast(d);
    return mlir.Type.tensor(dims_i64, dtypeToMlirType(ctx, t.dtype));
}

fn dtypeToDenseElementsType(dt: pr.DType) mlir.DenseElementsAttributeTypes {
    return switch (dt) {
        .f32 => .f32,
        .f64 => .f64,
        .i32 => .i32,
        .i64 => .i64,
        // StableHLO uses signless integers; keep the bits but lower u32/u64 as i32/i64.
        .u32 => .i32,
        .u64 => .i64,
    };
}

fn defaultLayoutAttr(ctx: mlir.Context, arena: std.mem.Allocator, rank: usize) !mlir.Attribute {
    const layout = try arena.alloc(usize, rank);
    for (0..rank) |i| {
        layout[i] = rank - i - 1;
    }
    const dims = [_]i64{@intCast(rank)};
    return mlir.Attribute.denseElements(ctx, dims[0..], .index, layout);
}

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

    for (func.eqns) |eqn| {
        switch (eqn) {
            .literal => |l| {
                const out_tensor = func.avals[@intCast(l.out)].asTensor() orelse return error.InvalidProgram;
                if (out_tensor.shape.rank() != 0) return error.InvalidProgram;

                const elem_type = dtypeToDenseElementsType(out_tensor.dtype);
                const raw_bytes = switch (l.value) {
                    inline else => |v| std.mem.asBytes(&v),
                };

                const op = stablehlo.constant(ctx, &.{}, elem_type, raw_bytes, loc);
                entry_block.appendOperation(op);
                value_map[@intCast(l.out)] = op.result(0);
            },
            .add => |b| {
                const lhs = value_map[@intCast(b.lhs)] orelse return error.InvalidProgram;
                const rhs = value_map[@intCast(b.rhs)] orelse return error.InvalidProgram;
                const out_tensor = func.avals[@intCast(b.out)].asTensor() orelse return error.InvalidProgram;
                const out_type = try tensorToMlirType(ctx, out_tensor, arena);
                const op = mlir.Operation.make(ctx, "stablehlo.add", .{
                    .operands = &.{ lhs, rhs },
                    .results = &.{out_type},
                    .verify = false,
                    .location = loc,
                });
                entry_block.appendOperation(op);
                value_map[@intCast(b.out)] = op.result(0);
            },
            .subtract => |b| {
                const lhs = value_map[@intCast(b.lhs)] orelse return error.InvalidProgram;
                const rhs = value_map[@intCast(b.rhs)] orelse return error.InvalidProgram;
                const out_tensor = func.avals[@intCast(b.out)].asTensor() orelse return error.InvalidProgram;
                const out_type = try tensorToMlirType(ctx, out_tensor, arena);
                const op = mlir.Operation.make(ctx, "stablehlo.subtract", .{
                    .operands = &.{ lhs, rhs },
                    .results = &.{out_type},
                    .verify = false,
                    .location = loc,
                });
                entry_block.appendOperation(op);
                value_map[@intCast(b.out)] = op.result(0);
            },
            .multiply => |b| {
                const lhs = value_map[@intCast(b.lhs)] orelse return error.InvalidProgram;
                const rhs = value_map[@intCast(b.rhs)] orelse return error.InvalidProgram;
                const out_tensor = func.avals[@intCast(b.out)].asTensor() orelse return error.InvalidProgram;
                const out_type = try tensorToMlirType(ctx, out_tensor, arena);
                const op = mlir.Operation.make(ctx, "stablehlo.multiply", .{
                    .operands = &.{ lhs, rhs },
                    .results = &.{out_type},
                    .verify = false,
                    .location = loc,
                });
                entry_block.appendOperation(op);
                value_map[@intCast(b.out)] = op.result(0);
            },
            .maximum => |b| {
                const lhs = value_map[@intCast(b.lhs)] orelse return error.InvalidProgram;
                const rhs = value_map[@intCast(b.rhs)] orelse return error.InvalidProgram;
                const out_tensor = func.avals[@intCast(b.out)].asTensor() orelse return error.InvalidProgram;
                const out_type = try tensorToMlirType(ctx, out_tensor, arena);
                const op = mlir.Operation.make(ctx, "stablehlo.maximum", .{
                    .operands = &.{ lhs, rhs },
                    .results = &.{out_type},
                    .verify = false,
                    .location = loc,
                });
                entry_block.appendOperation(op);
                value_map[@intCast(b.out)] = op.result(0);
            },
            .dot => |b| {
                const lhs = value_map[@intCast(b.lhs)] orelse return error.InvalidProgram;
                const rhs = value_map[@intCast(b.rhs)] orelse return error.InvalidProgram;
                const out_tensor = func.avals[@intCast(b.out)].asTensor() orelse return error.InvalidProgram;
                const out_type = try tensorToMlirType(ctx, out_tensor, arena);
                const op = stablehlo.dot_general(ctx, lhs, rhs, out_type, loc, .{
                    .lhs_batching_dimensions = &.{},
                    .rhs_batching_dimensions = &.{},
                    .lhs_contracting_dimensions = &.{1},
                    .rhs_contracting_dimensions = &.{0},
                    .precision = .fast,
                });
                entry_block.appendOperation(op);
                value_map[@intCast(b.out)] = op.result(0);
            },
            .reshape => |u| {
                const operand = value_map[@intCast(u.operand)] orelse return error.InvalidProgram;
                const out_tensor = func.avals[@intCast(u.out)].asTensor() orelse return error.InvalidProgram;
                const out_type = try tensorToMlirType(ctx, out_tensor, arena);
                const op = mlir.Operation.make(ctx, "stablehlo.reshape", .{
                    .operands = &.{operand},
                    .results = &.{out_type},
                    .verify = false,
                    .location = loc,
                });
                entry_block.appendOperation(op);
                value_map[@intCast(u.out)] = op.result(0);
            },
            .broadcast_in_dim => |b| {
                const operand = value_map[@intCast(b.operand)] orelse return error.InvalidProgram;
                const out_tensor = func.avals[@intCast(b.out)].asTensor() orelse return error.InvalidProgram;
                const out_type = try tensorToMlirType(ctx, out_tensor, arena);
                const op = mlir.Operation.make(ctx, "stablehlo.broadcast_in_dim", .{
                    .operands = &.{operand},
                    .results = &.{out_type},
                    .attributes = &.{
                        .{ "broadcast_dimensions", mlir.Attribute.dense(ctx, .i64, b.broadcast_dimensions) },
                    },
                    .verify = false,
                    .location = loc,
                });
                entry_block.appendOperation(op);
                value_map[@intCast(b.out)] = op.result(0);
            },
            .transpose => |t| {
                const operand = value_map[@intCast(t.operand)] orelse return error.InvalidProgram;
                const out_tensor = func.avals[@intCast(t.out)].asTensor() orelse return error.InvalidProgram;
                const out_type = try tensorToMlirType(ctx, out_tensor, arena);
                const op = mlir.Operation.make(ctx, "stablehlo.transpose", .{
                    .operands = &.{operand},
                    .results = &.{out_type},
                    .attributes = &.{
                        .{ "permutation", mlir.Attribute.dense(ctx, .i64, t.permutation) },
                    },
                    .verify = false,
                    .location = loc,
                });
                entry_block.appendOperation(op);
                value_map[@intCast(t.out)] = op.result(0);
            },
            .custom_call => |cc| {
                const out_tensor = func.avals[@intCast(cc.out)].asTensor() orelse return error.InvalidProgram;
                const out_type = try tensorToMlirType(ctx, out_tensor, arena);

                const operand_values = try arena.alloc(mlir.Value, cc.operands.len);
                const operand_layout_items = try arena.alloc(mlir.Attribute, cc.operands.len);
                for (cc.operands, 0..) |operand_id, i| {
                    operand_values[i] = value_map[@intCast(operand_id)] orelse return error.InvalidProgram;
                    const operand_tensor = func.avals[@intCast(operand_id)].asTensor() orelse return error.InvalidProgram;
                    operand_layout_items[i] = try defaultLayoutAttr(ctx, arena, operand_tensor.shape.rank());
                }

                const result_layout_item = try defaultLayoutAttr(ctx, arena, out_tensor.shape.rank());
                const result_layout_items = [_]mlir.Attribute{result_layout_item};

                const attrs = [_]mlir.AttrTuple{
                    .{ "api_version", mlir.Attribute.int(ctx, .i32, 4) }, // typed_ffi
                    .{ "call_target_name", mlir.Attribute.string(ctx, cc.target) },
                    .{ "has_side_effect", mlir.Attribute.boolean(ctx, cc.has_side_effect) },
                    .{ "backend_config", mlir.Attribute.dict(ctx, &.{}) },
                    .{ "output_operand_aliases", mlir.Attribute.array(ctx, &.{}) },
                    .{ "operand_layouts", mlir.Attribute.array(ctx, operand_layout_items) },
                    .{ "result_layouts", mlir.Attribute.array(ctx, &result_layout_items) },
                };

                const op = mlir.Operation.make(ctx, "stablehlo.custom_call", .{
                    .operands = operand_values,
                    .results = &.{out_type},
                    .attributes = &attrs,
                    .verify = false,
                    .location = loc,
                });
                entry_block.appendOperation(op);
                value_map[@intCast(cc.out)] = op.result(0);
            },
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

    var bytecode_buffer: [1024 * 1024]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&bytecode_buffer);
    try module.op().writeBytecode(&writer);
    return allocator.dupe(u8, writer.buffered());
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
