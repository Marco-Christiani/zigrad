const std = @import("std");

const pr = @import("pr.zig");
const ops = @import("ops/ops.zig");

pub const VjpError = ops.types.AdError;

fn zeroLike(bld: *pr.FunctionBuilder, tensor: pr.Tensor) pr.BuildError!pr.VarId {
    const z = try bld.literalScalar(ops.types.scalarLiteral(tensor.dtype, 0.0));
    if (tensor.shape.rank() == 0) return z;
    return try bld.broadcastInDim(z, tensor.shape.dims, &.{});
}

pub fn vjp(allocator: std.mem.Allocator, program: *pr.Program, func: pr.Function, name: []const u8) VjpError!pr.Function {
    try pr.validateFunction(func);

    var primal_map = try allocator.alloc(?pr.VarId, func.avals.len);
    defer allocator.free(primal_map);
    @memset(primal_map, null);

    var cot_map = try allocator.alloc(?pr.VarId, func.avals.len);
    defer allocator.free(cot_map);
    @memset(cot_map, null);

    var b = try pr.FunctionBuilder.init(program, name);
    defer b.deinit();

    // Create parameters for primals
    for (func.params) |param_id| {
        const tensor = func.avals[@intCast(param_id)].asTensor() orelse return error.UnsupportedEqn;
        if (tensor.dtype != .f32 and tensor.dtype != .f64) return error.UnsupportedDType;

        const new_param = try b.paramTensor(tensor.dtype, tensor.shape.dims);
        primal_map[@intCast(param_id)] = new_param;
    }

    // Create parameters for cotangents of outputs
    for (func.returns) |ret_id| {
        const tensor = func.avals[@intCast(ret_id)].asTensor() orelse return error.UnsupportedEqn;
        if (tensor.dtype != .f32 and tensor.dtype != .f64) return error.UnsupportedDType;

        const new_cot = try b.paramTensor(tensor.dtype, tensor.shape.dims);
        cot_map[@intCast(ret_id)] = new_cot;
    }

    // Forward pass: compute primals
    const ad_ctx = ops.types.AdContext{
        .builder = &b,
        .primal_map = primal_map,
        .cot_map = cot_map,
        .func = func,
        .allocator = allocator,
    };

    for (func.eqns) |eqn| {
        try ops.vjpForward(ad_ctx, eqn);
    }

    // Backward pass: propagate cotangents
    var eqn_index: usize = func.eqns.len;
    while (eqn_index > 0) {
        eqn_index -= 1;
        const eqn = func.eqns[eqn_index];
        try ops.vjpBackward(ad_ctx, eqn);
    }

    // Collect gradients for input parameters
    const returns = try allocator.alloc(pr.VarId, func.params.len);
    defer allocator.free(returns);
    for (func.params, 0..) |param_id, i| {
        if (cot_map[@intCast(param_id)]) |cot| {
            returns[i] = cot;
        } else {
            const tensor = func.avals[@intCast(param_id)].asTensor() orelse return error.UnsupportedEqn;
            returns[i] = try zeroLike(&b, tensor);
        }
    }

    return b.finish(returns);
}

test "vjp produces gradients matching input shapes" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a_id = try b.paramTensor(.f32, &.{ 2, 3 });
    const b_id = try b.paramTensor(.f32, &.{ 3, 2 });
    const c_id = try b.paramTensor(.f32, &.{ 2, 2 });

    const dot_id = try b.dot(a_id, b_id);
    const add_id = try b.add(dot_id, c_id);
    const out_id = try b.multiply(add_id, c_id);

    const func = try b.finish(&.{out_id});
    try program.addFunction(func);

    const vjp_func = try vjp(std.testing.allocator, &program, func, "vjp");
    try pr.validateFunction(vjp_func);

    try std.testing.expectEqual(@as(usize, func.params.len + func.returns.len), vjp_func.params.len);
    try std.testing.expectEqual(@as(usize, func.params.len), vjp_func.returns.len);

    for (func.params, 0..) |param_id, i| {
        const p_t = func.avals[@intCast(param_id)].asTensor().?;
        const g_id = vjp_func.returns[i];
        const g_t = vjp_func.avals[@intCast(g_id)].asTensor().?;
        try std.testing.expectEqual(p_t.dtype, g_t.dtype);
        try std.testing.expect(std.mem.eql(usize, p_t.shape.dims, g_t.shape.dims));
    }
}
