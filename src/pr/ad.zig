const std = @import("std");

const pr = @import("pr.zig");

pub const VjpError = pr.BuildError || error{
    UnsupportedEqn,
    UnsupportedDType,
};

fn scalarLiteral(value_dtype: pr.DType, value: f64) pr.Literal {
    return switch (value_dtype) {
        .f32 => .{ .f32 = @floatCast(value) },
        .f64 => .{ .f64 = value },
        .i32 => .{ .i32 = @intFromFloat(value) },
        .i64 => .{ .i64 = @intFromFloat(value) },
        .u32 => .{ .u32 = @intFromFloat(value) },
        .u64 => .{ .u64 = @intFromFloat(value) },
    };
}

fn addCot(
    bld: *pr.FunctionBuilder,
    cot_map_mut: []?pr.VarId,
    var_id: pr.VarId,
    new_cot: pr.VarId,
) pr.BuildError!void {
    const idx: usize = @intCast(var_id);
    if (cot_map_mut[idx]) |existing| {
        cot_map_mut[idx] = try bld.add(existing, new_cot);
    } else {
        cot_map_mut[idx] = new_cot;
    }
}

fn negateLike(
    bld: *pr.FunctionBuilder,
    tensor: pr.Tensor,
    value: pr.VarId,
) pr.BuildError!pr.VarId {
    const minus_one = try bld.literalScalar(scalarLiteral(tensor.dtype, -1.0));
    const minus_one_full = if (tensor.shape.rank() == 0)
        minus_one
    else
        try bld.broadcastInDim(minus_one, tensor.shape.dims, &.{});
    return try bld.multiply(value, minus_one_full);
}

fn zeroLike(bld: *pr.FunctionBuilder, tensor: pr.Tensor) pr.BuildError!pr.VarId {
    const z = try bld.literalScalar(scalarLiteral(tensor.dtype, 0.0));
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

    for (func.params) |param_id| {
        const tensor = func.avals[@intCast(param_id)].asTensor() orelse return error.UnsupportedEqn;
        if (tensor.dtype != .f32 and tensor.dtype != .f64) return error.UnsupportedDType;

        const new_param = try b.paramTensor(tensor.dtype, tensor.shape.dims);
        primal_map[@intCast(param_id)] = new_param;
    }

    for (func.returns) |ret_id| {
        const tensor = func.avals[@intCast(ret_id)].asTensor() orelse return error.UnsupportedEqn;
        if (tensor.dtype != .f32 and tensor.dtype != .f64) return error.UnsupportedDType;

        const new_cot = try b.paramTensor(tensor.dtype, tensor.shape.dims);
        cot_map[@intCast(ret_id)] = new_cot;
    }

    for (func.eqns) |eqn| {
        switch (eqn) {
            .literal => |l| {
                const out = try b.literalScalar(l.value);
                primal_map[@intCast(l.out)] = out;
            },
            .add => |op| {
                const lhs = primal_map[@intCast(op.lhs)] orelse return error.UnsupportedEqn;
                const rhs = primal_map[@intCast(op.rhs)] orelse return error.UnsupportedEqn;
                const out = try b.add(lhs, rhs);
                primal_map[@intCast(op.out)] = out;
            },
            .subtract => |op| {
                const lhs = primal_map[@intCast(op.lhs)] orelse return error.UnsupportedEqn;
                const rhs = primal_map[@intCast(op.rhs)] orelse return error.UnsupportedEqn;
                const out = try b.subtract(lhs, rhs);
                primal_map[@intCast(op.out)] = out;
            },
            .multiply => |op| {
                const lhs = primal_map[@intCast(op.lhs)] orelse return error.UnsupportedEqn;
                const rhs = primal_map[@intCast(op.rhs)] orelse return error.UnsupportedEqn;
                const out = try b.multiply(lhs, rhs);
                primal_map[@intCast(op.out)] = out;
            },
            .dot => |op| {
                const lhs = primal_map[@intCast(op.lhs)] orelse return error.UnsupportedEqn;
                const rhs = primal_map[@intCast(op.rhs)] orelse return error.UnsupportedEqn;
                const out = try b.dot(lhs, rhs);
                primal_map[@intCast(op.out)] = out;
            },
            .reshape => |op| {
                const operand = primal_map[@intCast(op.operand)] orelse return error.UnsupportedEqn;
                const out_tensor = func.avals[@intCast(op.out)].asTensor() orelse return error.UnsupportedEqn;
                const out = try b.reshape(operand, out_tensor.shape.dims);
                primal_map[@intCast(op.out)] = out;
            },
            .transpose => |op| {
                const operand = primal_map[@intCast(op.operand)] orelse return error.UnsupportedEqn;
                const out = try b.transpose(operand, op.permutation);
                primal_map[@intCast(op.out)] = out;
            },
            .broadcast_in_dim,
            .maximum,
            .custom_call,
            => return error.UnsupportedEqn,
        }
    }

    var eqn_index: usize = func.eqns.len;
    while (eqn_index > 0) {
        eqn_index -= 1;
        const eqn = func.eqns[eqn_index];
        switch (eqn) {
            .literal => {},
            .add => |op| {
                const out_cot = cot_map[@intCast(op.out)] orelse continue;
                try addCot(&b, cot_map, op.lhs, out_cot);
                try addCot(&b, cot_map, op.rhs, out_cot);
            },
            .subtract => |op| {
                const out_cot = cot_map[@intCast(op.out)] orelse continue;
                const rhs_tensor = func.avals[@intCast(op.rhs)].asTensor() orelse return error.UnsupportedEqn;
                const neg = try negateLike(&b, rhs_tensor, out_cot);
                try addCot(&b, cot_map, op.lhs, out_cot);
                try addCot(&b, cot_map, op.rhs, neg);
            },
            .multiply => |op| {
                const out_cot = cot_map[@intCast(op.out)] orelse continue;
                const lhs_primal = primal_map[@intCast(op.lhs)] orelse return error.UnsupportedEqn;
                const rhs_primal = primal_map[@intCast(op.rhs)] orelse return error.UnsupportedEqn;

                const lhs_contrib = try b.multiply(out_cot, rhs_primal);
                const rhs_contrib = try b.multiply(out_cot, lhs_primal);

                try addCot(&b, cot_map, op.lhs, lhs_contrib);
                try addCot(&b, cot_map, op.rhs, rhs_contrib);
            },
            .dot => |op| {
                const out_cot = cot_map[@intCast(op.out)] orelse continue;
                const lhs_primal = primal_map[@intCast(op.lhs)] orelse return error.UnsupportedEqn;
                const rhs_primal = primal_map[@intCast(op.rhs)] orelse return error.UnsupportedEqn;

                const rhs_t = try b.transpose(rhs_primal, &.{ 1, 0 });
                const lhs_t = try b.transpose(lhs_primal, &.{ 1, 0 });

                const lhs_contrib = try b.dot(out_cot, rhs_t);
                const rhs_contrib = try b.dot(lhs_t, out_cot);

                try addCot(&b, cot_map, op.lhs, lhs_contrib);
                try addCot(&b, cot_map, op.rhs, rhs_contrib);
            },
            .reshape => |op| {
                const out_cot = cot_map[@intCast(op.out)] orelse continue;
                const operand_tensor = func.avals[@intCast(op.operand)].asTensor() orelse return error.UnsupportedEqn;
                const contrib = try b.reshape(out_cot, operand_tensor.shape.dims);
                try addCot(&b, cot_map, op.operand, contrib);
            },
            .transpose => |op| {
                const out_cot = cot_map[@intCast(op.out)] orelse continue;

                const perm = op.permutation;
                const inv = try allocator.alloc(i64, perm.len);
                defer allocator.free(inv);
                for (perm, 0..) |p, i| inv[@intCast(p)] = @intCast(i);

                const contrib = try b.transpose(out_cot, inv);
                try addCot(&b, cot_map, op.operand, contrib);
            },
            .broadcast_in_dim,
            .maximum,
            .custom_call,
            => return error.UnsupportedEqn,
        }
    }

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
