//! Automatic differentiation transforms on PR functions.
//!
//! VJP and JVP transform `pr.Function` values into derivative functions.
//!
//! Higher-level traced transforms live in `transforms.zig`.
const std = @import("std");

const pr = @import("pr.zig");
const ops = @import("ops/ops.zig");

const log = std.log.scoped(.@"zg/ad");

pub const VjpError = ops.types.AdError;
pub const JvpError = ops.types.AdError;

/// Options for `vjp` and `vjp_with_value`.
pub const VjpOpts = struct {
    /// Select primal-input cotangents returned by the transformed function.
    ///
    /// `null` selects every input. The output order matches this slice, including
    ///  duplicate indices.
    wrt: ?[]const usize = null,
};

/// Direction used to propagate derivatives through PR operations.
const Mode = enum {
    /// Propagate cotangents from outputs to inputs.
    vjp,
    /// Propagate tangents from inputs to outputs.
    jvp,
};

/// Controls whether a derivative function also returns primal outputs.
const PrimalOutputs = enum { skip, emit };

/// Reports whether PR defines dual values for a dtype.
fn is_differentiable_dtype(dtype: pr.DType) bool {
    return switch (dtype) {
        .f32, .f64, .bf16, .f16 => true,
        else => false,
    };
}

/// Applies the AD traversal selected by `mode`.
///
/// VJP seeds output cotangents, traverses operations in reverse, and harvests
///  input cotangents. JVP seeds input tangents, traverses operations forward,
///  and harvests output tangents.
///
/// `wrt` restricts the harvested values. Traversal still covers every operation
///  so a later dead-code elimination pass can remove work for omitted results.
///
/// Missing rules on active differentiable paths return `error.UnsupportedEqn`.
/// Missing duals for non-differentiable dtypes become zero tensors. A missing
///  dual for a differentiable dtype returns `error.MissingDual`.
/// TODO(ad): Make the built-in differentiable dtype set an explicit AD policy.
fn ad_impl(
    comptime mode: Mode,
    allocator: std.mem.Allocator,
    program: *pr.Program,
    func: pr.Function,
    name: []const u8,
    primals: PrimalOutputs,
    wrt: ?[]const usize,
) ops.types.AdError!pr.Function {
    var primal_map = try allocator.alloc(?*pr.Var, func.var_count);
    defer allocator.free(primal_map);
    @memset(primal_map, null);

    var dual_map = try allocator.alloc(?*pr.Var, func.var_count);
    defer allocator.free(dual_map);
    @memset(dual_map, null);

    var b = try pr.FunctionBuilder.init(program, name);
    defer b.deinit();

    for (func.params) |param_var| {
        const tensor = param_var.aval.as_tensor();
        const new_param = try b.param_tensor(tensor.dtype, tensor.shape.dims);
        primal_map[param_var.id] = new_param;
    }

    // VJP seeds output cotangents. JVP seeds input tangents.
    const seed_vars = switch (mode) {
        .vjp => func.returns,
        .jvp => func.params,
    };
    for (seed_vars) |v| {
        const tensor = v.aval.as_tensor();
        if (mode == .vjp) {
            if (tensor.dtype != .f32 and tensor.dtype != .f64 and tensor.dtype != .bf16)
                return error.UnsupportedDType;
        }
        const new_seed = try b.param_tensor(tensor.dtype, tensor.shape.dims);
        dual_map[v.id] = new_seed;
    }

    const ad_ctx = ops.types.AdContext{
        .builder = &b,
        .primal_map = primal_map,
        .cot_map = if (mode == .vjp) dual_map else null,
        .tangent_map = if (mode == .jvp) dual_map else null,
        .allocator = allocator,
    };

    // TODO(ad): Emit a call to the primal function instead of duplicating its operations.
    //
    // Backends can inline calls during lowering. PR can also expose explicit
    //  inlining as a transform when needed.
    switch (mode) {
        .vjp => {
            for (func.ops) |op| try ops.emit_primal(ad_ctx, op);
            var i: usize = func.ops.len;
            while (i > 0) {
                i -= 1;
                try ops.vjp(ad_ctx, func.ops[i]);
            }
        },
        .jvp => {
            for (func.ops) |op| {
                try ops.emit_primal(ad_ctx, op);
                try ops.jvp(ad_ctx, op);
            }
        },
    }

    // VJP harvests input cotangents. JVP harvests output tangents.
    const all_harvest_vars = switch (mode) {
        .vjp => func.params,
        .jvp => func.returns,
    };
    if (wrt) |indices| {
        for (indices) |idx| {
            if (idx >= all_harvest_vars.len) return error.WrtIndexOutOfRange;
        }
    }
    const num_harvest: usize = if (wrt) |indices| indices.len else all_harvest_vars.len;

    const num_primal_outputs: usize = switch (primals) {
        .emit => func.returns.len,
        .skip => 0,
    };
    const returns = try allocator.alloc(*pr.Var, num_primal_outputs + num_harvest);
    defer allocator.free(returns);

    var out_idx: usize = 0;
    if (primals == .emit) {
        for (func.returns) |ret_var| {
            returns[out_idx] = primal_map[ret_var.id] orelse return error.UnsupportedEqn;
            out_idx += 1;
        }
    }
    for (0..num_harvest) |i| {
        const v = if (wrt) |indices| all_harvest_vars[indices[i]] else all_harvest_vars[i];
        if (dual_map[v.id]) |dual| {
            returns[out_idx] = dual;
        } else {
            const t = v.aval.as_tensor();
            if (is_differentiable_dtype(t.dtype)) {
                log.err(
                    "no dual for float harvest var id={} dtype={s} shape={any} " ++
                        "likely a missing {s} in the primal chain",
                    .{
                        v.id,
                        @tagName(t.dtype),
                        t.shape.dims,
                        if (mode == .vjp) "vjp" else "jvp",
                    },
                );
                return error.MissingDual;
            }
            // Integer inputs use zero duals to preserve result arity.
            returns[out_idx] = try b.scalar_broadcast(t.dtype, t.shape.dims, 0.0);
        }
        out_idx += 1;
    }

    return try b.finish(returns);
}

/// Reverse-mode AD (pullback) transforms \(f: M \to N\) into:
///
/// $$
/// \operatorname{vjp}_f: (T_x M, T^*_{f(x)} N) \to T^*_x M
/// $$
///
/// For a cotangent seed \(v\), it computes the transpose-Jacobian product
/// \(J^\mathsf{T}(x) \cdot v\), the pullback
/// \(f^*: T^*_{f(x)} N \to T^*_x M\) evaluated at \(x\).
///
/// The returned `pr.Function` takes \(N\) primal inputs followed by \(M\) output
///  cotangent seeds, and returns \(N\) input cotangent vectors. In the
///  Euclidean or Cartesian case (\(G = I\)), these equal gradients. In general,
///  they are covectors and must be raised with \(G^{-1}\) to obtain gradient
///  tangent vectors.
pub fn vjp(
    allocator: std.mem.Allocator,
    program: *pr.Program,
    func: pr.Function,
    name: []const u8,
    /// See `VjpOpts` for how `opts.wrt` restricts which parameter cotangents
    ///  appear in the returned function's outputs. Default `.{}` harvests all
    ///  parameter cotangents.
    opts: VjpOpts,
) VjpError!pr.Function {
    return try ad_impl(.vjp, allocator, program, func, name, .skip, opts.wrt);
}

/// Apply VJP and emit primal outputs before input cotangents.
///
/// Takes \(N\) primal inputs and \(M\) cotangent seeds, then returns \(M\)
/// primal outputs and \(K\) input cotangents. Here \(K\) is
/// `opts.wrt.?.len` when provided, otherwise \(N\).
///
/// See `vjp`
pub fn vjp_with_value(
    allocator: std.mem.Allocator,
    program: *pr.Program,
    func: pr.Function,
    name: []const u8,
    opts: VjpOpts,
) VjpError!pr.Function {
    return try ad_impl(.vjp, allocator, program, func, name, .emit, opts.wrt);
}

/// Forward-mode AD (pushforward / differential) transforms \(f: M \to N\) into:
///
/// $$
/// \operatorname{jvp}_f: (T_x M, T_x M) \to T_{f(x)} N
/// $$
///
/// For a tangent seed \(v\), it computes the Jacobian-vector product
/// \(J(x) \cdot v\), the differential
/// \(\mathrm{d}f_x: T_x M \to T_{f(x)} N\) applied to \(v\).
///
/// The returned function takes \(N\) primal inputs followed by \(N\) input
/// tangent vectors of the same shapes, and returns \(M\) output tangent vectors
/// matching the original function's output shapes.
pub fn jvp(allocator: std.mem.Allocator, program: *pr.Program, func: pr.Function, name: []const u8) JvpError!pr.Function {
    return try ad_impl(.jvp, allocator, program, func, name, .skip, null);
}

/// Apply JVP and emit primal outputs before output tangents.
///
/// Takes \(N\) primal inputs and \(N\) tangents, then returns \(M\) primal
/// outputs and \(M\) output tangents.
///
/// See `jvp`
pub fn jvp_with_value(allocator: std.mem.Allocator, program: *pr.Program, func: pr.Function, name: []const u8) JvpError!pr.Function {
    return try ad_impl(.jvp, allocator, program, func, name, .emit, null);
}

/// Emit a ones-like cotangent for VJP seeding.
pub fn emit_cotangent(builder: *pr.FunctionBuilder, tensor: pr.Tensor) pr.BuildError!*pr.Var {
    const s = try builder.scalar(tensor.dtype, 1.0);
    if (tensor.shape.rank() == 0) return s;
    return try builder.broadcast_in_dim(s, tensor.shape.dims, &.{});
}

test "vjp produces gradients matching input shapes" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a_id = try b.param_tensor(.f32, &.{ 2, 3 });
    const b_id = try b.param_tensor(.f32, &.{ 3, 2 });
    const c_id = try b.param_tensor(.f32, &.{ 2, 2 });

    const mm_id = try b.mm(a_id, b_id);
    const add_id = try b.add(mm_id, c_id);
    const out_id = try b.multiply(add_id, c_id);

    const func = try b.finish(&.{out_id});
    try program.add_function(func);

    const vjp_func = try vjp(std.testing.allocator, &program, func, "vjp", .{});
    try pr.validate_ops_in_func(vjp_func);

    try std.testing.expectEqual(@as(usize, func.params.len + func.returns.len), vjp_func.params.len);
    try std.testing.expectEqual(@as(usize, func.params.len), vjp_func.returns.len);

    for (func.params, 0..) |param_var, i| {
        const p_t = param_var.as_tensor();
        const g_var = vjp_func.returns[i];
        const g_t = g_var.as_tensor();
        try std.testing.expectEqual(p_t.dtype, g_t.dtype);
        try std.testing.expect(std.mem.eql(i64, p_t.shape.dims, g_t.shape.dims));
    }
}

test "vjp_with_value returns primals plus gradients" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 2 });
    const y = try b.multiply(x, x);
    const func = try b.finish(&.{y});
    try program.add_function(func);

    const vjp_func = try vjp_with_value(std.testing.allocator, &program, func, "vjp_with_value", .{});
    try pr.validate_ops_in_func(vjp_func);

    try std.testing.expectEqual(@as(usize, func.returns.len + func.params.len), vjp_func.returns.len);
}

test "dot_general vjp supports 2 batch dims" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const lhs = try b.param_tensor(.f32, &.{ 2, 3, 4, 5 }); // [B,H,M,K]
    const rhs = try b.param_tensor(.f32, &.{ 2, 3, 5, 6 }); // [B,H,K,N]
    const out = try b.dot_general(lhs, rhs, .{
        .lhs_batch_dims = &.{ 0, 1 },
        .rhs_batch_dims = &.{ 0, 1 },
        .lhs_contracting_dims = &.{3},
        .rhs_contracting_dims = &.{2},
    });

    const func = try b.finish(&.{out});
    try program.add_function(func);

    const vjp_func = try vjp(std.testing.allocator, &program, func, "vjp", .{});
    try pr.validate_ops_in_func(vjp_func);

    try std.testing.expectEqual(@as(usize, func.params.len + func.returns.len), vjp_func.params.len);
    try std.testing.expectEqual(@as(usize, func.params.len), vjp_func.returns.len);

    for (func.params, 0..) |param_var, i| {
        const p_t = param_var.as_tensor();
        const g_var = vjp_func.returns[i];
        const g_t = g_var.as_tensor();
        try std.testing.expectEqual(p_t.dtype, g_t.dtype);
        try std.testing.expect(std.mem.eql(i64, p_t.shape.dims, g_t.shape.dims));
    }
}

test "dot_general vjp supports non-prefix batch dims" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    // lhs/rhs: [B,S,H,D]
    const lhs = try b.param_tensor(.f32, &.{ 2, 4, 3, 5 });
    const rhs = try b.param_tensor(.f32, &.{ 2, 4, 3, 5 });
    const out = try b.dot_general(lhs, rhs, .{
        .lhs_batch_dims = &.{ 0, 2 },
        .rhs_batch_dims = &.{ 0, 2 },
        .lhs_contracting_dims = &.{3},
        .rhs_contracting_dims = &.{3},
    });

    const func = try b.finish(&.{out});
    try program.add_function(func);

    const vjp_func = try vjp(std.testing.allocator, &program, func, "vjp", .{});
    try pr.validate_ops_in_func(vjp_func);
}

test "dot_general vjp supports differing batch dim positions" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    // lhs: [B,H,S,S], rhs: [B,S,H,D] -> out: [B,H,S,D]
    const lhs = try b.param_tensor(.f32, &.{ 2, 3, 4, 4 });
    const rhs = try b.param_tensor(.f32, &.{ 2, 4, 3, 5 });
    const out = try b.dot_general(lhs, rhs, .{
        .lhs_batch_dims = &.{ 0, 1 },
        .rhs_batch_dims = &.{ 0, 2 },
        .lhs_contracting_dims = &.{3},
        .rhs_contracting_dims = &.{1},
    });

    const func = try b.finish(&.{out});
    try program.add_function(func);

    const vjp_func = try vjp(std.testing.allocator, &program, func, "vjp", .{});
    try pr.validate_ops_in_func(vjp_func);
}

test "dot_general vjp supports multi-contract dims" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    // lhs/rhs: [B,S,H,D], contract over H and D -> out [B,S]
    const lhs = try b.param_tensor(.f32, &.{ 2, 3, 4, 5 });
    const rhs = try b.param_tensor(.f32, &.{ 2, 3, 4, 5 });
    const out = try b.dot_general(lhs, rhs, .{
        .lhs_batch_dims = &.{ 0, 1 },
        .rhs_batch_dims = &.{ 0, 1 },
        .lhs_contracting_dims = &.{ 2, 3 },
        .rhs_contracting_dims = &.{ 2, 3 },
    });

    const func = try b.finish(&.{out});
    try program.add_function(func);

    const vjp_func = try vjp(std.testing.allocator, &program, func, "vjp", .{});
    try pr.validate_ops_in_func(vjp_func);
}

test "jvp produces tangent outputs matching function output shapes" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a_id = try b.param_tensor(.f32, &.{ 2, 3 });
    const b_id = try b.param_tensor(.f32, &.{ 3, 2 });
    const c_id = try b.param_tensor(.f32, &.{ 2, 2 });

    const mm_id = try b.mm(a_id, b_id);
    const add_id = try b.add(mm_id, c_id);
    const out_id = try b.multiply(add_id, c_id);

    const func = try b.finish(&.{out_id});
    try program.add_function(func);

    const jvp_func = try jvp(std.testing.allocator, &program, func, "jvp");
    try pr.validate_ops_in_func(jvp_func);

    try std.testing.expectEqual(func.params.len * 2, jvp_func.params.len);
    try std.testing.expectEqual(func.returns.len, jvp_func.returns.len);

    for (func.returns, 0..) |ret_var, i| {
        const orig_t = ret_var.as_tensor();
        const jvp_var = jvp_func.returns[i];
        const jvp_t = jvp_var.as_tensor();
        try std.testing.expectEqual(orig_t.dtype, jvp_t.dtype);
        try std.testing.expect(std.mem.eql(i64, orig_t.shape.dims, jvp_t.shape.dims));
    }
}

test "jvp_with_value returns primals plus tangents" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 2 });
    const y = try b.multiply(x, x);
    const func = try b.finish(&.{y});
    try program.add_function(func);

    const jvp_func = try jvp_with_value(std.testing.allocator, &program, func, "jvp_with_value");
    try pr.validate_ops_in_func(jvp_func);

    try std.testing.expectEqual(func.returns.len * 2, jvp_func.returns.len);
    try std.testing.expectEqual(func.params.len * 2, jvp_func.params.len);
}

test "dot_general jvp with batch dims" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const lhs = try b.param_tensor(.f32, &.{ 2, 3, 4, 5 }); // [B,H,M,K]
    const rhs = try b.param_tensor(.f32, &.{ 2, 3, 5, 6 }); // [B,H,K,N]
    const out = try b.dot_general(lhs, rhs, .{
        .lhs_batch_dims = &.{ 0, 1 },
        .rhs_batch_dims = &.{ 0, 1 },
        .lhs_contracting_dims = &.{3},
        .rhs_contracting_dims = &.{2},
    });

    const func = try b.finish(&.{out});
    try program.add_function(func);

    const jvp_func = try jvp(std.testing.allocator, &program, func, "jvp");
    try pr.validate_ops_in_func(jvp_func);

    const orig_t = func.returns[0].as_tensor();
    const jvp_t = jvp_func.returns[0].as_tensor();
    try std.testing.expectEqual(orig_t.dtype, jvp_t.dtype);
    try std.testing.expect(std.mem.eql(i64, orig_t.shape.dims, jvp_t.shape.dims));
}
