const std = @import("std");

const pr = @import("pr.zig");
const ops = @import("ops/ops.zig");

pub const VjpError = ops.types.AdError;
pub const JvpError = ops.types.AdError;

/// AD mode
const Mode = enum {
    /// Reverse-mode AD. Propagates cotangents from outputs to inputs
    vjp,
    /// Forward-mode AD. Propagates tangents forward from inputs to outputs.
    jvp,
};

/// Unified AD transform, specialized at comptime on `mode`.
///
/// Both modes share the same skeleton:
///  1. Validate the source function and create primal parameters.
///  2. Allocate a dual map and create seed parameters for it.
///      VJP seeds cotangent vectors at outputs (elements of T*_{f(x)}N).
///      JVP seeds tangent vectors at inputs (elements of T_xM).
///  3. Run per-op AD handlers over the equation list.
///  4. Collect results: optionally primal outputs, then dual values.
///      VJP harvests cotangent vectors at inputs (elements of T*_xM).
///      JVP harvests tangent vectors at outputs (elements of T_{f(x)}N).
///
/// NOTE: In the Euclidean case (G = I), input cotangents from VJP coincide
///       numerically with gradients via the trivial musical isomorphism. For
///       non-Cartesian metrics, converting to gradients requires applying G^{-1}.
///
/// NOTE: Missing dual entries (ops w/o AD support) fall back to zero-filled tensors.
/// TODO: missing AD support might require a better policy, tbd.
fn ad_impl(
    comptime mode: Mode,
    allocator: std.mem.Allocator,
    program: *pr.Program,
    func: pr.Function,
    name: []const u8,
    include_value: bool,
) ops.types.AdError!pr.Function {
    try pr.validate_function(func);

    var primal_map = try allocator.alloc(?pr.VarId, func.avals.len);
    defer allocator.free(primal_map);
    @memset(primal_map, null);

    var dual_map = try allocator.alloc(?pr.VarId, func.avals.len);
    defer allocator.free(dual_map);
    @memset(dual_map, null);

    var b = try pr.FunctionBuilder.init(program, name);
    defer b.deinit();

    // Create parameters for primals.
    for (func.params) |param_id| {
        const tensor = func.avals[@intCast(param_id)].as_tensor() orelse return error.UnsupportedEqn;
        const new_param = try b.param_tensor(tensor.dtype, tensor.shape.dims);
        primal_map[@intCast(param_id)] = new_param;
    }

    // Seed the dual map - stores the "other half" of the primal/dual pair.
    // VJP: cotangent vectors (elements of T*_{f(x)}N) seeded at each output.
    // JVP: tangent vectors (elements of T_xM) seeded at each input.
    // Duality is symmetric: cotangents are dual to tangents and vice versa,
    //  this is not in reference to dual numbers as in some forward-mode impls.
    const seed_ids = switch (mode) {
        .vjp => func.returns,
        .jvp => func.params,
    };
    for (seed_ids) |id| {
        const tensor = func.avals[@intCast(id)].as_tensor() orelse return error.UnsupportedEqn;
        if (mode == .vjp) {
            if (tensor.dtype != .f32 and tensor.dtype != .f64 and tensor.dtype != .bf16)
                return error.UnsupportedDType;
        }
        const new_seed = try b.param_tensor(tensor.dtype, tensor.shape.dims);
        dual_map[@intCast(id)] = new_seed;
    }

    const ad_ctx = ops.types.AdContext{
        .builder = &b,
        .primal_map = primal_map,
        .cot_map = if (mode == .vjp) dual_map else null,
        .tangent_map = if (mode == .jvp) dual_map else null,
        .func = func,
        .allocator = allocator,
    };

    switch (mode) {
        .vjp => {
            for (func.eqns) |eqn| try ops.vjp_forward(ad_ctx, eqn);
            // Backward: propagate cotangents in reverse equation order.
            var i: usize = func.eqns.len;
            while (i > 0) {
                i -= 1;
                try ops.vjp_backward(ad_ctx, func.eqns[i]);
            }
        },
        .jvp => {
            // Single forward pass to compute primals and tangents together
            for (func.eqns) |eqn| {
                // NOTE: `vjp_forward` is likely a poor name should consider a rename
                try ops.vjp_forward(ad_ctx, eqn); // primal computation, shared with VJP forward pass
                try ops.jvp(ad_ctx, eqn);
            }
        },
    }

    // VJP harvests cotangents at inputs, JVP harvests tangents at outputs.
    const harvest_ids = switch (mode) {
        .vjp => func.params,
        .jvp => func.returns,
    };
    const extra: usize = if (include_value) func.returns.len else 0;
    const returns = try allocator.alloc(pr.VarId, extra + harvest_ids.len);
    defer allocator.free(returns);

    var out_idx: usize = 0;
    if (include_value) {
        for (func.returns) |ret_id| {
            returns[out_idx] = primal_map[@intCast(ret_id)] orelse return error.UnsupportedEqn;
            out_idx += 1;
        }
    }
    for (harvest_ids) |id| {
        if (dual_map[@intCast(id)]) |dual| {
            returns[out_idx] = dual;
        } else {
            const tensor = func.avals[@intCast(id)].as_tensor() orelse return error.UnsupportedEqn;
            returns[out_idx] = try b.scalar_broadcast(tensor.dtype, tensor.shape.dims, 0.0);
        }
        out_idx += 1;
    }

    return b.finish(returns);
}

/// Reverse-mode AD (pullback): transforms `f: M -> N` into
/// `vjp_f: (T_xM, T*_{f(x)}N) -> T*_xM`.
///
/// Concretely, computes the transpose-Jacobian product J^T(x) * v for a
/// cotangent seed `v`, which is the pullback f*: T*_{f(x)}N -> T*_xM
/// evaluated at `x`.
///
/// The returned function takes `N` primal inputs followed by `M` output
/// cotangent seeds, and returns `N` input cotangent vectors. In the
/// Euclidean / Cartesian case (G = I) these equal gradients; in general
/// they are covectors and must be raised with G^{-1} to obtain gradient
/// tangent vectors.
///
/// NOTE: In the Euclidean case (G = I), input cotangents from VJP coincide
///       numerically with gradients via the trivial musical isomorphism. For
///       non-Cartesian metrics, converting to gradients requires applying G^{-1}.
///
/// NOTE: Missing dual entries (ops w/o AD support) fall back to zero-filled tensors.
pub fn vjp(allocator: std.mem.Allocator, program: *pr.Program, func: pr.Function, name: []const u8) VjpError!pr.Function {
    return ad_impl(.vjp, allocator, program, func, name, false);
}

/// Like `vjp`, but the returned function also emits primal outputs before the input cotangents:
/// `(N primals, M cotangent seeds) -> (M primal outputs, N input cotangents)`.
///
/// NOTE: In the Euclidean case (G = I), input cotangents from VJP coincide
///       numerically with gradients via the trivial musical isomorphism. For
///       non-Cartesian metrics, converting to gradients requires applying G^{-1}.
///
/// NOTE: Missing dual entries (ops w/o AD support) fall back to zero-filled tensors.
pub fn vjp_with_value(allocator: std.mem.Allocator, program: *pr.Program, func: pr.Function, name: []const u8) VjpError!pr.Function {
    return ad_impl(.vjp, allocator, program, func, name, true);
}

/// Forward-mode AD (pushforward / differential): transforms `f: M -> N` into
/// `jvp_f: (T_xM, T_xM) -> T_{f(x)}N`.
///
/// Concretely, computes the Jacobian-vector product J(x) * v for a tangent
/// seed `v`, which is the differential df_x: T_xM -> T_{f(x)}N applied to `v`.
///
/// The returned function takes `N` primal inputs followed by `N` input tangent
/// vectors (same shapes), and returns `M` output tangent vectors matching the
/// original function's output shapes.
///
/// NOTE: Missing dual entries (ops w/o AD support) fall back to zero-filled tensors.
pub fn jvp(allocator: std.mem.Allocator, program: *pr.Program, func: pr.Function, name: []const u8) JvpError!pr.Function {
    return ad_impl(.jvp, allocator, program, func, name, false);
}

/// Like `jvp`, but the returned function also emits primal outputs before the output tangents:
/// `(N primals, N tangents) -> (M primal outputs, M output tangents)`.
///
/// NOTE: Missing dual entries (ops w/o AD support) fall back to zero-filled tensors.
pub fn jvp_with_value(allocator: std.mem.Allocator, program: *pr.Program, func: pr.Function, name: []const u8) JvpError!pr.Function {
    return ad_impl(.jvp, allocator, program, func, name, true);
}

test "vjp produces gradients matching input shapes" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a_id = try b.param_tensor(.f32, &.{ 2, 3 });
    const b_id = try b.param_tensor(.f32, &.{ 3, 2 });
    const c_id = try b.param_tensor(.f32, &.{ 2, 2 });

    const dot_id = try b.dot(a_id, b_id);
    const add_id = try b.add(dot_id, c_id);
    const out_id = try b.multiply(add_id, c_id);

    const func = try b.finish(&.{out_id});
    try program.add_function(func);

    const vjp_func = try vjp(std.testing.allocator, &program, func, "vjp");
    try pr.validate_function(vjp_func);

    try std.testing.expectEqual(@as(usize, func.params.len + func.returns.len), vjp_func.params.len);
    try std.testing.expectEqual(@as(usize, func.params.len), vjp_func.returns.len);

    for (func.params, 0..) |param_id, i| {
        const p_t = func.avals[@intCast(param_id)].as_tensor().?;
        const g_id = vjp_func.returns[i];
        const g_t = vjp_func.avals[@intCast(g_id)].as_tensor().?;
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

    const vjp_func = try vjp_with_value(std.testing.allocator, &program, func, "vjp_with_value");
    try pr.validate_function(vjp_func);

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

    const vjp_func = try vjp(std.testing.allocator, &program, func, "vjp");
    try pr.validate_function(vjp_func);

    try std.testing.expectEqual(@as(usize, func.params.len + func.returns.len), vjp_func.params.len);
    try std.testing.expectEqual(@as(usize, func.params.len), vjp_func.returns.len);

    for (func.params, 0..) |param_id, i| {
        const p_t = func.avals[@intCast(param_id)].as_tensor().?;
        const g_id = vjp_func.returns[i];
        const g_t = vjp_func.avals[@intCast(g_id)].as_tensor().?;
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

    const vjp_func = try vjp(std.testing.allocator, &program, func, "vjp");
    try pr.validate_function(vjp_func);
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

    const vjp_func = try vjp(std.testing.allocator, &program, func, "vjp");
    try pr.validate_function(vjp_func);
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

    const vjp_func = try vjp(std.testing.allocator, &program, func, "vjp");
    try pr.validate_function(vjp_func);
}

// ============================================================================
// JVP Tests
// ============================================================================

test "jvp produces tangent outputs matching function output shapes" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a_id = try b.param_tensor(.f32, &.{ 2, 3 });
    const b_id = try b.param_tensor(.f32, &.{ 3, 2 });
    const c_id = try b.param_tensor(.f32, &.{ 2, 2 });

    const dot_id = try b.dot(a_id, b_id);
    const add_id = try b.add(dot_id, c_id);
    const out_id = try b.multiply(add_id, c_id);

    const func = try b.finish(&.{out_id});
    try program.add_function(func);

    const jvp_func = try jvp(std.testing.allocator, &program, func, "jvp");
    try pr.validate_function(jvp_func);

    // JVP takes N primals + N tangents as params
    try std.testing.expectEqual(func.params.len * 2, jvp_func.params.len);
    // JVP returns M tangent outputs
    try std.testing.expectEqual(func.returns.len, jvp_func.returns.len);

    // Tangent output shapes must match original output shapes
    for (func.returns, 0..) |ret_id, i| {
        const orig_t = func.avals[@intCast(ret_id)].as_tensor().?;
        const jvp_id = jvp_func.returns[i];
        const jvp_t = jvp_func.avals[@intCast(jvp_id)].as_tensor().?;
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
    try pr.validate_function(jvp_func);

    // Returns M primals + M tangents
    try std.testing.expectEqual(func.returns.len * 2, jvp_func.returns.len);
    // Params: N primals + N tangents
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
    try pr.validate_function(jvp_func);

    // Tangent output shape must match original output
    const orig_t = func.avals[@intCast(func.returns[0])].as_tensor().?;
    const jvp_t = jvp_func.avals[@intCast(jvp_func.returns[0])].as_tensor().?;
    try std.testing.expectEqual(orig_t.dtype, jvp_t.dtype);
    try std.testing.expect(std.mem.eql(i64, orig_t.shape.dims, jvp_t.shape.dims));
}
