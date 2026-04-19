//! Automatic differentiation transforms on PR functions.
//!
//! This module implements the core AD transforms (VJP and JVP) at the PR
//! level. It operates on `pr.Function` values: given a function, it produces
//! a new function that computes derivatives.
//!
//! This a lower layer. Higher-level entry points are in frontend.
const std = @import("std");

const pr = @import("pr.zig");
const ops = @import("ops/ops.zig");

const log = std.log.scoped(.@"zg/ad");

pub const VjpError = ops.types.AdError;
pub const JvpError = ops.types.AdError;

/// Options for `vjp` and `vjp_with_value`.
pub const VjpOpts = struct {
    /// Restrict which primal-input indices contribute cotangents to the
    ///  returned function's outputs. When `null`, every input cotangent
    ///  is harvested. Frontends typically pass a slice like
    ///  `&.{0, 1, ..., K-1}` to request gradients only for the first `K`
    ///  input leaves (e.g. "params, not batch data").
    ///
    /// Duplicates and unsorted orderings are legal, the output tuple
    ///  follows the given order exactly.
    wrt: ?[]const usize = null,
};

/// AD mode
const Mode = enum {
    /// Reverse-mode AD. Propagates cotangents from outputs to inputs
    vjp,
    /// Forward-mode AD. Propagates tangents forward from inputs to outputs.
    jvp,
};

/// Whether the transformed function should also return the primal outputs
///  alongside the dual (cotangent/tangent) values.
///
/// Used internally by `ad_impl` to distinguish `vjp` from `vjp_with_value`.
const PrimalOutputs = enum { skip, emit };

/// Dtypes for which gradients/cotangents are defined.
///
/// Integer and unsigned dtypes are non-differentiable and conventionally carry
///  zero duals.
/// This is the same set enforced when seeding VJP cotangents.
fn is_differentiable_dtype(dtype: pr.DType) bool {
    return switch (dtype) {
        .f32, .f64, .bf16, .f16 => true,
        else => false,
    };
}

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
/// `wrt` restricts which duals are harvested into the returned function's
///  outputs:
///   - For VJP, it selects a subset of `func.params` (by index) whose
///   cotangents appear in the output.
///   - For JVP, it selects a subset of `func.returns`, `null` means
///     "all of them" (the default).
///  The backward traversal still runs over every op -- only the output tuple
///   shrinks -- but a smaller output tuple lets downstream DCE drop any
///   sub-graphs that feed only into dropped outputs.
///
/// NOTE: In the Euclidean case (G = I), input cotangents from VJP coincide
///       numerically with gradients via the trivial musical isomorphism. For
///       non-Cartesian metrics, converting to gradients requires applying G^{-1}.
///
/// NOTE: Missing duals on non-differentiable dtypes (integer inputs) are
///       filled with zero tensors. Missing duals on float dtypes return
///       `error.MissingDual`, typically a missing `vjp_backward` / `jvp`
///       implementation somewhere in the primal chain.
/// TODO: missing AD support might require a better policy, tbd.
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

    // Create parameters for primals.
    for (func.params) |param_var| {
        const tensor = param_var.aval.as_tensor();
        const new_param = try b.param_tensor(tensor.dtype, tensor.shape.dims);
        primal_map[param_var.id] = new_param;
    }

    // Seed the dual map - stores the "other half" of the primal/dual pair.
    // VJP: cotangent vectors (elements of T*_{f(x)}N) seeded at each output.
    // JVP: tangent vectors (elements of T_xM) seeded at each input.
    // Duality is symmetric: cotangents are dual to tangents and vice versa,
    //  this is not in reference to dual numbers as in some forward-mode impls.
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

    // TODO: we inline the forward but we should call the function instead, all backends will inline
    //  functions early anyways, and we can trivially inline functions in a PR pass as well, so this
    //  is just confusing and makes the IR unreadable.
    switch (mode) {
        .vjp => {
            for (func.ops) |op| try ops.vjp_forward(ad_ctx, op);
            // Backward: propagate cotangents in reverse op order.
            var i: usize = func.ops.len;
            while (i > 0) {
                i -= 1;
                try ops.vjp_backward(ad_ctx, func.ops[i]);
            }
        },
        .jvp => {
            // Single forward pass to compute primals and tangents together
            for (func.ops) |op| {
                // NOTE: `vjp_forward` is likely a poor name should consider a rename
                try ops.vjp_forward(ad_ctx, op); // primal computation, shared with VJP forward pass
                try ops.jvp(ad_ctx, op);
            }
        },
    }

    // VJP harvests cotangents at inputs, JVP harvests tangents at outputs.
    //  `wrt` (if provided) restricts which indices into `all_harvest_vars`
    //  produce outputs, otherwise we harvest all of them.
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
                // Float input with no dual at harvest time, this almost always means an op
                //  in the primal chain is missing a `vjp_backward` / `jvp` handler
                //  and silently dropped the propagation.
                // The legitimate "orphan float param" case (param does not flow to any
                //  output) is rare enough that failing loud here is the right default, if
                //  it starts mattering, gate via a VjpOpts.allow_orphan flag.
                log.err(
                    "no dual for float harvest var id={} dtype={s} shape={any} " ++
                        "likely a missing {s} in the primal chain",
                    .{
                        v.id,
                        @tagName(t.dtype),
                        t.shape.dims,
                        if (mode == .vjp) "vjp_backward" else "jvp",
                    },
                );
                return error.MissingDual;
            }
            // NOTE: Non-differentiable dtype (i32, i64, u8, ...): gradients are not defined
            //  for integer-valued inputs. By convention we return a zero tensor of the same
            //  shape so the output tuple has consistent arity regardless of input dtype.
            returns[out_idx] = try b.scalar_broadcast(t.dtype, t.shape.dims, 0.0);
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
/// See `VjpOpts` for how `opts.wrt` restricts which parameter cotangents
///  appear in the returned function's outputs. Default `.{}` harvests all
///  parameter cotangents.
///
/// NOTE: In the Euclidean case (G = I), input cotangents from VJP coincide
///       numerically with gradients via the trivial musical isomorphism. For
///       non-Cartesian metrics, converting to gradients requires applying G^{-1}.
///
/// NOTE: Missing duals on non-differentiable dtypes (integer inputs) are
///       filled with zero tensors. Missing duals on float dtypes return
///       `error.MissingDual`, typically a missing `vjp_backward` / `jvp`
///       implementation somewhere in the primal chain.
pub fn vjp(
    allocator: std.mem.Allocator,
    program: *pr.Program,
    func: pr.Function,
    name: []const u8,
    opts: VjpOpts,
) VjpError!pr.Function {
    return ad_impl(.vjp, allocator, program, func, name, .skip, opts.wrt);
}

/// Like `vjp`, but the returned function also emits primal outputs before the input cotangents:
/// `(N primals, M cotangent seeds) -> (M primal outputs, K input cotangents)`
///  where `K = opts.wrt.?.len` if provided, else `N`.
///
/// NOTE: In the Euclidean case (G = I), input cotangents from VJP coincide
///       numerically with gradients via the trivial musical isomorphism. For
///       non-Cartesian metrics, converting to gradients requires applying G^{-1}.
///
/// NOTE: Missing duals on non-differentiable dtypes (integer inputs) are
///       filled with zero tensors. Missing duals on float dtypes return
///       `error.MissingDual`, typically a missing `vjp_backward` / `jvp`
///       implementation somewhere in the primal chain.
pub fn vjp_with_value(
    allocator: std.mem.Allocator,
    program: *pr.Program,
    func: pr.Function,
    name: []const u8,
    opts: VjpOpts,
) VjpError!pr.Function {
    return ad_impl(.vjp, allocator, program, func, name, .emit, opts.wrt);
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
/// NOTE: Missing duals on non-differentiable dtypes (integer inputs) are
///       filled with zero tensors. Missing duals on float dtypes return
///       `error.MissingDual`, typically a missing `vjp_backward` / `jvp`
///       implementation somewhere in the primal chain.
pub fn jvp(allocator: std.mem.Allocator, program: *pr.Program, func: pr.Function, name: []const u8) JvpError!pr.Function {
    return ad_impl(.jvp, allocator, program, func, name, .skip, null);
}

/// Like `jvp`, but the returned function also emits primal outputs before the output tangents:
/// `(N primals, N tangents) -> (M primal outputs, M output tangents)`.
///
/// NOTE: Missing duals on non-differentiable dtypes (integer inputs) are
///       filled with zero tensors. Missing duals on float dtypes return
///       `error.MissingDual`, typically a missing `vjp_backward` / `jvp`
///       implementation somewhere in the primal chain.
pub fn jvp_with_value(allocator: std.mem.Allocator, program: *pr.Program, func: pr.Function, name: []const u8) JvpError!pr.Function {
    return ad_impl(.jvp, allocator, program, func, name, .emit, null);
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

    const dot_id = try b.dot(a_id, b_id);
    const add_id = try b.add(dot_id, c_id);
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
    try pr.validate_ops_in_func(jvp_func);

    // JVP takes N primals + N tangents as params
    try std.testing.expectEqual(func.params.len * 2, jvp_func.params.len);
    // JVP returns M tangent outputs
    try std.testing.expectEqual(func.returns.len, jvp_func.returns.len);

    // Tangent output shapes must match original output shapes
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
    try pr.validate_ops_in_func(jvp_func);

    // Tangent output shape must match original output
    const orig_t = func.returns[0].as_tensor();
    const jvp_t = jvp_func.returns[0].as_tensor();
    try std.testing.expectEqual(orig_t.dtype, jvp_t.dtype);
    try std.testing.expect(std.mem.eql(i64, orig_t.shape.dims, jvp_t.shape.dims));
}
