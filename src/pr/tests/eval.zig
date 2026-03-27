/// Host-side f32 interpreter for PR functions.
///
/// Evaluates a PR Function on concrete f32 data without requiring a backend.
/// Primary use: numerical gradient checking for AD correctness.
const std = @import("std");
const pr = @import("../pr.zig");
const log = std.log.scoped(.@"zg/eval");

// ============================================================================
// HostTensor
// ============================================================================

/// Dense f32 tensor with owned shape and data.
pub const HostTensor = struct {
    data: []f32,
    shape: []const i64,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, shape: []const i64) !HostTensor {
        const n = num_elements(shape);
        const data = try allocator.alloc(f32, n);
        @memset(data, 0);
        const owned_shape = try allocator.dupe(i64, shape);
        return .{ .data = data, .shape = owned_shape, .allocator = allocator };
    }

    pub fn init_with_data(allocator: std.mem.Allocator, shape: []const i64, src: []const f32) !HostTensor {
        const n = num_elements(shape);
        if (src.len != n) return error.ShapeMismatch;
        const data = try allocator.dupe(f32, src);
        const owned_shape = try allocator.dupe(i64, shape);
        return .{ .data = data, .shape = owned_shape, .allocator = allocator };
    }

    pub fn deinit(self: *HostTensor) void {
        self.allocator.free(self.data);
        self.allocator.free(self.shape);
        self.* = undefined;
    }

    pub fn fill(self: *HostTensor, val: f32) void {
        @memset(self.data, val);
    }

    pub fn clone(self: HostTensor) !HostTensor {
        return init_with_data(self.allocator, self.shape, self.data);
    }

    pub fn rank(self: HostTensor) usize {
        return self.shape.len;
    }

    pub fn num_elements(shape: []const i64) usize {
        var n: usize = 1;
        for (shape) |d| n *= @intCast(d);
        return n;
    }
};

pub const EvalError = error{
    ShapeMismatch,
    UnsupportedOp,
    MissingParam,
    InvalidParam,
    OutOfMemory,
    InvalidVarId,
};

// ============================================================================
// Public API
// ============================================================================

/// Evaluate a PR function on concrete f32 inputs.
///
/// Returns one HostTensor per return value. Caller owns all returned tensors
/// and must deinit them. The allocator is used for working memory and results
/// (NOT the program arena).
pub fn eval(
    allocator: std.mem.Allocator,
    func: pr.Function,
    inputs: []const HostTensor,
) EvalError![]HostTensor {
    if (inputs.len != func.params.len) return error.ShapeMismatch;

    // Environment: one optional HostTensor per VarId
    var env = try allocator.alloc(?HostTensor, func.avals.len);
    defer {
        // Free all intermediates that aren't returned
        for (env) |*slot| {
            if (slot.*) |*t| t.deinit();
        }
        allocator.free(env);
    }
    @memset(env, null);

    // Bind input parameters
    for (func.params, inputs) |param_id, input| {
        env[@intCast(param_id)] = try input.clone();
    }

    // Execute equations
    for (func.eqns) |eqn| {
        try eval_eqn(allocator, func, eqn, env);
    }

    // Extract return values. When the same VarId appears multiple times
    // in returns, clone on subsequent occurrences.
    const results = try allocator.alloc(HostTensor, func.returns.len);
    for (func.returns, 0..) |ret_id, i| {
        const idx: usize = @intCast(ret_id);
        if (env[idx]) |t| {
            results[i] = t;
            env[idx] = null; // Transfer ownership
        } else {
            // Already transferred - must be a duplicate return. Find the
            // earlier result that took ownership and clone from it.
            var found = false;
            for (func.returns[0..i]) |prev_id| {
                if (prev_id == ret_id) {
                    // Clone from the result that already owns this tensor
                    for (func.returns[0..i], 0..) |pid, j| {
                        if (pid == ret_id) {
                            results[i] = try results[j].clone();
                            found = true;
                            break;
                        }
                    }
                    break;
                }
            }
            if (!found) {
                log.err("return var {d} is null in env (func={s})", .{ idx, func.name });
                return error.InvalidVarId;
            }
        }
    }
    return results;
}

// ============================================================================
// Equation Dispatch
// ============================================================================

fn eval_eqn(
    allocator: std.mem.Allocator,
    func: pr.Function,
    eqn: pr.Eqn,
    env: []?HostTensor,
) EvalError!void {
    const inputs = eqn.inputs.slice(pr.VarId, func.varids_store);
    const outputs = eqn.outputs.slice(pr.VarId, func.varids_store);
    const params = eqn.params.slice(pr.Param, func.params_store);

    const result: HostTensor = switch (eqn.prim) {
        .literal => try eval_literal(allocator, params),
        .add => try eval_binary(allocator, env, inputs, add_fn),
        .subtract => try eval_binary(allocator, env, inputs, sub_fn),
        .multiply => try eval_binary(allocator, env, inputs, mul_fn),
        .divide => try eval_binary(allocator, env, inputs, div_fn),
        .maximum => try eval_binary(allocator, env, inputs, max_fn),
        .exp => try eval_unary(allocator, env, inputs, exp_fn),
        .log => try eval_unary(allocator, env, inputs, log_fn),
        .rsqrt => try eval_unary(allocator, env, inputs, rsqrt_fn),
        .logistic => try eval_unary(allocator, env, inputs, logistic_fn),
        .compare => try eval_compare(allocator, env, inputs, params),
        .select => try eval_select(allocator, env, inputs),
        .convert => try eval_convert(allocator, env, inputs, func, outputs),
        .reshape => try eval_reshape(allocator, env, inputs, params),
        .transpose => try eval_transpose(allocator, env, inputs, params),
        .broadcast_in_dim => try eval_broadcast_in_dim(allocator, env, inputs, params),
        .reduce_sum => try eval_reduce_sum(allocator, env, inputs, params, func, outputs),
        .reduce_max => try eval_reduce_max(allocator, env, inputs, params, func, outputs),
        .dot => try eval_dot(allocator, env, inputs),
        .dot_general => try eval_dot_general(allocator, env, inputs, params, func, outputs),
        .gather => try eval_gather(allocator, env, inputs, params, func, outputs),
        .scatter => try eval_scatter(allocator, env, inputs, params, func, outputs),
        .iota => try eval_iota(allocator, params, func, outputs),
        .slice => try eval_slice(allocator, env, inputs, params),
        .concatenate => try eval_concatenate(allocator, env, inputs, params, func),
        .call, .custom_call => return error.UnsupportedOp,
    };

    if (outputs.len != 1) return error.UnsupportedOp;
    env[@intCast(outputs[0])] = result;
}

// ============================================================================
// Op Implementations
// ============================================================================

fn add_fn(a: f32, b: f32) f32 {
    return a + b;
}
fn sub_fn(a: f32, b: f32) f32 {
    return a - b;
}
fn mul_fn(a: f32, b: f32) f32 {
    return a * b;
}
fn div_fn(a: f32, b: f32) f32 {
    return a / b;
}
fn max_fn(a: f32, b: f32) f32 {
    return @max(a, b);
}
fn exp_fn(x: f32) f32 {
    return @exp(x);
}
fn log_fn(x: f32) f32 {
    return @log(x);
}
fn rsqrt_fn(x: f32) f32 {
    return 1.0 / @sqrt(x);
}
fn logistic_fn(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn eval_literal(allocator: std.mem.Allocator, params: []const pr.Param) EvalError!HostTensor {
    const lit = pr.param(.literal,params) orelse return error.MissingParam;
    var t = try HostTensor.init(allocator, &.{});
    t.data[0] = switch (lit) {
        .f32 => |v| v,
        .f64 => |v| @floatCast(v),
        .i8 => |v| @floatFromInt(v),
        .u8 => |v| @floatFromInt(v),
        .i32 => |v| @floatFromInt(v),
        .i64 => |v| @floatFromInt(v),
        .u32 => |v| @floatFromInt(v),
        .u64 => |v| @floatFromInt(v),
        .bool => |v| if (v) @as(f32, 1.0) else 0.0,
        .f16, .bf16 => |v| blk: {
            const bits: u32 = @as(u32, v) << 16;
            break :blk @bitCast(bits);
        },
    };
    return t;
}

fn eval_binary(
    allocator: std.mem.Allocator,
    env: []?HostTensor,
    inputs: []const pr.VarId,
    op: *const fn (f32, f32) f32,
) EvalError!HostTensor {
    const lhs = env[@intCast(inputs[0])] orelse return error.InvalidVarId;
    const rhs = env[@intCast(inputs[1])] orelse return error.InvalidVarId;
    const result = try HostTensor.init(allocator, lhs.shape);
    for (result.data, 0..) |*out, i| {
        out.* = op(lhs.data[i], rhs.data[i]);
    }
    return result;
}

fn eval_unary(
    allocator: std.mem.Allocator,
    env: []?HostTensor,
    inputs: []const pr.VarId,
    op: *const fn (f32) f32,
) EvalError!HostTensor {
    const operand = env[@intCast(inputs[0])] orelse return error.InvalidVarId;
    const result = try HostTensor.init(allocator, operand.shape);
    for (result.data, 0..) |*out, i| {
        out.* = op(operand.data[i]);
    }
    return result;
}

fn eval_compare(
    allocator: std.mem.Allocator,
    env: []?HostTensor,
    inputs: []const pr.VarId,
    params: []const pr.Param,
) EvalError!HostTensor {
    const lhs = env[@intCast(inputs[0])] orelse return error.InvalidVarId;
    const rhs = env[@intCast(inputs[1])] orelse return error.InvalidVarId;
    const cparams = pr.param(.compare,params) orelse return error.MissingParam;
    const result = try HostTensor.init(allocator, lhs.shape);
    for (result.data, 0..) |*out, i| {
        const a = lhs.data[i];
        const b = rhs.data[i];
        const cmp: bool = switch (cparams.direction) {
            .EQ => a == b,
            .NE => a != b,
            .GE => a >= b,
            .GT => a > b,
            .LE => a <= b,
            .LT => a < b,
        };
        out.* = if (cmp) 1.0 else 0.0;
    }
    return result;
}

fn eval_select(
    allocator: std.mem.Allocator,
    env: []?HostTensor,
    inputs: []const pr.VarId,
) EvalError!HostTensor {
    const cond = env[@intCast(inputs[0])] orelse return error.InvalidVarId;
    const on_true = env[@intCast(inputs[1])] orelse return error.InvalidVarId;
    const on_false = env[@intCast(inputs[2])] orelse return error.InvalidVarId;
    const result = try HostTensor.init(allocator, on_true.shape);
    for (result.data, 0..) |*out, i| {
        out.* = if (cond.data[i] != 0.0) on_true.data[i] else on_false.data[i];
    }
    return result;
}

fn eval_convert(
    allocator: std.mem.Allocator,
    env: []?HostTensor,
    inputs: []const pr.VarId,
    func: pr.Function,
    outputs: []const pr.VarId,
) EvalError!HostTensor {
    // For f32 eval, convert is effectively identity on shape
    const operand = env[@intCast(inputs[0])] orelse return error.InvalidVarId;
    const out_tensor = func.avals[@intCast(outputs[0])].as_tensor() orelse return error.InvalidVarId;
    const result = try HostTensor.init(allocator, out_tensor.shape.dims);
    @memcpy(result.data, operand.data);
    return result;
}

fn eval_reshape(
    allocator: std.mem.Allocator,
    env: []?HostTensor,
    inputs: []const pr.VarId,
    params: []const pr.Param,
) EvalError!HostTensor {
    const operand = env[@intCast(inputs[0])] orelse return error.InvalidVarId;
    const out_shape = pr.param(.out_shape,params) orelse return error.MissingParam;
    const result = try HostTensor.init(allocator, out_shape);
    @memcpy(result.data, operand.data);
    return result;
}

fn eval_transpose(
    allocator: std.mem.Allocator,
    env: []?HostTensor,
    inputs: []const pr.VarId,
    params: []const pr.Param,
) EvalError!HostTensor {
    const operand = env[@intCast(inputs[0])] orelse return error.InvalidVarId;
    const perm = pr.param(.permutation,params) orelse return error.MissingParam;
    const in_shape = operand.shape;
    const ndim = in_shape.len;

    var out_shape_buf: [64]i64 = undefined;
    for (perm, 0..) |p, i| {
        out_shape_buf[i] = in_shape[@intCast(p)];
    }
    const out_shape = out_shape_buf[0..ndim];

    var result = try HostTensor.init(allocator, out_shape);

    // For each output element, compute the corresponding input element
    const n = result.data.len;
    for (0..n) |out_flat| {
        // Convert flat -> multi-index in output
        var out_idx: [64]usize = undefined;
        flat_to_multi(out_flat, out_shape, out_idx[0..ndim]);

        // Map to input multi-index via inverse permutation
        var in_idx: [64]usize = undefined;
        for (perm, 0..) |p, d| {
            in_idx[@intCast(p)] = out_idx[d];
        }

        const in_flat = multi_to_flat(in_idx[0..ndim], in_shape);
        result.data[out_flat] = operand.data[in_flat];
    }
    return result;
}

fn eval_broadcast_in_dim(
    allocator: std.mem.Allocator,
    env: []?HostTensor,
    inputs: []const pr.VarId,
    params: []const pr.Param,
) EvalError!HostTensor {
    const operand = env[@intCast(inputs[0])] orelse return error.InvalidVarId;
    const out_shape = pr.param(.out_shape,params) orelse return error.MissingParam;
    const broadcast_dims = pr.param(.broadcast_dimensions,params) orelse return error.MissingParam;

    var result = try HostTensor.init(allocator, out_shape);
    const out_ndim = out_shape.len;
    const n = result.data.len;

    for (0..n) |out_flat| {
        var out_idx: [64]usize = undefined;
        flat_to_multi(out_flat, out_shape, out_idx[0..out_ndim]);

        // Map output index to input index
        var in_idx: [64]usize = undefined;
        for (broadcast_dims, 0..) |bd, i| {
            const out_d = out_idx[@intCast(bd)];
            // If input dim is 1, broadcast (index 0), else use output index
            in_idx[i] = if (operand.shape[i] == 1) 0 else out_d;
        }

        const in_flat = multi_to_flat(in_idx[0..operand.shape.len], operand.shape);
        result.data[out_flat] = operand.data[in_flat];
    }
    return result;
}

fn eval_reduce_sum(
    allocator: std.mem.Allocator,
    env: []?HostTensor,
    inputs: []const pr.VarId,
    params: []const pr.Param,
    func: pr.Function,
    outputs: []const pr.VarId,
) EvalError!HostTensor {
    const operand = env[@intCast(inputs[0])] orelse return error.InvalidVarId;
    const axes = pr.param(.reduce_axes,params) orelse return error.MissingParam;
    const out_tensor = func.avals[@intCast(outputs[0])].as_tensor() orelse return error.InvalidVarId;
    const out_shape = out_tensor.shape.dims;

    var result = try HostTensor.init(allocator, out_shape);
    const in_shape = operand.shape;
    const in_ndim = in_shape.len;

    // Mark reduced axes
    var is_reduced: [64]bool = .{false} ** 64;
    for (axes) |a| is_reduced[@intCast(a)] = true;

    // For each input element, accumulate into the right output position
    for (0..operand.data.len) |in_flat| {
        var in_idx: [64]usize = undefined;
        flat_to_multi(in_flat, in_shape, in_idx[0..in_ndim]);

        // Build output index by dropping reduced dims
        var out_idx: [64]usize = undefined;
        var oi: usize = 0;
        for (0..in_ndim) |d| {
            if (!is_reduced[d]) {
                out_idx[oi] = in_idx[d];
                oi += 1;
            }
        }

        const out_flat = if (out_shape.len == 0) 0 else multi_to_flat(out_idx[0..out_shape.len], out_shape);
        result.data[out_flat] += operand.data[in_flat];
    }
    return result;
}

fn eval_reduce_max(
    allocator: std.mem.Allocator,
    env: []?HostTensor,
    inputs: []const pr.VarId,
    params: []const pr.Param,
    func: pr.Function,
    outputs: []const pr.VarId,
) EvalError!HostTensor {
    const operand = env[@intCast(inputs[0])] orelse return error.InvalidVarId;
    const axes = pr.param(.reduce_axes,params) orelse return error.MissingParam;
    const out_tensor = func.avals[@intCast(outputs[0])].as_tensor() orelse return error.InvalidVarId;
    const out_shape = out_tensor.shape.dims;

    var result = try HostTensor.init(allocator, out_shape);
    @memset(result.data, -std.math.inf(f32));
    const in_shape = operand.shape;
    const in_ndim = in_shape.len;

    var is_reduced: [64]bool = .{false} ** 64;
    for (axes) |a| is_reduced[@intCast(a)] = true;

    for (0..operand.data.len) |in_flat| {
        var in_idx: [64]usize = undefined;
        flat_to_multi(in_flat, in_shape, in_idx[0..in_ndim]);

        var out_idx: [64]usize = undefined;
        var oi: usize = 0;
        for (0..in_ndim) |d| {
            if (!is_reduced[d]) {
                out_idx[oi] = in_idx[d];
                oi += 1;
            }
        }

        const out_flat = if (out_shape.len == 0) 0 else multi_to_flat(out_idx[0..out_shape.len], out_shape);
        result.data[out_flat] = @max(result.data[out_flat], operand.data[in_flat]);
    }
    return result;
}

fn eval_dot(
    allocator: std.mem.Allocator,
    env: []?HostTensor,
    inputs: []const pr.VarId,
) EvalError!HostTensor {
    const lhs = env[@intCast(inputs[0])] orelse return error.InvalidVarId;
    const rhs = env[@intCast(inputs[1])] orelse return error.InvalidVarId;
    const m: usize = @intCast(lhs.shape[0]);
    const k: usize = @intCast(lhs.shape[1]);
    const n: usize = @intCast(rhs.shape[1]);

    var result = try HostTensor.init(allocator, &.{ lhs.shape[0], rhs.shape[1] });
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..k) |p| {
                sum += lhs.data[i * k + p] * rhs.data[p * n + j];
            }
            result.data[i * n + j] = sum;
        }
    }
    return result;
}

fn eval_dot_general(
    allocator: std.mem.Allocator,
    env: []?HostTensor,
    inputs: []const pr.VarId,
    params: []const pr.Param,
    func: pr.Function,
    outputs: []const pr.VarId,
) EvalError!HostTensor {
    const lhs = env[@intCast(inputs[0])] orelse return error.InvalidVarId;
    const rhs = env[@intCast(inputs[1])] orelse return error.InvalidVarId;
    const dg = pr.param(.dot_general,params) orelse return error.MissingParam;
    const out_tensor = func.avals[@intCast(outputs[0])].as_tensor() orelse return error.InvalidVarId;
    const out_shape = out_tensor.shape.dims;

    const lhs_shape = lhs.shape;
    const rhs_shape = rhs.shape;
    const lhs_rank = lhs_shape.len;
    const rhs_rank = rhs_shape.len;

    // Classify lhs dims
    var lhs_is_batch: [64]bool = .{false} ** 64;
    var lhs_is_contract: [64]bool = .{false} ** 64;
    for (dg.lhs_batch_dims) |d| lhs_is_batch[@intCast(d)] = true;
    for (dg.lhs_contracting_dims) |d| lhs_is_contract[@intCast(d)] = true;

    var rhs_is_batch: [64]bool = .{false} ** 64;
    var rhs_is_contract: [64]bool = .{false} ** 64;
    for (dg.rhs_batch_dims) |d| rhs_is_batch[@intCast(d)] = true;
    for (dg.rhs_contracting_dims) |d| rhs_is_contract[@intCast(d)] = true;

    // Collect free dims
    var lhs_free: [64]usize = undefined;
    var lhs_free_len: usize = 0;
    for (0..lhs_rank) |d| {
        if (!lhs_is_batch[d] and !lhs_is_contract[d]) {
            lhs_free[lhs_free_len] = d;
            lhs_free_len += 1;
        }
    }
    var rhs_free: [64]usize = undefined;
    var rhs_free_len: usize = 0;
    for (0..rhs_rank) |d| {
        if (!rhs_is_batch[d] and !rhs_is_contract[d]) {
            rhs_free[rhs_free_len] = d;
            rhs_free_len += 1;
        }
    }

    var result = try HostTensor.init(allocator, out_shape);
    const out_ndim = out_shape.len;
    const n_out = result.data.len;

    // Contract dim sizes
    var contract_shape: [64]i64 = undefined;
    const contract_len = dg.lhs_contracting_dims.len;
    for (dg.lhs_contracting_dims, 0..) |d, i| {
        contract_shape[i] = lhs_shape[@intCast(d)];
    }

    var contract_size: usize = 1;
    for (contract_shape[0..contract_len]) |s| contract_size *= @as(usize, @intCast(s));

    for (0..n_out) |out_flat| {
        var out_idx: [64]usize = undefined;
        flat_to_multi(out_flat, out_shape, out_idx[0..out_ndim]);

        // Build partial lhs/rhs indices from output
        var lhs_idx: [64]usize = undefined;
        var rhs_idx: [64]usize = undefined;

        // Batch dims
        for (dg.lhs_batch_dims, dg.rhs_batch_dims, 0..) |lb, rb, bi| {
            lhs_idx[@intCast(lb)] = out_idx[bi];
            rhs_idx[@intCast(rb)] = out_idx[bi];
        }

        // Free dims from output
        const batch_len = dg.lhs_batch_dims.len;
        for (0..lhs_free_len) |fi| {
            lhs_idx[lhs_free[fi]] = out_idx[batch_len + fi];
        }
        for (0..rhs_free_len) |fi| {
            rhs_idx[rhs_free[fi]] = out_idx[batch_len + lhs_free_len + fi];
        }

        // Sum over contracting dims
        var sum: f32 = 0;
        for (0..contract_size) |c| {
            // Decompose c into contracting multi-index
            var contract_idx: [64]usize = undefined;
            flat_to_multi(c, contract_shape[0..contract_len], contract_idx[0..contract_len]);

            for (dg.lhs_contracting_dims, dg.rhs_contracting_dims, 0..) |lc, rc, ci| {
                lhs_idx[@intCast(lc)] = contract_idx[ci];
                rhs_idx[@intCast(rc)] = contract_idx[ci];
            }

            const lhs_flat = multi_to_flat(lhs_idx[0..lhs_rank], lhs_shape);
            const rhs_flat = multi_to_flat(rhs_idx[0..rhs_rank], rhs_shape);
            sum += lhs.data[lhs_flat] * rhs.data[rhs_flat];
        }
        result.data[out_flat] = sum;
    }
    return result;
}

fn eval_gather(
    allocator: std.mem.Allocator,
    env: []?HostTensor,
    inputs: []const pr.VarId,
    params: []const pr.Param,
    func: pr.Function,
    outputs: []const pr.VarId,
) EvalError!HostTensor {
    const operand = env[@intCast(inputs[0])] orelse return error.InvalidVarId;
    const indices = env[@intCast(inputs[1])] orelse return error.InvalidVarId;
    const gp = pr.param(.gather,params) orelse return error.MissingParam;
    const out_tensor = func.avals[@intCast(outputs[0])].as_tensor() orelse return error.InvalidVarId;
    const out_shape = out_tensor.shape.dims;

    const operand_shape = operand.shape;
    const indices_shape = indices.shape;
    const operand_rank = operand_shape.len;
    const out_ndim = out_shape.len;
    const index_vector_dim: usize = @intCast(gp.index_vector_dim);

    var is_collapsed: [64]bool = .{false} ** 64;
    for (gp.collapsed_slice_dims) |d| is_collapsed[@intCast(d)] = true;

    var result = try HostTensor.init(allocator, out_shape);

    for (0..result.data.len) |out_flat| {
        var out_idx: [64]usize = undefined;
        flat_to_multi(out_flat, out_shape, out_idx[0..out_ndim]);

        // Split output index into batch_indices and offset_indices
        // Batch dims: all output dims not in offset_dims
        // offset_dims: gp.offset_dims

        // Extract the index vector from indices tensor
        // Build indices tensor multi-index from batch portion of output
        var indices_idx: [64]usize = undefined;
        var bi: usize = 0;
        for (0..out_ndim) |d| {
            var is_offset = false;
            for (gp.offset_dims) |od| {
                if (d == @as(usize, @intCast(od))) {
                    is_offset = true;
                    break;
                }
            }
            if (!is_offset) {
                // This is a batch dim in the output
                if (bi < index_vector_dim) {
                    indices_idx[bi] = out_idx[d];
                } else {
                    indices_idx[bi + 1] = out_idx[d];
                }
                bi += 1;
            }
        }

        // Get start indices from the indices tensor
        var operand_idx: [64]usize = undefined;
        for (gp.start_index_map, 0..) |sim, si| {
            // Read the index value
            if (index_vector_dim < indices_shape.len) {
                indices_idx[index_vector_dim] = si;
            }
            const idx_flat = if (indices_shape.len == 0) 0 else multi_to_flat(
                indices_idx[0..indices_shape.len],
                indices_shape,
            );
            const start: usize = @intFromFloat(indices.data[idx_flat]);
            operand_idx[@intCast(sim)] = start;
        }

        // Add offset indices
        var offset_i: usize = 0;
        for (0..operand_rank) |d| {
            if (is_collapsed[d]) continue;
            // Find this offset dim in out_idx
            const offset_dim: usize = @intCast(gp.offset_dims[offset_i]);
            operand_idx[d] += out_idx[offset_dim];
            offset_i += 1;
        }

        // For collapsed dims, start index was already set above (or 0)
        for (0..operand_rank) |d| {
            if (!is_collapsed[d]) continue;
            // Already set via start_index_map, nothing to add
            var found_in_map = false;
            for (gp.start_index_map) |sim| {
                if (@as(usize, @intCast(sim)) == d) {
                    found_in_map = true;
                    break;
                }
            }
            if (!found_in_map) {
                operand_idx[d] = 0;
            }
        }

        const op_flat = multi_to_flat(operand_idx[0..operand_rank], operand_shape);
        result.data[out_flat] = operand.data[op_flat];
    }
    return result;
}

fn eval_scatter(
    allocator: std.mem.Allocator,
    env: []?HostTensor,
    inputs: []const pr.VarId,
    params: []const pr.Param,
    func: pr.Function,
    outputs: []const pr.VarId,
) EvalError!HostTensor {
    const input = env[@intCast(inputs[0])] orelse return error.InvalidVarId;
    const scatter_indices = env[@intCast(inputs[1])] orelse return error.InvalidVarId;
    const updates = env[@intCast(inputs[2])] orelse return error.InvalidVarId;
    const sp = pr.param(.scatter,params) orelse return error.MissingParam;
    const out_tensor = func.avals[@intCast(outputs[0])].as_tensor() orelse return error.InvalidVarId;
    const out_shape = out_tensor.shape.dims;
    const out_rank = out_shape.len;

    // Start with copy of input
    var result = try HostTensor.init_with_data(allocator, out_shape, input.data);

    const updates_shape = updates.shape;
    const updates_ndim = updates_shape.len;
    const indices_shape = scatter_indices.shape;
    const index_vector_dim: usize = @intCast(sp.index_vector_dim);

    var is_inserted: [64]bool = .{false} ** 64;
    for (sp.inserted_window_dims) |d| is_inserted[@intCast(d)] = true;

    // For each update element
    for (0..updates.data.len) |update_flat| {
        var update_idx: [64]usize = undefined;
        flat_to_multi(update_flat, updates_shape, update_idx[0..updates_ndim]);

        // Split update index into scatter dims and window dims
        // Scatter dims: dims not in update_window_dims
        var is_window_dim: [64]bool = .{false} ** 64;
        for (sp.update_window_dims) |d| is_window_dim[@intCast(d)] = true;

        // Build indices tensor index from scatter dims
        var indices_idx: [64]usize = undefined;
        var si: usize = 0;
        for (0..updates_ndim) |d| {
            if (!is_window_dim[d]) {
                if (si < index_vector_dim) {
                    indices_idx[si] = update_idx[d];
                } else {
                    indices_idx[si + 1] = update_idx[d];
                }
                si += 1;
            }
        }

        // Get scatter start indices
        var operand_idx: [64]usize = undefined;
        @memset(operand_idx[0..out_rank], 0);

        for (sp.scatter_dims_to_operand_dims, 0..) |sdod, di| {
            if (index_vector_dim < indices_shape.len) {
                indices_idx[index_vector_dim] = di;
            }
            const idx_flat = if (indices_shape.len == 0) 0 else multi_to_flat(
                indices_idx[0..indices_shape.len],
                indices_shape,
            );
            const start: usize = @intFromFloat(scatter_indices.data[idx_flat]);
            operand_idx[@intCast(sdod)] = start;
        }

        // Add window offsets
        var wi: usize = 0;
        for (0..out_rank) |d| {
            if (is_inserted[d]) continue;
            const window_dim: usize = @intCast(sp.update_window_dims[wi]);
            operand_idx[d] += update_idx[window_dim];
            wi += 1;
        }

        const op_flat = multi_to_flat(operand_idx[0..out_rank], out_shape);

        // Apply reduction
        switch (sp.reduction) {
            .add => result.data[op_flat] += updates.data[update_flat],
            .mul => result.data[op_flat] *= updates.data[update_flat],
            .max => result.data[op_flat] = @max(result.data[op_flat], updates.data[update_flat]),
            .min => result.data[op_flat] = @min(result.data[op_flat], updates.data[update_flat]),
        }
    }
    return result;
}

fn eval_iota(
    allocator: std.mem.Allocator,
    params: []const pr.Param,
    func: pr.Function,
    outputs: []const pr.VarId,
) EvalError!HostTensor {
    const out_shape = pr.param(.out_shape,params) orelse return error.MissingParam;
    const iota_dim = pr.param(.iota_dimension,params) orelse return error.MissingParam;
    _ = func.avals[@intCast(outputs[0])].as_tensor() orelse return error.InvalidVarId;

    var result = try HostTensor.init(allocator, out_shape);
    const ndim = out_shape.len;
    const dim_idx: usize = @intCast(iota_dim);

    for (0..result.data.len) |flat| {
        var idx: [64]usize = undefined;
        flat_to_multi(flat, out_shape, idx[0..ndim]);
        result.data[flat] = @floatFromInt(idx[dim_idx]);
    }
    return result;
}

fn eval_slice(
    allocator: std.mem.Allocator,
    env: []?HostTensor,
    inputs: []const pr.VarId,
    params: []const pr.Param,
) EvalError!HostTensor {
    const operand = env[@intCast(inputs[0])] orelse return error.InvalidVarId;
    const sp = pr.param(.slice,params) orelse return error.MissingParam;
    const in_shape = operand.shape;
    const ndim = in_shape.len;

    // Compute output shape
    var out_shape_buf: [64]i64 = undefined;
    for (0..ndim) |d| {
        const start: i64 = sp.start_indices[d];
        const limit: i64 = sp.limit_indices[d];
        const stride: i64 = sp.strides[d];
        out_shape_buf[d] = @divTrunc(limit - start + stride - 1, stride);
    }
    const out_shape = out_shape_buf[0..ndim];

    var result = try HostTensor.init(allocator, out_shape);

    for (0..result.data.len) |out_flat| {
        var out_idx: [64]usize = undefined;
        flat_to_multi(out_flat, out_shape, out_idx[0..ndim]);

        var in_idx: [64]usize = undefined;
        for (0..ndim) |d| {
            in_idx[d] = @as(usize, @intCast(sp.start_indices[d])) + out_idx[d] * @as(usize, @intCast(sp.strides[d]));
        }

        const in_flat = multi_to_flat(in_idx[0..ndim], in_shape);
        result.data[out_flat] = operand.data[in_flat];
    }
    return result;
}

fn eval_concatenate(
    allocator: std.mem.Allocator,
    env: []?HostTensor,
    inputs: []const pr.VarId,
    params: []const pr.Param,
    func: pr.Function,
) EvalError!HostTensor {
    const axis: usize = @intCast(pr.param(.concat_axis,params) orelse return error.MissingParam);

    // Compute output shape
    const first = env[@intCast(inputs[0])] orelse return error.InvalidVarId;
    const ndim = first.shape.len;
    var out_shape_buf: [64]i64 = undefined;
    @memcpy(out_shape_buf[0..ndim], first.shape);

    var total_axis: i64 = first.shape[axis];
    for (inputs[1..]) |inp_id| {
        const t = env[@intCast(inp_id)] orelse return error.InvalidVarId;
        total_axis += t.shape[axis];
    }
    out_shape_buf[axis] = total_axis;
    const out_shape = out_shape_buf[0..ndim];

    var result = try HostTensor.init(allocator, out_shape);

    // Copy data from each input
    const varids = func.varids_store;
    _ = varids;
    var axis_offset: usize = 0;
    for (inputs) |inp_id| {
        const t = env[@intCast(inp_id)] orelse return error.InvalidVarId;
        const t_shape = t.shape;

        for (0..t.data.len) |in_flat| {
            var in_idx: [64]usize = undefined;
            flat_to_multi(in_flat, t_shape, in_idx[0..ndim]);

            var out_idx: [64]usize = undefined;
            @memcpy(out_idx[0..ndim], in_idx[0..ndim]);
            out_idx[axis] += axis_offset;

            const out_flat = multi_to_flat(out_idx[0..ndim], out_shape);
            result.data[out_flat] = t.data[in_flat];
        }
        axis_offset += @intCast(t_shape[axis]);
    }
    return result;
}

// ============================================================================
// Indexing Helpers
// ============================================================================

fn flat_to_multi(flat: usize, shape: []const i64, out: []usize) void {
    var remaining = flat;
    var d: usize = shape.len;
    while (d > 0) {
        d -= 1;
        const dim: usize = @intCast(shape[d]);
        if (dim == 0) {
            out[d] = 0;
        } else {
            out[d] = remaining % dim;
            remaining /= dim;
        }
    }
}

fn multi_to_flat(idx: []const usize, shape: []const i64) usize {
    var flat: usize = 0;
    var stride: usize = 1;
    var d: usize = shape.len;
    while (d > 0) {
        d -= 1;
        flat += idx[d] * stride;
        stride *= @as(usize, @intCast(shape[d]));
    }
    return flat;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

fn build_and_finish(program: *pr.Program, b: *pr.FunctionBuilder, returns: []const pr.VarId) !pr.Function {
    const func = try b.finish(returns);
    try program.add_function(func);
    return func;
}

test "eval: literal scalar" {
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "lit");
    defer b.deinit();
    const v = try b.literal_scalar(.{ .f32 = 3.14 });
    const func = try build_and_finish(&program, &b, &.{v});

    const results = try eval(testing.allocator, func, &.{});
    defer {
        for (results) |*r| r.deinit();
        testing.allocator.free(results);
    }
    try testing.expectEqual(@as(usize, 1), results.len);
    try testing.expectApproxEqAbs(@as(f32, 3.14), results[0].data[0], 1e-6);
}

test "eval: add elementwise" {
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "add");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{3});
    const y = try b.param_tensor(.f32, &.{3});
    const z = try b.add(x, y);
    const func = try build_and_finish(&program, &b, &.{z});

    var in_x = try HostTensor.init_with_data(testing.allocator, &.{3}, &.{ 1.0, 2.0, 3.0 });
    defer in_x.deinit();
    var in_y = try HostTensor.init_with_data(testing.allocator, &.{3}, &.{ 4.0, 5.0, 6.0 });
    defer in_y.deinit();

    const results = try eval(testing.allocator, func, &.{ in_x, in_y });
    defer {
        for (results) |*r| r.deinit();
        testing.allocator.free(results);
    }
    try testing.expectApproxEqAbs(@as(f32, 5.0), results[0].data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 7.0), results[0].data[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 9.0), results[0].data[2], 1e-6);
}

test "eval: multiply elementwise" {
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "mul");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 2 });
    const y = try b.param_tensor(.f32, &.{ 2, 2 });
    const z = try b.multiply(x, y);
    const func = try build_and_finish(&program, &b, &.{z});

    var in_x = try HostTensor.init_with_data(testing.allocator, &.{ 2, 2 }, &.{ 1.0, 2.0, 3.0, 4.0 });
    defer in_x.deinit();
    var in_y = try HostTensor.init_with_data(testing.allocator, &.{ 2, 2 }, &.{ 5.0, 6.0, 7.0, 8.0 });
    defer in_y.deinit();

    const results = try eval(testing.allocator, func, &.{ in_x, in_y });
    defer {
        for (results) |*r| r.deinit();
        testing.allocator.free(results);
    }
    try testing.expectApproxEqAbs(@as(f32, 5.0), results[0].data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 12.0), results[0].data[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 21.0), results[0].data[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 32.0), results[0].data[3], 1e-6);
}

test "eval: dot matmul 2x3 @ 3x2" {
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "dot");
    defer b.deinit();
    const a = try b.param_tensor(.f32, &.{ 2, 3 });
    const bb = try b.param_tensor(.f32, &.{ 3, 2 });
    const c = try b.dot(a, bb);
    const func = try build_and_finish(&program, &b, &.{c});

    // [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]]
    var in_a = try HostTensor.init_with_data(testing.allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer in_a.deinit();
    var in_b = try HostTensor.init_with_data(testing.allocator, &.{ 3, 2 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer in_b.deinit();

    const results = try eval(testing.allocator, func, &.{ in_a, in_b });
    defer {
        for (results) |*r| r.deinit();
        testing.allocator.free(results);
    }
    // [22, 28, 49, 64]
    try testing.expectApproxEqAbs(@as(f32, 22.0), results[0].data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 28.0), results[0].data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 49.0), results[0].data[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 64.0), results[0].data[3], 1e-5);
}

test "eval: dot_general batched" {
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "dg");
    defer b.deinit();
    // [2,2,3] x [2,3,2] -> [2,2,2], batch dim 0
    const lhs = try b.param_tensor(.f32, &.{ 2, 2, 3 });
    const rhs = try b.param_tensor(.f32, &.{ 2, 3, 2 });
    const out = try b.dot_general(lhs, rhs, .{
        .lhs_batch_dims = &.{0},
        .rhs_batch_dims = &.{0},
        .lhs_contracting_dims = &.{2},
        .rhs_contracting_dims = &.{1},
    });
    const func = try build_and_finish(&program, &b, &.{out});

    // Batch 0: [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]]
    var in_lhs = try HostTensor.init_with_data(testing.allocator, &.{ 2, 2, 3 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer in_lhs.deinit();
    var in_rhs = try HostTensor.init_with_data(testing.allocator, &.{ 2, 3, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer in_rhs.deinit();

    const results = try eval(testing.allocator, func, &.{ in_lhs, in_rhs });
    defer {
        for (results) |*r| r.deinit();
        testing.allocator.free(results);
    }
    try testing.expectEqual(@as(usize, 3), results[0].shape.len);
    try testing.expectEqual(@as(i64, 2), results[0].shape[0]);
    try testing.expectEqual(@as(i64, 2), results[0].shape[1]);
    try testing.expectEqual(@as(i64, 2), results[0].shape[2]);
    // Batch 0: [22, 28, 49, 64]
    try testing.expectApproxEqAbs(@as(f32, 22.0), results[0].data[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 28.0), results[0].data[1], 1e-4);
}

test "eval: reshape" {
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "reshape");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.reshape(x, &.{ 3, 2 });
    const func = try build_and_finish(&program, &b, &.{y});

    var in_x = try HostTensor.init_with_data(testing.allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer in_x.deinit();

    const results = try eval(testing.allocator, func, &.{in_x});
    defer {
        for (results) |*r| r.deinit();
        testing.allocator.free(results);
    }
    try testing.expectEqual(@as(usize, 2), results[0].shape.len);
    try testing.expectEqual(@as(i64, 3), results[0].shape[0]);
    try testing.expectEqual(@as(i64, 2), results[0].shape[1]);
    // Data unchanged
    try testing.expectApproxEqAbs(@as(f32, 1.0), results[0].data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 6.0), results[0].data[5], 1e-6);
}

test "eval: transpose 2D" {
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "transpose");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.transpose(x, &.{ 1, 0 });
    const func = try build_and_finish(&program, &b, &.{y});

    // [[1,2,3],[4,5,6]]
    var in_x = try HostTensor.init_with_data(testing.allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer in_x.deinit();

    const results = try eval(testing.allocator, func, &.{in_x});
    defer {
        for (results) |*r| r.deinit();
        testing.allocator.free(results);
    }
    // [[1,4],[2,5],[3,6]]
    try testing.expectEqual(@as(i64, 3), results[0].shape[0]);
    try testing.expectEqual(@as(i64, 2), results[0].shape[1]);
    try testing.expectApproxEqAbs(@as(f32, 1.0), results[0].data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 4.0), results[0].data[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2.0), results[0].data[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 5.0), results[0].data[3], 1e-6);
}

test "eval: broadcast_in_dim scalar to matrix" {
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "bcast");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{});
    const y = try b.broadcast_in_dim(x, &.{ 2, 3 }, &.{});
    const func = try build_and_finish(&program, &b, &.{y});

    var in_x = try HostTensor.init_with_data(testing.allocator, &.{}, &.{7.0});
    defer in_x.deinit();

    const results = try eval(testing.allocator, func, &.{in_x});
    defer {
        for (results) |*r| r.deinit();
        testing.allocator.free(results);
    }
    try testing.expectEqual(@as(usize, 6), results[0].data.len);
    for (results[0].data) |v| {
        try testing.expectApproxEqAbs(@as(f32, 7.0), v, 1e-6);
    }
}

test "eval: reduce_sum single axis" {
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "rsum");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.reduce_sum(x, &.{1});
    const func = try build_and_finish(&program, &b, &.{y});

    // [[1,2,3],[4,5,6]] -> sum over axis 1 -> [6, 15]
    var in_x = try HostTensor.init_with_data(testing.allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer in_x.deinit();

    const results = try eval(testing.allocator, func, &.{in_x});
    defer {
        for (results) |*r| r.deinit();
        testing.allocator.free(results);
    }
    try testing.expectEqual(@as(usize, 1), results[0].shape.len);
    try testing.expectEqual(@as(i64, 2), results[0].shape[0]);
    try testing.expectApproxEqAbs(@as(f32, 6.0), results[0].data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 15.0), results[0].data[1], 1e-6);
}

test "eval: gather basic" {
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "gather");
    defer b.deinit();
    // operand [5], indices [3,1] -> gather 3 elements
    const operand = try b.param_tensor(.f32, &.{5});
    const indices = try b.param_tensor(.i32, &.{ 3, 1 });
    const out = try b.gather(operand, indices, .{
        .slice_sizes = &.{1},
        .offset_dims = &.{},
        .collapsed_slice_dims = &.{0},
        .start_index_map = &.{0},
        .index_vector_dim = 1,
    });
    const func = try build_and_finish(&program, &b, &.{out});

    var in_op = try HostTensor.init_with_data(testing.allocator, &.{5}, &.{ 10, 20, 30, 40, 50 });
    defer in_op.deinit();
    var in_idx = try HostTensor.init_with_data(testing.allocator, &.{ 3, 1 }, &.{ 0, 2, 4 });
    defer in_idx.deinit();

    const results = try eval(testing.allocator, func, &.{ in_op, in_idx });
    defer {
        for (results) |*r| r.deinit();
        testing.allocator.free(results);
    }
    try testing.expectEqual(@as(usize, 3), results[0].data.len);
    try testing.expectApproxEqAbs(@as(f32, 10.0), results[0].data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 30.0), results[0].data[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 50.0), results[0].data[2], 1e-6);
}

test "eval: scatter add" {
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "scatter");
    defer b.deinit();
    const input = try b.param_tensor(.f32, &.{5});
    const indices = try b.param_tensor(.i32, &.{ 3, 1 });
    const updates = try b.param_tensor(.f32, &.{3});
    const out = try b.scatter(input, indices, updates, .{
        .update_window_dims = &.{},
        .inserted_window_dims = &.{0},
        .scatter_dims_to_operand_dims = &.{0},
        .index_vector_dim = 1,
    });
    const func = try build_and_finish(&program, &b, &.{out});

    var in_input = try HostTensor.init_with_data(testing.allocator, &.{5}, &.{ 0, 0, 0, 0, 0 });
    defer in_input.deinit();
    var in_idx = try HostTensor.init_with_data(testing.allocator, &.{ 3, 1 }, &.{ 1, 3, 1 });
    defer in_idx.deinit();
    var in_upd = try HostTensor.init_with_data(testing.allocator, &.{3}, &.{ 10, 20, 30 });
    defer in_upd.deinit();

    const results = try eval(testing.allocator, func, &.{ in_input, in_idx, in_upd });
    defer {
        for (results) |*r| r.deinit();
        testing.allocator.free(results);
    }
    // indices 1,3,1 with values 10,20,30: idx1 = 10+30=40, idx3 = 20
    try testing.expectApproxEqAbs(@as(f32, 0.0), results[0].data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 40.0), results[0].data[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), results[0].data[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 20.0), results[0].data[3], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), results[0].data[4], 1e-6);
}

test "eval: iota" {
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "iota");
    defer b.deinit();
    const y = try b.iota(.i32, &.{ 2, 3 }, 1);
    const func = try build_and_finish(&program, &b, &.{y});

    const results = try eval(testing.allocator, func, &.{});
    defer {
        for (results) |*r| r.deinit();
        testing.allocator.free(results);
    }
    // iota dim=1: [[0,1,2],[0,1,2]]
    try testing.expectApproxEqAbs(@as(f32, 0.0), results[0].data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), results[0].data[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2.0), results[0].data[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), results[0].data[3], 1e-6);
}

test "eval: compare eq" {
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "cmp");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{3});
    const y = try b.param_tensor(.f32, &.{3});
    const z = try b.compare(x, y, .{ .direction = .EQ, .compare_type = .FLOAT });
    const func = try build_and_finish(&program, &b, &.{z});

    var in_x = try HostTensor.init_with_data(testing.allocator, &.{3}, &.{ 1, 2, 3 });
    defer in_x.deinit();
    var in_y = try HostTensor.init_with_data(testing.allocator, &.{3}, &.{ 1, 5, 3 });
    defer in_y.deinit();

    const results = try eval(testing.allocator, func, &.{ in_x, in_y });
    defer {
        for (results) |*r| r.deinit();
        testing.allocator.free(results);
    }
    try testing.expectApproxEqAbs(@as(f32, 1.0), results[0].data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), results[0].data[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), results[0].data[2], 1e-6);
}

test "eval: select" {
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "sel");
    defer b.deinit();
    const cond = try b.param_tensor(.bool, &.{3});
    const on_true = try b.param_tensor(.f32, &.{3});
    const on_false = try b.param_tensor(.f32, &.{3});
    const out = try b.select(cond, on_true, on_false);
    const func = try build_and_finish(&program, &b, &.{out});

    var in_cond = try HostTensor.init_with_data(testing.allocator, &.{3}, &.{ 1, 0, 1 });
    defer in_cond.deinit();
    var in_true = try HostTensor.init_with_data(testing.allocator, &.{3}, &.{ 10, 20, 30 });
    defer in_true.deinit();
    var in_false = try HostTensor.init_with_data(testing.allocator, &.{3}, &.{ 100, 200, 300 });
    defer in_false.deinit();

    const results = try eval(testing.allocator, func, &.{ in_cond, in_true, in_false });
    defer {
        for (results) |*r| r.deinit();
        testing.allocator.free(results);
    }
    try testing.expectApproxEqAbs(@as(f32, 10.0), results[0].data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 200.0), results[0].data[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 30.0), results[0].data[2], 1e-6);
}

test "eval: slice" {
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "slc");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 4, 4 });
    const y = try b.slice(x, .{
        .start_indices = &.{ 1, 1 },
        .limit_indices = &.{ 3, 3 },
        .strides = &.{ 1, 1 },
    });
    const func = try build_and_finish(&program, &b, &.{y});

    // 4x4 matrix: 0..15
    var data: [16]f32 = undefined;
    for (&data, 0..) |*v, i| v.* = @floatFromInt(i);
    var in_x = try HostTensor.init_with_data(testing.allocator, &.{ 4, 4 }, &data);
    defer in_x.deinit();

    const results = try eval(testing.allocator, func, &.{in_x});
    defer {
        for (results) |*r| r.deinit();
        testing.allocator.free(results);
    }
    try testing.expectEqual(@as(i64, 2), results[0].shape[0]);
    try testing.expectEqual(@as(i64, 2), results[0].shape[1]);
    // [1,1]=5, [1,2]=6, [2,1]=9, [2,2]=10
    try testing.expectApproxEqAbs(@as(f32, 5.0), results[0].data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 6.0), results[0].data[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 9.0), results[0].data[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 10.0), results[0].data[3], 1e-6);
}

test "eval: concatenate" {
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "cat");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 2 });
    const y = try b.param_tensor(.f32, &.{ 2, 3 });
    const z = try b.concatenate(&.{ x, y }, 1);
    const func = try build_and_finish(&program, &b, &.{z});

    var in_x = try HostTensor.init_with_data(testing.allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer in_x.deinit();
    var in_y = try HostTensor.init_with_data(testing.allocator, &.{ 2, 3 }, &.{ 5, 6, 7, 8, 9, 10 });
    defer in_y.deinit();

    const results = try eval(testing.allocator, func, &.{ in_x, in_y });
    defer {
        for (results) |*r| r.deinit();
        testing.allocator.free(results);
    }
    try testing.expectEqual(@as(i64, 2), results[0].shape[0]);
    try testing.expectEqual(@as(i64, 5), results[0].shape[1]);
    // Row 0: [1,2,5,6,7], Row 1: [3,4,8,9,10]
    try testing.expectApproxEqAbs(@as(f32, 1.0), results[0].data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2.0), results[0].data[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 5.0), results[0].data[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 10.0), results[0].data[9], 1e-6);
}
