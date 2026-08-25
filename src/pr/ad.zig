//! Automatic differentiation transforms on PR functions.
//!
//! Linearization produces an augmented primal and a residualized linear
//!  function. JVP applies that linear function. VJP transposes it.
//!
//! JVP and VJP register the complete derived call graph in the supplied
//!  program and return the derived callable's identity.
//! Higher-level traced transforms live in `transforms.zig`.
const std = @import("std");

const pr = @import("pr.zig");
const ops = @import("ops/ops.zig");

pub const AdError = ops.types.AdError || pr.FunctionRegistrationError || error{
    /// PR differentiation does not transform recursive call graphs.
    RecursiveDifferentiationUnsupported,
};

/// Options for `vjp` and `vjp_with_value`.
pub const VjpOpts = struct {
    /// Select source-function outputs whose cotangents seed the VJP.
    ///
    /// `null` selects every output. A provided slice must be nonempty. The
    ///  transformed function accepts seeds in this slice's order. Duplicate
    ///  indices contribute independent seeds to the same output cotangent.
    of: ?[]const usize = null,

    /// Select primal-input cotangents returned by the transformed function.
    ///
    /// `null` selects every input. The output order matches this slice, including
    ///  duplicate indices.
    wrt: ?[]const usize = null,
};

/// Functions and output mapping produced by linearizing one PR function.
///
/// Function identities and the tangent-index map have the program's lifetime.
pub const Linearization = struct {
    /// Accepts source parameters and returns source results followed by residuals.
    augmented_primal: pr.FunctionId,
    /// Accepts source-parameter tangents followed by residuals.
    linear: pr.FunctionId,
    /// Number of augmented-primal results consumed as linear-function residuals.
    residual_count: usize,
    /// Maps each source result to its linear-function tangent result.
    ///
    /// `null` means the transform emitted no tangent value for that source
    ///  result. Callers materialize zero when they require full output arity.
    output_tangent_indices: []const ?usize,
};

const ResidualCandidate = struct {
    primal: *pr.Var,
    linear_param: *pr.Var,
};

const GeneratedLinearization = struct {
    source: pr.FunctionId,
    linearization: Linearization,
};

fn pop_active(active: *std.ArrayList(pr.FunctionId), expected: pr.FunctionId) void {
    const source = active.pop() orelse unreachable;
    std.debug.assert(source == expected);
}

const LinearizationTraversal = struct {
    allocator: std.mem.Allocator,
    program: *pr.Program,
    generated: std.ArrayList(GeneratedLinearization) = .empty,
    active: std.ArrayList(pr.FunctionId) = .empty,

    fn deinit(self: *LinearizationTraversal) void {
        self.generated.deinit(self.allocator);
        self.active.deinit(self.allocator);
        self.* = undefined;
    }

    fn get_or_create(self: *LinearizationTraversal, source: pr.FunctionId) AdError!Linearization {
        for (self.generated.items) |entry| {
            if (entry.source == source) return entry.linearization;
        }
        for (self.active.items) |active| {
            if (active == source) return error.RecursiveDifferentiationUnsupported;
        }

        // The active stack rejects direct and mutually recursive call graphs.
        try self.active.append(self.allocator, source);
        defer pop_active(&self.active, source);

        const func = self.program.get_function_by_id(source) orelse
            return error.CallUnresolvedCallee;
        const base_name = try std.fmt.allocPrint(self.allocator, "{s}_linearized", .{func.name});
        defer self.allocator.free(base_name);
        const result = try linearize_impl(self, func, base_name);
        try self.generated.append(self.allocator, .{
            .source = source,
            .linearization = result,
        });
        return result;
    }
};

/// Linearize `func` into an augmented primal and a residualized linear map.
///
/// The generated functions are registered in `program`. Their names are
///  derived from `name` and made unique within the program. Generated
///  functions for callees are shared within this traversal.
pub fn linearize(
    allocator: std.mem.Allocator,
    program: *pr.Program,
    source: pr.FunctionId,
    name: []const u8,
) AdError!Linearization {
    var traversal = LinearizationTraversal{
        .allocator = allocator,
        .program = program,
    };
    defer traversal.deinit();

    const func = program.get_function_by_id(source) orelse
        return error.CallUnresolvedCallee;
    try traversal.active.append(allocator, source);
    defer pop_active(&traversal.active, source);
    return try linearize_impl(&traversal, func, name);
}

fn linearize_impl(
    traversal: *LinearizationTraversal,
    func: pr.Function,
    name: []const u8,
) AdError!Linearization {
    const allocator = traversal.allocator;
    const program = traversal.program;
    const saved = program.checkpoint_appends();
    errdefer program.restore_appends(saved);

    const primal_name_base = try std.fmt.allocPrint(allocator, "{s}_primal", .{name});
    defer allocator.free(primal_name_base);
    const primal_name = try program.reserve_unique_function_name(primal_name_base);
    const linear_name_base = try std.fmt.allocPrint(allocator, "{s}_linear", .{name});
    defer allocator.free(linear_name_base);
    const linear_name = try program.reserve_unique_function_name(linear_name_base);

    var primal_builder = try pr.FunctionBuilder.init(program, primal_name);
    defer primal_builder.deinit();
    var linear_builder = try pr.FunctionBuilder.init(program, linear_name);
    defer linear_builder.deinit();

    // Each source value may have a primal value in the augmented function,
    //  a residual parameter in the linear function, and a tangent value.
    const primal_map = try allocator.alloc(?*pr.Var, func.var_count);
    defer allocator.free(primal_map);
    @memset(primal_map, null);
    const linear_primal_map = try allocator.alloc(?*pr.Var, func.var_count);
    defer allocator.free(linear_primal_map);
    @memset(linear_primal_map, null);
    const tangent_map = try allocator.alloc(?*pr.Var, func.var_count);
    defer allocator.free(tangent_map);
    @memset(tangent_map, null);

    var residuals: std.ArrayList(ResidualCandidate) = .empty;
    defer residuals.deinit(allocator);
    const tangent_params = try allocator.alloc(*pr.Var, func.params.len);
    defer allocator.free(tangent_params);

    for (func.params, 0..) |source_param, index| {
        const primal_param = try primal_builder.param_like(source_param.aval);
        primal_map[source_param.id] = primal_param;

        const tangent_param = try linear_builder.param_like(source_param.aval);
        tangent_params[index] = tangent_param;
        tangent_map[source_param.id] = tangent_param;

        const residual_param = try linear_builder.param_like(source_param.aval);
        linear_primal_map[source_param.id] = residual_param;
        try residuals.append(allocator, .{
            .primal = primal_param,
            .linear_param = residual_param,
        });
    }

    const primal_ctx = ops.types.AdContext{
        .builder = &primal_builder,
        .primal_map = primal_map,
        .cot_map = null,
        .tangent_map = null,
        .allocator = allocator,
    };
    const linear_ctx = ops.types.AdContext{
        .builder = &linear_builder,
        .primal_map = linear_primal_map,
        .cot_map = null,
        .tangent_map = tangent_map,
        .allocator = allocator,
    };

    for (func.ops) |op| {
        switch (op.params) {
            .call => try linearize_call(
                traversal,
                primal_ctx,
                linear_ctx,
                &residuals,
                op,
            ),
            else => {
                try replay_primal(primal_ctx, op);
                for (op.outputs) |source_output| {
                    const primal = primal_ctx.get_primal(source_output) orelse
                        return error.UnsupportedEqn;
                    const residual_param = try linear_builder.param_like(source_output.aval);
                    linear_ctx.set_primal(source_output, residual_param);
                    try residuals.append(allocator, .{
                        .primal = primal,
                        .linear_param = residual_param,
                    });
                }
                try ops.jvp(linear_ctx, op);
            },
        }
    }

    // The linear function only returns tangents that have PR values. Keep the
    //  source-result mapping so wrappers can restore full output arity.
    const output_tangent_indices = try program.allocator().alloc(?usize, func.returns.len);
    const linear_returns = try allocator.alloc(*pr.Var, func.returns.len);
    defer allocator.free(linear_returns);
    var linear_return_count: usize = 0;
    for (func.returns, output_tangent_indices) |source_return, *tangent_index| {
        if (linear_ctx.get_tangent(source_return)) |tangent| {
            tangent_index.* = linear_return_count;
            linear_returns[linear_return_count] = tangent;
            linear_return_count += 1;
        } else {
            tangent_index.* = null;
        }
    }

    var linear_func = try linear_builder.finish(linear_returns[0..linear_return_count]);

    // Residual parameters are provisional. Their use lists identify the
    //  primal values the completed linear function actually needs.
    var used_residual_count: usize = 0;
    for (residuals.items) |candidate| {
        if (candidate.linear_param.first_use != null) used_residual_count += 1;
    }
    const linear_params = try program.allocator().alloc(
        *pr.Var,
        tangent_params.len + used_residual_count,
    );
    @memcpy(linear_params[0..tangent_params.len], tangent_params);
    var residual_index: usize = 0;
    for (residuals.items) |candidate| {
        if (candidate.linear_param.first_use == null) continue;
        linear_params[tangent_params.len + residual_index] = candidate.linear_param;
        residual_index += 1;
    }
    std.debug.assert(residual_index == used_residual_count);
    linear_func.params = linear_params;
    reindex_vars(&linear_func);

    const primal_returns = try allocator.alloc(
        *pr.Var,
        func.returns.len + used_residual_count,
    );
    defer allocator.free(primal_returns);
    for (func.returns, primal_returns[0..func.returns.len]) |source_return, *output| {
        output.* = primal_ctx.get_primal(source_return) orelse
            return error.UnsupportedEqn;
    }
    residual_index = 0;
    for (residuals.items) |candidate| {
        if (candidate.linear_param.first_use == null) continue;
        primal_returns[func.returns.len + residual_index] = candidate.primal;
        residual_index += 1;
    }
    std.debug.assert(residual_index == used_residual_count);

    const primal_func = try primal_builder.finish(primal_returns);
    const primal_id = try program.add_function(primal_func);
    const linear_id = try program.add_function(linear_func);
    return .{
        .augmented_primal = primal_id,
        .linear = linear_id,
        .residual_count = used_residual_count,
        .output_tangent_indices = output_tangent_indices,
    };
}

fn replay_primal(ctx: ops.types.AdContext, op: *const pr.Op) AdError!void {
    const inputs = try ctx.allocator.alloc(*pr.Var, op.inputs.len);
    defer ctx.allocator.free(inputs);
    for (op.inputs, inputs) |operand, *input| {
        input.* = ctx.get_primal(operand.value) orelse return error.UnsupportedEqn;
    }

    const replayed = try ctx.builder.replay_op(op, inputs);
    if (replayed.outputs.len != op.outputs.len) return error.UnsupportedEqn;
    for (op.outputs, replayed.outputs) |source, output| ctx.set_primal(source, output);
}

/// Assign dense function-local SSA ids after removing provisional parameters.
fn reindex_vars(func: *pr.Function) void {
    var next_id: u32 = 0;
    for (func.params) |param| {
        param.id = next_id;
        next_id += 1;
    }
    for (func.ops) |op| {
        for (op.outputs) |output| {
            output.id = next_id;
            next_id += 1;
        }
    }
    std.debug.assert(next_id <= func.var_count);
    func.var_count = next_id;
}

fn linearize_call(
    traversal: *LinearizationTraversal,
    primal_ctx: ops.types.AdContext,
    linear_ctx: ops.types.AdContext,
    residuals: *std.ArrayList(ResidualCandidate),
    op: *const pr.Op,
) AdError!void {
    const callee_id = op.params.call.callee;
    const callee = traversal.program.get_function_by_id(callee_id) orelse
        return error.CallUnresolvedCallee;
    const callee_linearization = try traversal.get_or_create(callee_id);

    const primal_inputs = try traversal.allocator.alloc(*pr.Var, op.inputs.len);
    defer traversal.allocator.free(primal_inputs);
    for (op.inputs, primal_inputs) |operand, *input| {
        input.* = primal_ctx.get_primal(operand.value) orelse
            return error.UnsupportedEqn;
    }
    const primal_outputs = (try primal_ctx.builder.call(
        callee_linearization.augmented_primal,
        primal_inputs,
    )).outputs;
    if (primal_outputs.len != callee.returns.len + callee_linearization.residual_count)
        return error.UnsupportedEqn;
    for (op.outputs, primal_outputs[0..callee.returns.len]) |source_output, output| {
        primal_ctx.set_primal(source_output, output);

        // A caller operation may use the call result as a JVP coefficient.
        //  Unused result residuals are removed with the other candidates.
        const residual_param = try linear_ctx.builder.param_like(source_output.aval);
        linear_ctx.set_primal(source_output, residual_param);
        try residuals.append(traversal.allocator, .{
            .primal = output,
            .linear_param = residual_param,
        });
    }

    // The augmented callee returns ordinary results followed by residuals.
    //  Its linear callee consumes operand tangents followed by those residuals.
    const linear_inputs = try traversal.allocator.alloc(
        *pr.Var,
        op.inputs.len + callee_linearization.residual_count,
    );
    defer traversal.allocator.free(linear_inputs);
    for (op.inputs, linear_inputs[0..op.inputs.len]) |operand, *input| {
        input.* = linear_ctx.get_tangent(operand.value) orelse zero: {
            const tensor = operand.value.as_tensor();
            break :zero try linear_ctx.builder.scalar_broadcast(
                tensor.dtype,
                tensor.shape.dims,
                0.0,
            );
        };
    }
    for (
        primal_outputs[callee.returns.len..],
        linear_inputs[op.inputs.len..],
    ) |primal_residual, *linear_input| {
        const residual_param = try linear_ctx.builder.param_like(primal_residual.aval);
        linear_input.* = residual_param;
        try residuals.append(traversal.allocator, .{
            .primal = primal_residual,
            .linear_param = residual_param,
        });
    }

    const tangent_outputs = (try linear_ctx.builder.call(
        callee_linearization.linear,
        linear_inputs,
    )).outputs;
    for (op.outputs, callee_linearization.output_tangent_indices) |source_output, tangent_index| {
        const index = tangent_index orelse continue;
        if (index >= tangent_outputs.len) return error.UnsupportedEqn;
        linear_ctx.set_tangent(source_output, tangent_outputs[index]);
    }
}

/// Controls whether a derivative function also returns primal outputs.
const PrimalOutputs = enum { skip, emit };

const GeneratedTranspose = struct {
    callee: pr.FunctionId,
    selected_outputs: []const usize,
    function_id: pr.FunctionId,
};

const TransposeTraversal = struct {
    allocator: std.mem.Allocator,
    program: *pr.Program,
    generated: std.ArrayList(GeneratedTranspose) = .empty,
    active: std.ArrayList(pr.FunctionId) = .empty,

    fn deinit(self: *TransposeTraversal) void {
        for (self.generated.items) |entry| {
            self.allocator.free(entry.selected_outputs);
        }
        self.generated.deinit(self.allocator);
        self.active.deinit(self.allocator);
        self.* = undefined;
    }

    fn get_or_create(
        self: *TransposeTraversal,
        callee_id: pr.FunctionId,
        selected_outputs: []const usize,
    ) AdError!pr.FunctionId {
        for (self.generated.items) |existing| {
            if (existing.callee != callee_id) continue;
            if (!std.mem.eql(usize, existing.selected_outputs, selected_outputs)) continue;
            return existing.function_id;
        }
        for (self.active.items) |active| {
            if (active == callee_id) return error.RecursiveDifferentiationUnsupported;
        }

        // The active stack rejects direct and mutually recursive call graphs.
        try self.active.append(self.allocator, callee_id);
        defer pop_active(&self.active, callee_id);

        const callee = self.program.get_function_by_id(callee_id) orelse
            return error.CallUnresolvedCallee;
        const base_name = try std.fmt.allocPrint(self.allocator, "{s}_transpose", .{callee.name});
        defer self.allocator.free(base_name);
        const transpose_name = try self.program.reserve_unique_function_name(base_name);
        const transpose = try transpose_impl(
            self,
            callee,
            transpose_name,
            null,
            selected_outputs,
        );
        const transpose_id = try self.program.add_function(transpose);

        const owned_outputs = try self.allocator.dupe(usize, selected_outputs);
        errdefer self.allocator.free(owned_outputs);
        try self.generated.append(self.allocator, .{
            .callee = callee_id,
            .selected_outputs = owned_outputs,
            .function_id = transpose_id,
        });
        return transpose_id;
    }
};

fn transpose_linear(
    allocator: std.mem.Allocator,
    program: *pr.Program,
    function_id: pr.FunctionId,
    name: []const u8,
    wrt: ?[]const usize,
    of: []const usize,
) AdError!pr.Function {
    var traversal = TransposeTraversal{ .allocator = allocator, .program = program };
    defer traversal.deinit();
    const func = program.get_function_by_id(function_id) orelse
        return error.CallUnresolvedCallee;
    try traversal.active.append(allocator, function_id);
    defer pop_active(&traversal.active, function_id);
    return try transpose_impl(&traversal, func, name, wrt, of);
}

/// Reports whether PR defines dual values for a dtype.
fn is_differentiable_dtype(dtype: pr.DType) bool {
    return switch (dtype) {
        .f32, .f64, .bf16, .f16 => true,
        else => false,
    };
}

/// Transpose a linear function by propagating output cotangents in reverse.
///
/// `of` restricts cotangent seeds and `wrt` restricts harvested input
///  cotangents. The transform replays primal operations required by the
///  registered transpose rules.
///
/// Missing rules on active differentiable paths return `AdError.UnsupportedEqn`.
/// An input outside the selected output's dependency path has a zero
///  cotangent.
/// TODO(ad): Make the built-in differentiable dtype set an explicit AD policy.
fn transpose_impl(
    traversal: *TransposeTraversal,
    func: pr.Function,
    name: []const u8,
    wrt: ?[]const usize,
    of: []const usize,
) AdError!pr.Function {
    const allocator = traversal.allocator;
    const program = traversal.program;
    const saved = program.checkpoint_appends();
    errdefer program.restore_appends(saved);

    var primal_map = try allocator.alloc(?*pr.Var, func.var_count);
    defer allocator.free(primal_map);
    @memset(primal_map, null);

    var cotangent_map = try allocator.alloc(?*pr.Var, func.var_count);
    defer allocator.free(cotangent_map);
    @memset(cotangent_map, null);

    var b = try pr.FunctionBuilder.init(program, name);
    defer b.deinit();

    for (func.params) |param_var| {
        const new_param = try b.param_like(param_var.aval);
        primal_map[param_var.id] = new_param;
    }

    if (of.len == 0) return AdError.EmptyOutputSelection;
    for (of) |index| {
        if (index >= func.returns.len) return AdError.OfIndexOutOfRange;
    }

    for (of) |output_index| {
        const v = func.returns[output_index];
        const tensor = v.aval.as_tensor();
        if (!is_differentiable_dtype(tensor.dtype)) return AdError.UnsupportedDType;
        const new_seed = try b.param_like(v.aval);
        if (cotangent_map[v.id] != null) {
            cotangent_map[v.id] = try b.add(cotangent_map[v.id].?, new_seed);
        } else {
            cotangent_map[v.id] = new_seed;
        }
    }

    const ad_ctx = ops.types.AdContext{
        .builder = &b,
        .primal_map = primal_map,
        .cot_map = cotangent_map,
        .tangent_map = null,
        .allocator = allocator,
    };

    // Transpose rules may need primal values, so replay the linear function
    //  before propagating its selected output cotangents in reverse.
    for (func.ops) |op| try replay_primal(ad_ctx, op);
    var op_index: usize = func.ops.len;
    while (op_index > 0) {
        op_index -= 1;
        switch (func.ops[op_index].params) {
            .call => try transpose_call(
                traversal,
                ad_ctx,
                func.ops[op_index],
            ),
            else => try ops.vjp(ad_ctx, func.ops[op_index]),
        }
    }

    if (wrt) |indices| {
        for (indices) |idx| {
            if (idx >= func.params.len) return AdError.WrtIndexOutOfRange;
        }
    }
    const return_count: usize = if (wrt) |indices| indices.len else func.params.len;

    const returns = try allocator.alloc(*pr.Var, return_count);
    defer allocator.free(returns);

    for (0..return_count) |i| {
        const v = if (wrt) |indices| func.params[indices[i]] else func.params[i];
        if (cotangent_map[v.id]) |cotangent| {
            returns[i] = cotangent;
        } else {
            const t = v.aval.as_tensor();
            returns[i] = try b.scalar_broadcast(t.dtype, t.shape.dims, 0.0);
        }
    }

    return try b.finish(returns);
}

fn transpose_call(
    traversal: *TransposeTraversal,
    ctx: ops.types.AdContext,
    op: *const pr.Op,
) AdError!void {
    const selected_outputs = try ctx.allocator.alloc(usize, op.outputs.len);
    defer ctx.allocator.free(selected_outputs);
    var selected_count: usize = 0;
    for (op.outputs, 0..) |output, index| {
        if (ctx.get_cot(output) == null) continue;
        selected_outputs[selected_count] = index;
        selected_count += 1;
    }
    if (selected_count == 0) return;

    const transpose_id = try traversal.get_or_create(
        op.params.call.callee,
        selected_outputs[0..selected_count],
    );

    const call_inputs = try ctx.allocator.alloc(*pr.Var, op.inputs.len + selected_count);
    defer ctx.allocator.free(call_inputs);
    for (op.inputs, call_inputs[0..op.inputs.len]) |operand, *input| {
        input.* = ctx.get_primal(operand.value) orelse return error.UnsupportedEqn;
    }
    for (selected_outputs[0..selected_count], call_inputs[op.inputs.len..]) |index, *input| {
        input.* = ctx.get_cot(op.outputs[index]).?;
    }

    const input_cotangents = (try ctx.builder.call(transpose_id, call_inputs)).outputs;
    if (input_cotangents.len != op.inputs.len) return error.UnsupportedEqn;
    for (op.inputs, input_cotangents) |operand, cotangent| {
        try ctx.add_cot(operand.value, cotangent);
    }
}

/// Reverse-mode AD applies the pullback of \(f: M \to N\):
///
/// $$
/// \operatorname{vjp}_f:
/// (x, v) \in M \times T^*_{f(x)}N
/// \mapsto \mathrm{d}f_x^*(v) \in T_x^*M.
/// $$
///
/// For a cotangent seed \(v\), it computes the transpose-Jacobian product
/// \(J^\mathsf{T}(x) \cdot v\), the pullback
/// \(f^*: T^*_{f(x)} N \to T^*_x M\) evaluated at \(x\).
///
/// For
///
/// $$
/// f: X_1 \times \cdots \times X_n \to Y_1 \times \cdots \times Y_m,
/// $$
///
/// let \(O = (o_1, \ldots, o_s)\) be `opts.of` and
/// \(W = (w_1, \ldots, w_k)\) be `opts.wrt`. The transformed function has
/// signature
///
/// $$
/// \operatorname{vjp}^{O,W}_f:
/// (X_1 \times \cdots \times X_n)
/// \times (T^*_{f_{o_1}(x)}Y_{o_1} \times \cdots
/// \times T^*_{f_{o_s}(x)}Y_{o_s})
/// \to T^*_{x_{w_1}}X_{w_1} \times \cdots
/// \times T^*_{x_{w_k}}X_{w_k}.
/// $$
///
/// `null` expands \(O\) or \(W\) to every corresponding index. In the
///  Euclidean or Cartesian case (\(G = I\)), these equal gradients. In general,
///  they are covectors and must be raised with \(G^{-1}\) to obtain gradient
///  tangent vectors.
pub fn vjp(
    allocator: std.mem.Allocator,
    program: *pr.Program,
    source: pr.FunctionId,
    name: []const u8,
    /// See `VjpOpts` for seed and result selection. Default `.{}` seeds every
    ///  source output and returns every parameter cotangent.
    opts: VjpOpts,
) AdError!pr.FunctionId {
    return try apply_transposed_linearization(
        allocator,
        program,
        source,
        name,
        .skip,
        opts,
    );
}

/// Apply VJP and emit primal outputs before input cotangents.
///
/// Using the notation from `vjp`, this changes the codomain to
///
/// $$
/// (Y_1 \times \cdots \times Y_m)
/// \times (T^*_{x_{w_1}}X_{w_1} \times \cdots
/// \times T^*_{x_{w_k}}X_{w_k}).
/// $$
///
/// See `vjp`
pub fn vjp_with_value(
    allocator: std.mem.Allocator,
    program: *pr.Program,
    source: pr.FunctionId,
    name: []const u8,
    opts: VjpOpts,
) AdError!pr.FunctionId {
    return try apply_transposed_linearization(
        allocator,
        program,
        source,
        name,
        .emit,
        opts,
    );
}

fn apply_transposed_linearization(
    allocator: std.mem.Allocator,
    program: *pr.Program,
    source: pr.FunctionId,
    name: []const u8,
    primals: PrimalOutputs,
    opts: VjpOpts,
) AdError!pr.FunctionId {
    const saved = program.checkpoint_appends();
    errdefer program.restore_appends(saved);
    const func = program.get_function_by_id(source) orelse
        return error.CallUnresolvedCallee;
    if (opts.of) |indices| {
        if (indices.len == 0) return error.EmptyOutputSelection;
        for (indices) |index| {
            if (index >= func.returns.len) return error.OfIndexOutOfRange;
            if (!is_differentiable_dtype(func.returns[index].as_tensor().dtype))
                return error.UnsupportedDType;
        }
    } else {
        for (func.returns) |source_output| {
            if (!is_differentiable_dtype(source_output.as_tensor().dtype))
                return error.UnsupportedDType;
        }
    }
    if (opts.wrt) |indices| {
        for (indices) |index| {
            if (index >= func.params.len) return error.WrtIndexOutOfRange;
        }
    }

    const result = try linearize(allocator, program, source, name);
    const augmented_primal = program.get_function_by_id(result.augmented_primal) orelse
        return error.CallUnresolvedCallee;
    const linear = program.get_function_by_id(result.linear) orelse
        return error.CallUnresolvedCallee;

    const selected_count = if (opts.of) |indices| indices.len else func.returns.len;
    const selected_linear_outputs = try allocator.alloc(usize, selected_count);
    defer allocator.free(selected_linear_outputs);
    var selected_linear_count: usize = 0;
    for (0..selected_count) |selection_index| {
        const source_index = if (opts.of) |indices| indices[selection_index] else selection_index;
        const linear_index = result.output_tangent_indices[source_index] orelse continue;
        selected_linear_outputs[selected_linear_count] = linear_index;
        selected_linear_count += 1;
    }

    var transpose_id: ?pr.FunctionId = null;
    if (selected_linear_count != 0) {
        const tangent_wrt_count = if (opts.wrt) |indices| indices.len else func.params.len;
        const tangent_wrt = try allocator.alloc(usize, tangent_wrt_count);
        defer allocator.free(tangent_wrt);
        for (0..tangent_wrt_count) |index| {
            tangent_wrt[index] = if (opts.wrt) |indices| indices[index] else index;
        }

        const transpose_name_base = try std.fmt.allocPrint(allocator, "{s}_transpose", .{name});
        defer allocator.free(transpose_name_base);
        const transpose_name = try program.reserve_unique_function_name(transpose_name_base);
        const transpose_func = try transpose_linear(
            allocator,
            program,
            result.linear,
            transpose_name,
            tangent_wrt,
            selected_linear_outputs[0..selected_linear_count],
        );
        transpose_id = try program.add_function(transpose_func);
    }

    var builder = try pr.FunctionBuilder.init(program, name);
    defer builder.deinit();
    const params = try allocator.alloc(*pr.Var, func.params.len);
    defer allocator.free(params);
    for (func.params, params) |source_param, *param| {
        param.* = try builder.param_like(source_param.aval);
    }
    const seeds = try allocator.alloc(*pr.Var, selected_count);
    defer allocator.free(seeds);
    for (0..selected_count) |selection_index| {
        const source_index = if (opts.of) |indices| indices[selection_index] else selection_index;
        seeds[selection_index] = try builder.param_like(func.returns[source_index].aval);
    }

    const primal_outputs = (try builder.call(result.augmented_primal, params)).outputs;
    if (primal_outputs.len != augmented_primal.returns.len or
        primal_outputs.len != func.returns.len + result.residual_count)
        return error.UnsupportedEqn;

    const gradient_count = if (opts.wrt) |indices| indices.len else func.params.len;
    const gradients = try allocator.alloc(*pr.Var, gradient_count);
    defer allocator.free(gradients);
    if (transpose_id) |callee| {
        const transpose = program.get_function_by_id(callee) orelse
            return error.CallUnresolvedCallee;
        const transpose_inputs = try allocator.alloc(*pr.Var, transpose.params.len);
        defer allocator.free(transpose_inputs);
        if (transpose_inputs.len != linear.params.len + selected_linear_count)
            return error.UnsupportedEqn;

        // The first linear inputs are tangent variables. Their primal values
        //  are zero because only residual coefficients affect the transpose.
        for (func.params, transpose_inputs[0..func.params.len]) |source_param, *input| {
            const tensor = source_param.as_tensor();
            input.* = try builder.scalar_broadcast(tensor.dtype, tensor.shape.dims, 0.0);
        }
        @memcpy(
            transpose_inputs[func.params.len..linear.params.len],
            primal_outputs[func.returns.len..],
        );
        var seed_index: usize = 0;
        for (0..selected_count) |selection_index| {
            const source_index = if (opts.of) |indices| indices[selection_index] else selection_index;
            if (result.output_tangent_indices[source_index] == null) continue;
            transpose_inputs[linear.params.len + seed_index] = seeds[selection_index];
            seed_index += 1;
        }
        std.debug.assert(seed_index == selected_linear_count);
        const transpose_outputs = (try builder.call(callee, transpose_inputs)).outputs;
        if (transpose_outputs.len != gradients.len) return error.UnsupportedEqn;
        @memcpy(gradients, transpose_outputs);
    } else {
        for (0..gradient_count) |gradient_index| {
            const source_index = if (opts.wrt) |indices| indices[gradient_index] else gradient_index;
            const tensor = func.params[source_index].as_tensor();
            gradients[gradient_index] = try builder.scalar_broadcast(
                tensor.dtype,
                tensor.shape.dims,
                0.0,
            );
        }
    }

    const primal_count: usize = if (primals == .emit) func.returns.len else 0;
    const outputs = try allocator.alloc(*pr.Var, primal_count + gradients.len);
    defer allocator.free(outputs);
    if (primals == .emit) {
        @memcpy(outputs[0..primal_count], primal_outputs[0..primal_count]);
    }
    @memcpy(outputs[primal_count..], gradients);
    return try program.add_function(try builder.finish(outputs));
}

/// Forward-mode AD applies the differential of \(f: M \to N\):
///
/// $$
/// \operatorname{jvp}_f:
/// (x, v) \in M \times T_xM
/// \mapsto \mathrm{d}f_x(v) \in T_{f(x)}N.
/// $$
///
/// For a tangent seed \(v\), it computes the Jacobian-vector product
/// \(J(x) \cdot v\), the differential
/// \(\mathrm{d}f_x: T_x M \to T_{f(x)} N\) applied to \(v\).
///
/// For
///
/// $$
/// f: X_1 \times \cdots \times X_n \to Y_1 \times \cdots \times Y_m,
/// $$
///
/// the transformed function has signature
///
/// $$
/// \operatorname{jvp}_f:
/// (X_1 \times \cdots \times X_n)
/// \times (T_{x_1}X_1 \times \cdots \times T_{x_n}X_n)
/// \to T_{f_1(x)}Y_1 \times \cdots \times T_{f_m(x)}Y_m.
/// $$
pub fn jvp(allocator: std.mem.Allocator, program: *pr.Program, source: pr.FunctionId, name: []const u8) AdError!pr.FunctionId {
    return try apply_linearization(allocator, program, source, name, .skip);
}

/// Apply JVP and emit primal outputs before output tangents.
///
/// Using the notation from `jvp`, this changes the codomain to
///
/// $$
/// (Y_1 \times \cdots \times Y_m)
/// \times (T_{f_1(x)}Y_1 \times \cdots \times T_{f_m(x)}Y_m).
/// $$
///
/// See `jvp`
pub fn jvp_with_value(allocator: std.mem.Allocator, program: *pr.Program, source: pr.FunctionId, name: []const u8) AdError!pr.FunctionId {
    return try apply_linearization(allocator, program, source, name, .emit);
}

fn apply_linearization(
    allocator: std.mem.Allocator,
    program: *pr.Program,
    source: pr.FunctionId,
    name: []const u8,
    primals: PrimalOutputs,
) AdError!pr.FunctionId {
    const saved = program.checkpoint_appends();
    errdefer program.restore_appends(saved);
    const func = program.get_function_by_id(source) orelse
        return error.CallUnresolvedCallee;
    const result = try linearize(allocator, program, source, name);
    const augmented_primal = program.get_function_by_id(result.augmented_primal) orelse
        return error.CallUnresolvedCallee;
    const linear = program.get_function_by_id(result.linear) orelse
        return error.CallUnresolvedCallee;

    var builder = try pr.FunctionBuilder.init(program, name);
    defer builder.deinit();

    const params = try allocator.alloc(*pr.Var, func.params.len);
    defer allocator.free(params);
    const tangents = try allocator.alloc(*pr.Var, func.params.len);
    defer allocator.free(tangents);
    for (func.params, params) |source_param, *param| {
        param.* = try builder.param_like(source_param.aval);
    }
    for (func.params, tangents) |source_param, *tangent| {
        tangent.* = try builder.param_like(source_param.aval);
    }

    const primal_outputs = (try builder.call(result.augmented_primal, params)).outputs;
    if (primal_outputs.len != augmented_primal.returns.len or
        primal_outputs.len != func.returns.len + result.residual_count)
        return error.UnsupportedEqn;

    const linear_inputs = try allocator.alloc(*pr.Var, linear.params.len);
    defer allocator.free(linear_inputs);
    if (linear_inputs.len != tangents.len + result.residual_count)
        return error.UnsupportedEqn;
    @memcpy(linear_inputs[0..tangents.len], tangents);
    @memcpy(linear_inputs[tangents.len..], primal_outputs[func.returns.len..]);
    const linear_outputs = (try builder.call(result.linear, linear_inputs)).outputs;
    if (linear_outputs.len != linear.returns.len) return error.UnsupportedEqn;

    const output_count = switch (primals) {
        .emit => func.returns.len * 2,
        .skip => func.returns.len,
    };
    const outputs = try allocator.alloc(*pr.Var, output_count);
    defer allocator.free(outputs);
    var output_index: usize = 0;
    if (primals == .emit) {
        @memcpy(outputs[0..func.returns.len], primal_outputs[0..func.returns.len]);
        output_index = func.returns.len;
    }

    for (func.returns, result.output_tangent_indices) |source_output, tangent_index| {
        outputs[output_index] = if (tangent_index) |index| tangent: {
            if (index >= linear_outputs.len) return error.UnsupportedEqn;
            break :tangent linear_outputs[index];
        } else zero: {
            const tensor = source_output.as_tensor();
            break :zero try builder.scalar_broadcast(
                tensor.dtype,
                tensor.shape.dims,
                0.0,
            );
        };
        output_index += 1;
    }
    if (output_index != outputs.len) return error.UnsupportedEqn;
    return try program.add_function(try builder.finish(outputs));
}

/// Emit a ones-like cotangent for VJP seeding.
pub fn emit_cotangent(builder: *pr.FunctionBuilder, tensor: pr.Tensor) pr.BuildError!*pr.Var {
    const s = try builder.scalar(tensor.dtype, 1.0);
    if (tensor.shape.rank() == 0) return s;
    return try builder.broadcast_in_dim(s, tensor.shape.dims, &.{});
}

test "linearize separates residuals from tangent inputs" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var builder = try pr.FunctionBuilder.init(&program, "square");
    defer builder.deinit();
    const x = try builder.param_tensor(.f32, &.{});
    const y = try builder.multiply(x, x);
    const source = try builder.finish(&.{y});
    const source_id = try program.add_function(source);

    const result = try linearize(std.testing.allocator, &program, source_id, "square_linearized");
    try std.testing.expectEqual(@as(usize, 1), result.residual_count);
    try std.testing.expectEqualSlices(?usize, &.{0}, result.output_tangent_indices);

    const augmented_primal = program.get_function_by_id(result.augmented_primal).?;
    const linear = program.get_function_by_id(result.linear).?;
    try std.testing.expectEqual(@as(usize, 1), augmented_primal.params.len);
    try std.testing.expectEqual(@as(usize, 2), augmented_primal.returns.len);
    try std.testing.expectEqual(@as(usize, 2), linear.params.len);
    try std.testing.expectEqual(@as(usize, 1), linear.returns.len);

    const pr_eval = @import("tests/eval.zig");
    var primal = try pr_eval.HostTensor.init_with_data(std.testing.allocator, &.{}, &.{3.0});
    defer primal.deinit();
    const primal_results = try pr_eval.eval(
        std.testing.allocator,
        &program,
        augmented_primal,
        &.{primal},
    );
    defer {
        for (primal_results) |*value| value.deinit();
        std.testing.allocator.free(primal_results);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), primal_results[0].data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), primal_results[1].data[0], 1e-6);

    var tangent = try pr_eval.HostTensor.init_with_data(std.testing.allocator, &.{}, &.{2.0});
    defer tangent.deinit();
    const tangent_results = try pr_eval.eval(
        std.testing.allocator,
        &program,
        linear,
        &.{ tangent, primal_results[1] },
    );
    defer {
        for (tangent_results) |*value| value.deinit();
        std.testing.allocator.free(tangent_results);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), tangent_results[0].data[0], 1e-6);
}

test "linearized programs serialize" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var builder = try pr.FunctionBuilder.init(&program, "square");
    defer builder.deinit();
    const x = try builder.param_tensor(.f32, &.{});
    const y = try builder.multiply(x, x);
    const source = try builder.finish(&.{y});
    const source_id = try program.add_function(source);
    _ = try linearize(std.testing.allocator, &program, source_id, "square_linearized");

    const serialize = @import("serialize.zig");
    var writer: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer writer.deinit();
    try serialize.emit(&program, &writer.writer);
    const bytes = try writer.toOwnedSlice();
    defer std.testing.allocator.free(bytes);

    var parsed = try serialize.parse(std.testing.allocator, bytes);
    defer parsed.deinit();
    try pr.validate_program(&parsed);
    try std.testing.expectEqual(program.functions().len, parsed.functions().len);
}

test "linearize rejects recursive call graphs" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var placeholder_builder = try pr.FunctionBuilder.init(&program, "recursive");
    defer placeholder_builder.deinit();
    const placeholder_param = try placeholder_builder.param_tensor(.f32, &.{});
    const function_id = try program.add_function(
        try placeholder_builder.finish(&.{placeholder_param}),
    );

    var recursive_builder = try pr.FunctionBuilder.init(&program, "recursive");
    defer recursive_builder.deinit();
    const recursive_param = try recursive_builder.param_tensor(.f32, &.{});
    const recursive_call = try recursive_builder.call(function_id, &.{recursive_param});
    const recursive = try recursive_builder.finish(recursive_call.outputs);
    try program.replace_function(function_id, recursive);

    try std.testing.expectError(
        error.RecursiveDifferentiationUnsupported,
        linearize(std.testing.allocator, &program, function_id, "linearized"),
    );
    try std.testing.expectEqual(@as(usize, 1), program.functions().len);
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
    const source_id = try program.add_function(func);

    const vjp_id = try vjp(std.testing.allocator, &program, source_id, "vjp", .{});
    const vjp_func = program.get_function_by_id(vjp_id).?;
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

test "vjp selects output cotangent seeds" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{});
    const square = try b.multiply(x, x);
    const double = try b.add(x, x);
    const func = try b.finish(&.{ square, double });
    const source_id = try program.add_function(func);

    const vjp_id = try vjp(std.testing.allocator, &program, source_id, "vjp", .{
        .of = &.{1},
    });
    const vjp_func = program.get_function_by_id(vjp_id).?;
    try pr.validate_ops_in_func(vjp_func);

    try std.testing.expectEqual(@as(usize, 2), vjp_func.params.len);
    try std.testing.expectEqual(@as(usize, 1), vjp_func.returns.len);

    const pr_eval = @import("tests/eval.zig");
    var primal = try pr_eval.HostTensor.init_with_data(std.testing.allocator, &.{}, &.{3.0});
    defer primal.deinit();
    var seed = try pr_eval.HostTensor.init_with_data(std.testing.allocator, &.{}, &.{1.0});
    defer seed.deinit();
    const results = try pr_eval.eval(
        std.testing.allocator,
        &program,
        vjp_func,
        &.{ primal, seed },
    );
    defer {
        for (results) |*result| result.deinit();
        std.testing.allocator.free(results);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), results[0].data[0], 1e-6);
}

test "vjp accumulates duplicate output seeds" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{});
    const y = try b.multiply(x, x);
    const func = try b.finish(&.{y});
    const source_id = try program.add_function(func);

    const vjp_id = try vjp(std.testing.allocator, &program, source_id, "vjp", .{
        .of = &.{ 0, 0 },
    });
    const vjp_func = program.get_function_by_id(vjp_id).?;
    try pr.validate_ops_in_func(vjp_func);

    try std.testing.expectEqual(@as(usize, 3), vjp_func.params.len);
    try std.testing.expectEqual(@as(usize, 1), vjp_func.returns.len);
}

test "vjp rejects invalid output selection" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{});
    const y = try b.multiply(x, x);
    const func = try b.finish(&.{y});
    const source_id = try program.add_function(func);

    try std.testing.expectError(
        error.EmptyOutputSelection,
        vjp(std.testing.allocator, &program, source_id, "empty", .{ .of = &.{} }),
    );
    try std.testing.expectError(
        error.OfIndexOutOfRange,
        vjp(std.testing.allocator, &program, source_id, "out_of_range", .{ .of = &.{1} }),
    );
}

test "vjp_with_value returns primals plus gradients" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 2 });
    const y = try b.multiply(x, x);
    const func = try b.finish(&.{y});
    const source_id = try program.add_function(func);

    const vjp_id = try vjp_with_value(std.testing.allocator, &program, source_id, "vjp_with_value", .{});
    const vjp_func = program.get_function_by_id(vjp_id).?;
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
    const source_id = try program.add_function(func);

    const vjp_id = try vjp(std.testing.allocator, &program, source_id, "vjp", .{});
    const vjp_func = program.get_function_by_id(vjp_id).?;
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
    const source_id = try program.add_function(func);

    const vjp_id = try vjp(std.testing.allocator, &program, source_id, "vjp", .{});
    const vjp_func = program.get_function_by_id(vjp_id).?;
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
    const source_id = try program.add_function(func);

    const vjp_id = try vjp(std.testing.allocator, &program, source_id, "vjp", .{});
    const vjp_func = program.get_function_by_id(vjp_id).?;
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
    const source_id = try program.add_function(func);

    const vjp_id = try vjp(std.testing.allocator, &program, source_id, "vjp", .{});
    const vjp_func = program.get_function_by_id(vjp_id).?;
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
    const source_id = try program.add_function(func);

    const jvp_id = try jvp(std.testing.allocator, &program, source_id, "jvp");
    const jvp_func = program.get_function_by_id(jvp_id).?;
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

fn build_repeated_square_calls(program: *pr.Program) !pr.FunctionId {
    var callee_builder = try pr.FunctionBuilder.init(program, "square");
    defer callee_builder.deinit();
    const callee_input = try callee_builder.param_tensor(.f32, &.{});
    const callee_output = try callee_builder.multiply(callee_input, callee_input);
    const callee_id = try program.add_function(try callee_builder.finish(&.{callee_output}));

    var caller_builder = try pr.FunctionBuilder.init(program, "caller");
    defer caller_builder.deinit();
    const caller_input = try caller_builder.param_tensor(.f32, &.{});
    const first_outputs = (try caller_builder.call(callee_id, &.{caller_input})).outputs;
    const second_outputs = (try caller_builder.call(callee_id, &.{caller_input})).outputs;
    const sum = try caller_builder.add(first_outputs[0], second_outputs[0]);
    return try program.add_function(try caller_builder.finish(&.{sum}));
}

test "linearize reuses a callee linearization across call sites" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    const caller_id = try build_repeated_square_calls(&program);
    const result = try linearize(std.testing.allocator, &program, caller_id, "caller");
    try pr.validate_program(&program);

    const augmented_primal = program.get_function_by_id(result.augmented_primal).?;
    const linear = program.get_function_by_id(result.linear).?;
    try std.testing.expectEqual(@as(usize, 3), augmented_primal.ops.len);
    try std.testing.expectEqual(@as(usize, 3), linear.ops.len);

    const first_primal_call = augmented_primal.ops[0].params.call.callee;
    const second_primal_call = augmented_primal.ops[1].params.call.callee;
    try std.testing.expectEqual(first_primal_call, second_primal_call);
    const first_linear_call = linear.ops[0].params.call.callee;
    const second_linear_call = linear.ops[1].params.call.callee;
    try std.testing.expectEqual(first_linear_call, second_linear_call);
    try std.testing.expect(first_primal_call != first_linear_call);
    try std.testing.expectEqual(@as(usize, 6), program.functions().len);
}

test "vjp composes through repeated calls" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    const caller_id = try build_repeated_square_calls(&program);
    const differentiated_id = try vjp(std.testing.allocator, &program, caller_id, "caller_vjp", .{});
    const differentiated = program.get_function_by_id(differentiated_id).?;
    try pr.validate_program(&program);

    try std.testing.expectEqual(@as(usize, 2), differentiated.params.len);
    try std.testing.expectEqual(@as(usize, 1), differentiated.returns.len);

    const pr_eval = @import("tests/eval.zig");
    var primal = try pr_eval.HostTensor.init_with_data(std.testing.allocator, &.{}, &.{3.0});
    defer primal.deinit();
    var seed = try pr_eval.HostTensor.init_with_data(std.testing.allocator, &.{}, &.{1.0});
    defer seed.deinit();
    const results = try pr_eval.eval(
        std.testing.allocator,
        &program,
        differentiated,
        &.{ primal, seed },
    );
    defer {
        for (results) |*result| result.deinit();
        std.testing.allocator.free(results);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), results[0].data[0], 1e-6);
}

test "AD carries call results into nonlinear consumers" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var callee_builder = try pr.FunctionBuilder.init(&program, "square");
    defer callee_builder.deinit();
    const callee_input = try callee_builder.param_tensor(.f32, &.{});
    const callee_output = try callee_builder.multiply(callee_input, callee_input);
    const callee_id = try program.add_function(
        try callee_builder.finish(&.{callee_output}),
    );

    var caller_builder = try pr.FunctionBuilder.init(&program, "fourth_power");
    defer caller_builder.deinit();
    const caller_input = try caller_builder.param_tensor(.f32, &.{});
    const call_outputs = (try caller_builder.call(callee_id, &.{caller_input})).outputs;
    const caller_output = try caller_builder.multiply(call_outputs[0], call_outputs[0]);
    const caller = try caller_builder.finish(&.{caller_output});
    const caller_id = try program.add_function(caller);

    const differentiated_id = try jvp(
        std.testing.allocator,
        &program,
        caller_id,
        "fourth_power_jvp",
    );
    const differentiated = program.get_function_by_id(differentiated_id).?;
    try pr.validate_program(&program);

    const pr_eval = @import("tests/eval.zig");
    var primal = try pr_eval.HostTensor.init_with_data(
        std.testing.allocator,
        &.{},
        &.{2.0},
    );
    defer primal.deinit();
    var tangent = try pr_eval.HostTensor.init_with_data(
        std.testing.allocator,
        &.{},
        &.{3.0},
    );
    defer tangent.deinit();
    const results = try pr_eval.eval(
        std.testing.allocator,
        &program,
        differentiated,
        &.{ primal, tangent },
    );
    defer {
        for (results) |*value| value.deinit();
        std.testing.allocator.free(results);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 96.0), results[0].data[0], 1e-5);

    const reverse_id = try vjp(
        std.testing.allocator,
        &program,
        caller_id,
        "fourth_power_vjp",
        .{},
    );
    const reverse = program.get_function_by_id(reverse_id).?;
    var seed = try pr_eval.HostTensor.init_with_data(
        std.testing.allocator,
        &.{},
        &.{3.0},
    );
    defer seed.deinit();
    const cotangents = try pr_eval.eval(
        std.testing.allocator,
        &program,
        reverse,
        &.{ primal, seed },
    );
    defer {
        for (cotangents) |*value| value.deinit();
        std.testing.allocator.free(cotangents);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 96.0), cotangents[0].data[0], 1e-5);
}

test "jvp_with_value returns primals plus tangents" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 2 });
    const y = try b.multiply(x, x);
    const func = try b.finish(&.{y});
    const source_id = try program.add_function(func);

    const jvp_id = try jvp_with_value(std.testing.allocator, &program, source_id, "jvp_with_value");
    const jvp_func = program.get_function_by_id(jvp_id).?;
    try pr.validate_ops_in_func(jvp_func);

    try std.testing.expectEqual(func.returns.len * 2, jvp_func.returns.len);
    try std.testing.expectEqual(func.params.len * 2, jvp_func.params.len);
}

test "jvp routes maximum tangents through the selected operand" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "maximum");
    defer b.deinit();
    const lhs = try b.param_tensor(.f32, &.{2});
    const rhs = try b.param_tensor(.f32, &.{2});
    const source = try b.finish(&.{try b.maximum(lhs, rhs)});
    const source_id = try program.add_function(source);
    const differentiated_id = try jvp(
        std.testing.allocator,
        &program,
        source_id,
        "maximum_jvp",
    );
    const differentiated = program.get_function_by_id(differentiated_id).?;

    const pr_eval = @import("tests/eval.zig");
    var lhs_value = try pr_eval.HostTensor.init_with_data(
        std.testing.allocator,
        &.{2},
        &.{ 3.0, 1.0 },
    );
    defer lhs_value.deinit();
    var rhs_value = try pr_eval.HostTensor.init_with_data(
        std.testing.allocator,
        &.{2},
        &.{ 2.0, 4.0 },
    );
    defer rhs_value.deinit();
    var lhs_tangent = try pr_eval.HostTensor.init_with_data(
        std.testing.allocator,
        &.{2},
        &.{ 5.0, 6.0 },
    );
    defer lhs_tangent.deinit();
    var rhs_tangent = try pr_eval.HostTensor.init_with_data(
        std.testing.allocator,
        &.{2},
        &.{ 7.0, 8.0 },
    );
    defer rhs_tangent.deinit();

    const results = try pr_eval.eval(
        std.testing.allocator,
        &program,
        differentiated,
        &.{ lhs_value, rhs_value, lhs_tangent, rhs_tangent },
    );
    defer {
        for (results) |*result| result.deinit();
        std.testing.allocator.free(results);
    }
    try std.testing.expectEqualSlices(f32, &.{ 5.0, 8.0 }, results[0].data);
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
    const source_id = try program.add_function(func);

    const jvp_id = try jvp(std.testing.allocator, &program, source_id, "jvp");
    const jvp_func = program.get_function_by_id(jvp_id).?;
    try pr.validate_ops_in_func(jvp_func);

    const orig_t = func.returns[0].as_tensor();
    const jvp_t = jvp_func.returns[0].as_tensor();
    try std.testing.expectEqual(orig_t.dtype, jvp_t.dtype);
    try std.testing.expect(std.mem.eql(i64, orig_t.shape.dims, jvp_t.shape.dims));
}
