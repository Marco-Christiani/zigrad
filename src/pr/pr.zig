const std = @import("std");

pub const DType = enum {
    f32,
    f64,
    i32,
    i64,
    u32,
    u64,
};

pub const Shape = struct {
    dims: []const usize,

    pub fn rank(self: Shape) usize {
        return self.dims.len;
    }
};

pub const Aval = union(enum) {
    tensor: Tensor,

    pub fn as_tensor(self: Aval) ?Tensor {
        return switch (self) {
            .tensor => |t| t,
        };
    }
};

pub const Tensor = struct {
    dtype: DType,
    shape: Shape,
};

pub const VarId = u32;

pub const Span = struct {
    start: u32,
    len: u32,

    pub fn slice(self: Span, comptime T: type, backing: []const T) []const T {
        const start: usize = @intCast(self.start);
        const end: usize = start + @as(usize, @intCast(self.len));
        return backing[start..end];
    }
};

pub const Literal = union(enum) {
    f32: f32,
    f64: f64,
    i32: i32,
    i64: i64,
    u32: u32,
    u64: u64,

    pub fn dtype(self: Literal) DType {
        return switch (self) {
            .f32 => .f32,
            .f64 => .f64,
            .i32 => .i32,
            .i64 => .i64,
            .u32 => .u32,
            .u64 => .u64,
        };
    }
};

pub const Prim = enum {
    literal,
    add,
    subtract,
    multiply,
    maximum,
    dot,
    reshape,
    broadcast_in_dim,
    transpose,
    reduce_sum,
    call,
    custom_call,
};

pub const Param = union(enum) {
    literal: Literal,
    out_shape: []const usize,
    broadcast_dimensions: []const i64,
    permutation: []const i64,
    reduce_axes: []const i64,
    call_callee: []const u8,
    call_target_name: []const u8,
    has_side_effect: bool,
    out_aval: Aval,
    /// Steering hint: request that this equation be outlined into a separate
    /// function call boundary during lowering (best-effort).
    outline: bool,

    /// Steering hint: request kernelization of this equation/region by a named provider.
    /// This does not change semantics; it is a compilation steering annotation.
    kernelize_provider: []const u8,
};

pub const Eqn = struct {
    prim: Prim,
    inputs: Span, // []VarId (Function.varids_store)
    outputs: Span, // []VarId (Function.varids_store)
    params: Span, // []Param (Function.params_store)
};

pub const Function = struct {
    name: []const u8,
    params: []const VarId,
    returns: []const VarId,
    avals: []const Aval,
    eqns: []const Eqn,
    varids_store: []const VarId,
    params_store: []const Param,
};

pub const Program = struct {
    arena: std.heap.ArenaAllocator,
    functions: []const Function,

    pub fn init(backing_allocator: std.mem.Allocator) Program {
        return .{
            .arena = std.heap.ArenaAllocator.init(backing_allocator),
            .functions = &.{},
        };
    }

    pub fn allocator(self: *Program) std.mem.Allocator {
        return self.arena.allocator();
    }

    pub fn add_function(self: *Program, func: Function) error{OutOfMemory}!void {
        const a = self.allocator();
        const new_items = try a.alloc(Function, self.functions.len + 1);
        @memcpy(new_items[0..self.functions.len], self.functions);
        new_items[self.functions.len] = func;
        self.functions = new_items;
    }

    pub fn deinit(self: *Program) void {
        self.arena.deinit();
    }
};

pub const ValidationError = error{
    InvalidVarId,
    UnsupportedAval,
    InvalidEqnArity,
    InvalidParams,
    LiteralTypeMismatch,
    AddTypeMismatch,
    SubtractTypeMismatch,
    MultiplyTypeMismatch,
    MaximumTypeMismatch,
    DotTypeMismatch,
    ReshapeTypeMismatch,
    BroadcastInDimTypeMismatch,
    TransposeTypeMismatch,
    ReduceSumTypeMismatch,
    CallTypeMismatch,
    CustomCallTypeMismatch,
    DuplicateFunctionName,
};

fn expect_var_in_range(func: Function, id: VarId) ValidationError!void {
    if (@as(usize, @intCast(id)) >= func.avals.len) return error.InvalidVarId;
}

fn expect_tensor(func: Function, id: VarId) ValidationError!Tensor {
    try expect_var_in_range(func, id);
    const aval = func.avals[@intCast(id)];
    return aval.as_tensor() orelse error.UnsupportedAval;
}

fn same_tensor_type(a: Tensor, b: Tensor) bool {
    if (a.dtype != b.dtype) return false;
    if (a.shape.rank() != b.shape.rank()) return false;
    return std.mem.eql(usize, a.shape.dims, b.shape.dims);
}

fn num_elements(dims: []const usize) usize {
    var n: usize = 1;
    for (dims) |d| n *= d;
    return n;
}

fn is_permutation(perm: []const i64, rank: usize) bool {
    if (perm.len != rank) return false;
    if (rank == 0) return true;

    const max_rank: usize = 64;
    if (rank > max_rank) return false;
    var seen = [_]bool{false} ** max_rank;

    for (perm) |p| {
        if (p < 0) return false;
        const idx: usize = @intCast(p);
        if (idx >= rank) return false;
        if (seen[idx]) return false;
        seen[idx] = true;
    }
    return true;
}

fn reduce_sum_output_dims(allocator: std.mem.Allocator, in_dims: []const usize, axes: []const i64) BuildError![]const usize {
    const rank = in_dims.len;
    const max_rank: usize = 64;
    if (rank > max_rank) return error.ReduceSumTypeMismatch;

    var reduce = [_]bool{false} ** max_rank;
    for (axes) |axis| {
        if (axis < 0) return error.ReduceSumTypeMismatch;
        const idx: usize = @intCast(axis);
        if (idx >= rank) return error.ReduceSumTypeMismatch;
        if (reduce[idx]) return error.ReduceSumTypeMismatch;
        reduce[idx] = true;
    }

    var out_count: usize = 0;
    for (0..rank) |i| {
        if (!reduce[i]) out_count += 1;
    }
    const out_dims = try allocator.alloc(usize, out_count);
    var out_i: usize = 0;
    for (0..rank) |i| {
        if (reduce[i]) continue;
        out_dims[out_i] = in_dims[i];
        out_i += 1;
    }
    return out_dims;
}

fn reduce_sum_matches(in_dims: []const usize, out_dims: []const usize, axes: []const i64) bool {
    const rank = in_dims.len;
    const max_rank: usize = 64;
    if (rank > max_rank) return false;

    var reduce = [_]bool{false} ** max_rank;
    for (axes) |axis| {
        if (axis < 0) return false;
        const idx: usize = @intCast(axis);
        if (idx >= rank) return false;
        if (reduce[idx]) return false;
        reduce[idx] = true;
    }

    var out_i: usize = 0;
    for (0..rank) |i| {
        if (reduce[i]) continue;
        if (out_i >= out_dims.len) return false;
        if (in_dims[i] != out_dims[out_i]) return false;
        out_i += 1;
    }

    return out_i == out_dims.len;
}

pub fn param_literal(params: []const Param) ?Literal {
    for (params) |p| {
        switch (p) {
            .literal => |v| return v,
            else => {},
        }
    }
    return null;
}

pub fn param_out_shape(params: []const Param) ?[]const usize {
    for (params) |p| {
        switch (p) {
            .out_shape => |v| return v,
            else => {},
        }
    }
    return null;
}

pub fn param_broadcast_dims(params: []const Param) ?[]const i64 {
    for (params) |p| {
        switch (p) {
            .broadcast_dimensions => |v| return v,
            else => {},
        }
    }
    return null;
}

pub fn param_permutation(params: []const Param) ?[]const i64 {
    for (params) |p| {
        switch (p) {
            .permutation => |v| return v,
            else => {},
        }
    }
    return null;
}

pub fn param_reduce_axes(params: []const Param) ?[]const i64 {
    for (params) |p| {
        switch (p) {
            .reduce_axes => |v| return v,
            else => {},
        }
    }
    return null;
}

pub fn param_call_callee(params: []const Param) ?[]const u8 {
    for (params) |p| {
        switch (p) {
            .call_callee => |v| return v,
            else => {},
        }
    }
    return null;
}

pub fn param_call_target_name(params: []const Param) ?[]const u8 {
    for (params) |p| {
        switch (p) {
            .call_target_name => |v| return v,
            else => {},
        }
    }
    return null;
}

fn same_tensor_signature(a: Tensor, b: Tensor) bool {
    if (a.dtype != b.dtype) return false;
    if (a.shape.dims.len != b.shape.dims.len) return false;
    for (a.shape.dims, 0..) |d, i| {
        if (d != b.shape.dims[i]) return false;
    }
    return true;
}

pub fn param_has_side_effect(params: []const Param) ?bool {
    for (params) |p| {
        switch (p) {
            .has_side_effect => |v| return v,
            else => {},
        }
    }
    return null;
}

pub fn param_out_aval(params: []const Param) ?Aval {
    for (params) |p| {
        switch (p) {
            .out_aval => |v| return v,
            else => {},
        }
    }
    return null;
}

pub fn param_outline(params: []const Param) ?bool {
    for (params) |p| {
        switch (p) {
            .outline => |v| return v,
            else => {},
        }
    }
    return null;
}

pub fn param_kernelize_provider(params: []const Param) ?[]const u8 {
    for (params) |p| {
        switch (p) {
            .kernelize_provider => |v| return v,
            else => {},
        }
    }
    return null;
}

fn validate_broadcast_in_dim_op(operand: Tensor, out: Tensor, broadcast_dimensions: []const i64) ValidationError!void {
    if (operand.dtype != out.dtype) return error.BroadcastInDimTypeMismatch;

    if (broadcast_dimensions.len != operand.shape.rank()) return error.BroadcastInDimTypeMismatch;
    if (out.shape.rank() < operand.shape.rank()) return error.BroadcastInDimTypeMismatch;

    const max_rank: usize = 64;
    if (out.shape.rank() > max_rank) return error.BroadcastInDimTypeMismatch;
    var seen = [_]bool{false} ** max_rank;

    for (broadcast_dimensions, 0..) |d, i| {
        if (d < 0) return error.BroadcastInDimTypeMismatch;
        const out_dim_index: usize = @intCast(d);
        if (out_dim_index >= out.shape.rank()) return error.BroadcastInDimTypeMismatch;
        if (seen[out_dim_index]) return error.BroadcastInDimTypeMismatch;
        seen[out_dim_index] = true;

        const in_dim = operand.shape.dims[i];
        const out_dim = out.shape.dims[out_dim_index];
        if (in_dim != 1 and in_dim != out_dim) return error.BroadcastInDimTypeMismatch;
    }
}

pub fn validate_function(func: Function) ValidationError!void {
    for (func.params) |p| try expect_var_in_range(func, p);
    for (func.returns) |r| try expect_var_in_range(func, r);

    for (func.eqns) |eqn| {
        const inputs = eqn.inputs.slice(VarId, func.varids_store);
        const outputs = eqn.outputs.slice(VarId, func.varids_store);
        const params = eqn.params.slice(Param, func.params_store);

        for (inputs) |in_id| try expect_var_in_range(func, in_id);
        for (outputs) |out_id| try expect_var_in_range(func, out_id);

        switch (eqn.prim) {
            .literal => {
                if (inputs.len != 0 or outputs.len != 1) return error.InvalidEqnArity;
                const lit = param_literal(params) orelse return error.InvalidParams;
                const out = try expect_tensor(func, outputs[0]);
                if (out.dtype != lit.dtype()) return error.LiteralTypeMismatch;
                if (out.shape.rank() != 0) return error.LiteralTypeMismatch;
            },
            .add => {
                if (inputs.len != 2 or outputs.len != 1) return error.InvalidEqnArity;
                const lhs = try expect_tensor(func, inputs[0]);
                const rhs = try expect_tensor(func, inputs[1]);
                const out = try expect_tensor(func, outputs[0]);
                if (!same_tensor_type(lhs, rhs) or !same_tensor_type(lhs, out)) return error.AddTypeMismatch;
            },
            .subtract => {
                if (inputs.len != 2 or outputs.len != 1) return error.InvalidEqnArity;
                const lhs = try expect_tensor(func, inputs[0]);
                const rhs = try expect_tensor(func, inputs[1]);
                const out = try expect_tensor(func, outputs[0]);
                if (!same_tensor_type(lhs, rhs) or !same_tensor_type(lhs, out)) return error.SubtractTypeMismatch;
            },
            .multiply => {
                if (inputs.len != 2 or outputs.len != 1) return error.InvalidEqnArity;
                const lhs = try expect_tensor(func, inputs[0]);
                const rhs = try expect_tensor(func, inputs[1]);
                const out = try expect_tensor(func, outputs[0]);
                if (!same_tensor_type(lhs, rhs) or !same_tensor_type(lhs, out)) return error.MultiplyTypeMismatch;
            },
            .maximum => {
                if (inputs.len != 2 or outputs.len != 1) return error.InvalidEqnArity;
                const lhs = try expect_tensor(func, inputs[0]);
                const rhs = try expect_tensor(func, inputs[1]);
                const out = try expect_tensor(func, outputs[0]);
                if (!same_tensor_type(lhs, rhs) or !same_tensor_type(lhs, out)) return error.MaximumTypeMismatch;
            },
            .dot => {
                if (inputs.len != 2 or outputs.len != 1) return error.InvalidEqnArity;
                const lhs = try expect_tensor(func, inputs[0]);
                const rhs = try expect_tensor(func, inputs[1]);
                const out = try expect_tensor(func, outputs[0]);

                if (lhs.dtype != rhs.dtype or lhs.dtype != out.dtype) return error.DotTypeMismatch;
                if (lhs.shape.rank() != 2 or rhs.shape.rank() != 2 or out.shape.rank() != 2) return error.DotTypeMismatch;
                if (lhs.shape.dims[1] != rhs.shape.dims[0]) return error.DotTypeMismatch;
                if (out.shape.dims[0] != lhs.shape.dims[0] or out.shape.dims[1] != rhs.shape.dims[1]) return error.DotTypeMismatch;
            },
            .reshape => {
                if (inputs.len != 1 or outputs.len != 1) return error.InvalidEqnArity;
                const out_shape = param_out_shape(params) orelse return error.InvalidParams;
                const operand = try expect_tensor(func, inputs[0]);
                const out = try expect_tensor(func, outputs[0]);
                if (!std.mem.eql(usize, out.shape.dims, out_shape)) return error.ReshapeTypeMismatch;
                if (operand.dtype != out.dtype) return error.ReshapeTypeMismatch;
                if (num_elements(operand.shape.dims) != num_elements(out.shape.dims)) return error.ReshapeTypeMismatch;
            },
            .broadcast_in_dim => {
                if (inputs.len != 1 or outputs.len != 1) return error.InvalidEqnArity;
                const out_shape = param_out_shape(params) orelse return error.InvalidParams;
                const bd = param_broadcast_dims(params) orelse return error.InvalidParams;
                const operand = try expect_tensor(func, inputs[0]);
                const out = try expect_tensor(func, outputs[0]);
                if (!std.mem.eql(usize, out.shape.dims, out_shape)) return error.BroadcastInDimTypeMismatch;
                try validate_broadcast_in_dim_op(operand, out, bd);
            },
            .transpose => {
                if (inputs.len != 1 or outputs.len != 1) return error.InvalidEqnArity;
                const perm = param_permutation(params) orelse return error.InvalidParams;
                const operand = try expect_tensor(func, inputs[0]);
                const out = try expect_tensor(func, outputs[0]);
                if (operand.dtype != out.dtype) return error.TransposeTypeMismatch;
                if (!is_permutation(perm, operand.shape.rank())) return error.TransposeTypeMismatch;
                if (out.shape.rank() != operand.shape.rank()) return error.TransposeTypeMismatch;
                for (perm, 0..) |p, out_axis| {
                    const in_axis: usize = @intCast(p);
                    if (out.shape.dims[out_axis] != operand.shape.dims[in_axis]) return error.TransposeTypeMismatch;
                }
            },
            .reduce_sum => {
                if (inputs.len != 1 or outputs.len != 1) return error.InvalidEqnArity;
                const axes = param_reduce_axes(params) orelse return error.InvalidParams;
                const operand = try expect_tensor(func, inputs[0]);
                const out = try expect_tensor(func, outputs[0]);
                if (!reduce_sum_matches(operand.shape.dims, out.shape.dims, axes)) return error.ReduceSumTypeMismatch;
            },
            .call => {
                _ = param_call_callee(params) orelse return error.InvalidParams;
                for (inputs) |in_id| _ = try expect_tensor(func, in_id);
                for (outputs) |out_id| _ = try expect_tensor(func, out_id);
            },
            .custom_call => {
                if (outputs.len != 1) return error.InvalidEqnArity;
                _ = param_call_target_name(params) orelse return error.InvalidParams;
                _ = param_has_side_effect(params) orelse return error.InvalidParams;
                _ = param_out_aval(params) orelse return error.InvalidParams;
                _ = try expect_tensor(func, outputs[0]);
                for (inputs) |in_id| _ = try expect_tensor(func, in_id);
            },
        }
    }
}

pub fn validate_program(program: *const Program) ValidationError!void {
    var i: usize = 0;
    while (i < program.functions.len) : (i += 1) {
        var j: usize = i + 1;
        while (j < program.functions.len) : (j += 1) {
            if (std.mem.eql(u8, program.functions[i].name, program.functions[j].name)) {
                return error.DuplicateFunctionName;
            }
        }
    }

    for (program.functions) |func| {
        try validate_function(func);
    }

    for (program.functions) |func| {
        for (func.eqns) |eqn| {
            if (eqn.prim != .call) continue;

            const params = func.params_store[eqn.params.start..][0..eqn.params.len];
            const callee_name = param_call_callee(params) orelse return error.InvalidParams;

            var callee: ?Function = null;
            for (program.functions) |candidate| {
                if (std.mem.eql(u8, candidate.name, callee_name)) {
                    callee = candidate;
                    break;
                }
            }
            const callee_func = callee orelse return error.CallTypeMismatch;

            const inputs = func.varids_store[eqn.inputs.start..][0..eqn.inputs.len];
            const outputs = func.varids_store[eqn.outputs.start..][0..eqn.outputs.len];

            if (inputs.len != callee_func.params.len) return error.CallTypeMismatch;
            if (outputs.len != callee_func.returns.len) return error.CallTypeMismatch;

            for (inputs, 0..) |in_id, idx| {
                const in_tensor = func.avals[@intCast(in_id)].as_tensor() orelse return error.CallTypeMismatch;
                const callee_tensor = callee_func.avals[@intCast(callee_func.params[idx])].as_tensor() orelse return error.CallTypeMismatch;
                if (!same_tensor_signature(in_tensor, callee_tensor)) return error.CallTypeMismatch;
            }
            for (outputs, 0..) |out_id, idx| {
                const out_tensor = func.avals[@intCast(out_id)].as_tensor() orelse return error.CallTypeMismatch;
                const callee_tensor = callee_func.avals[@intCast(callee_func.returns[idx])].as_tensor() orelse return error.CallTypeMismatch;
                if (!same_tensor_signature(out_tensor, callee_tensor)) return error.CallTypeMismatch;
            }
        }
    }
}

pub const BuildError = ValidationError || error{OutOfMemory};

pub const FunctionBuilder = struct {
    program: *Program,
    name: []const u8,
    avals: std.ArrayList(Aval),
    eqns: std.ArrayList(Eqn),
    varids_store: std.ArrayList(VarId),
    params_store: std.ArrayList(Param),
    params: std.ArrayList(VarId),

    pub fn init(program: *Program, name: []const u8) BuildError!FunctionBuilder {
        const a = program.allocator();
        return .{
            .program = program,
            .name = name,
            .avals = try std.ArrayList(Aval).initCapacity(a, 16),
            .eqns = try std.ArrayList(Eqn).initCapacity(a, 16),
            .varids_store = try std.ArrayList(VarId).initCapacity(a, 64),
            .params_store = try std.ArrayList(Param).initCapacity(a, 64),
            .params = try std.ArrayList(VarId).initCapacity(a, 8),
        };
    }

    pub fn deinit(self: *FunctionBuilder) void {
        const a = self.program.allocator();
        self.avals.deinit(a);
        self.eqns.deinit(a);
        self.varids_store.deinit(a);
        self.params_store.deinit(a);
        self.params.deinit(a);
    }

    fn alloc(self: *FunctionBuilder) std.mem.Allocator {
        return self.program.allocator();
    }

    fn var_with_aval(self: *FunctionBuilder, aval: Aval) BuildError!VarId {
        const a = self.alloc();
        const id: VarId = @intCast(self.avals.items.len);
        try self.avals.append(a, aval);
        return id;
    }

    fn tensor_of(self: *FunctionBuilder, id: VarId) ValidationError!Tensor {
        if (@as(usize, @intCast(id)) >= self.avals.items.len) return error.InvalidVarId;
        const aval = self.avals.items[@intCast(id)];
        return aval.as_tensor() orelse error.UnsupportedAval;
    }

    fn infer_output_aval(self: *FunctionBuilder, prim: Prim, inputs: []const VarId, params: []const Param) BuildError!Aval {
        const a = self.alloc();
        switch (prim) {
            .literal => {
                const lit = param_literal(params) orelse return error.InvalidParams;
                return .{ .tensor = .{ .dtype = lit.dtype(), .shape = .{ .dims = &.{} } } };
            },
            .add, .subtract, .multiply, .maximum => {
                if (inputs.len != 2) return error.InvalidEqnArity;
                const lhs = try self.tensor_of(inputs[0]);
                const rhs = try self.tensor_of(inputs[1]);
                if (!same_tensor_type(lhs, rhs)) return switch (prim) {
                    .add => error.AddTypeMismatch,
                    .subtract => error.SubtractTypeMismatch,
                    .multiply => error.MultiplyTypeMismatch,
                    .maximum => error.MaximumTypeMismatch,
                    else => unreachable,
                };
                return .{ .tensor = lhs };
            },
            .dot => {
                if (inputs.len != 2) return error.InvalidEqnArity;
                const lhs = try self.tensor_of(inputs[0]);
                const rhs = try self.tensor_of(inputs[1]);

                if (lhs.dtype != rhs.dtype) return error.DotTypeMismatch;
                if (lhs.shape.rank() != 2 or rhs.shape.rank() != 2) return error.DotTypeMismatch;
                if (lhs.shape.dims[1] != rhs.shape.dims[0]) return error.DotTypeMismatch;

                const out_dims = try a.dupe(usize, &[_]usize{ lhs.shape.dims[0], rhs.shape.dims[1] });
                return .{ .tensor = .{ .dtype = lhs.dtype, .shape = .{ .dims = out_dims } } };
            },
            .reshape => {
                if (inputs.len != 1) return error.InvalidEqnArity;
                const out_shape = param_out_shape(params) orelse return error.InvalidParams;
                const operand = try self.tensor_of(inputs[0]);
                if (num_elements(operand.shape.dims) != num_elements(out_shape)) return error.ReshapeTypeMismatch;
                return .{ .tensor = .{ .dtype = operand.dtype, .shape = .{ .dims = out_shape } } };
            },
            .broadcast_in_dim => {
                if (inputs.len != 1) return error.InvalidEqnArity;
                const out_shape = param_out_shape(params) orelse return error.InvalidParams;
                const bd = param_broadcast_dims(params) orelse return error.InvalidParams;
                const operand = try self.tensor_of(inputs[0]);
                const out_tensor = Tensor{ .dtype = operand.dtype, .shape = .{ .dims = out_shape } };
                try validate_broadcast_in_dim_op(operand, out_tensor, bd);
                return .{ .tensor = out_tensor };
            },
            .transpose => {
                if (inputs.len != 1) return error.InvalidEqnArity;
                const perm = param_permutation(params) orelse return error.InvalidParams;
                const operand = try self.tensor_of(inputs[0]);
                if (!is_permutation(perm, operand.shape.rank())) return error.TransposeTypeMismatch;

                const out_dims = try a.alloc(usize, operand.shape.rank());
                for (perm, 0..) |p, i| out_dims[i] = operand.shape.dims[@intCast(p)];
                return .{ .tensor = .{ .dtype = operand.dtype, .shape = .{ .dims = out_dims } } };
            },
            .reduce_sum => {
                if (inputs.len != 1) return error.InvalidEqnArity;
                const axes = param_reduce_axes(params) orelse return error.InvalidParams;
                const operand = try self.tensor_of(inputs[0]);
                const out_dims = try reduce_sum_output_dims(a, operand.shape.dims, axes);
                return .{ .tensor = .{ .dtype = operand.dtype, .shape = .{ .dims = out_dims } } };
            },
            .call => {
                _ = param_call_callee(params) orelse return error.InvalidParams;
                return error.InvalidEqnArity;
            },
            .custom_call => {
                const out_aval = param_out_aval(params) orelse return error.InvalidParams;
                _ = param_call_target_name(params) orelse return error.InvalidParams;
                _ = param_has_side_effect(params) orelse return error.InvalidParams;
                _ = out_aval.as_tensor() orelse return error.CustomCallTypeMismatch;
                for (inputs) |in_id| _ = try self.tensor_of(in_id);
                return out_aval;
            },
        }
    }

    pub fn emit(self: *FunctionBuilder, prim: Prim, inputs: []const VarId, params: []const Param) BuildError!VarId {
        const a = self.alloc();

        const out_aval = try self.infer_output_aval(prim, inputs, params);
        const out = try self.var_with_aval(out_aval);

        const inputs_start: u32 = @intCast(self.varids_store.items.len);
        try self.varids_store.appendSlice(a, inputs);
        const inputs_span: Span = .{ .start = inputs_start, .len = @intCast(inputs.len) };

        const outputs_start: u32 = @intCast(self.varids_store.items.len);
        try self.varids_store.append(a, out);
        const outputs_span: Span = .{ .start = outputs_start, .len = 1 };

        const params_start: u32 = @intCast(self.params_store.items.len);
        try self.params_store.appendSlice(a, params);
        const params_span: Span = .{ .start = params_start, .len = @intCast(params.len) };

        try self.eqns.append(a, .{
            .prim = prim,
            .inputs = inputs_span,
            .outputs = outputs_span,
            .params = params_span,
        });

        return out;
    }

    fn emit_with_outputs(self: *FunctionBuilder, prim: Prim, inputs: []const VarId, outputs: []const VarId, params: []const Param) BuildError!void {
        const a = self.alloc();

        const inputs_start: u32 = @intCast(self.varids_store.items.len);
        try self.varids_store.appendSlice(a, inputs);
        const inputs_span: Span = .{ .start = inputs_start, .len = @intCast(inputs.len) };

        const outputs_start: u32 = @intCast(self.varids_store.items.len);
        try self.varids_store.appendSlice(a, outputs);
        const outputs_span: Span = .{ .start = outputs_start, .len = @intCast(outputs.len) };

        const params_start: u32 = @intCast(self.params_store.items.len);
        try self.params_store.appendSlice(a, params);
        const params_span: Span = .{ .start = params_start, .len = @intCast(params.len) };

        try self.eqns.append(a, .{
            .prim = prim,
            .inputs = inputs_span,
            .outputs = outputs_span,
            .params = params_span,
        });
    }

    pub fn param_tensor(self: *FunctionBuilder, dtype: DType, dims: []const usize) BuildError!VarId {
        const a = self.alloc();
        const dims_copy = try a.dupe(usize, dims);
        const id = try self.var_with_aval(.{ .tensor = .{ .dtype = dtype, .shape = .{ .dims = dims_copy } } });
        try self.params.append(a, id);
        return id;
    }

    pub fn literal_scalar(self: *FunctionBuilder, value: Literal) BuildError!VarId {
        return self.emit(.literal, &.{}, &.{.{ .literal = value }});
    }

    pub fn add(self: *FunctionBuilder, lhs: VarId, rhs: VarId) BuildError!VarId {
        return self.emit(.add, &.{ lhs, rhs }, &.{});
    }

    pub fn subtract(self: *FunctionBuilder, lhs: VarId, rhs: VarId) BuildError!VarId {
        return self.emit(.subtract, &.{ lhs, rhs }, &.{});
    }

    pub fn multiply(self: *FunctionBuilder, lhs: VarId, rhs: VarId) BuildError!VarId {
        return self.emit(.multiply, &.{ lhs, rhs }, &.{});
    }

    pub fn maximum(self: *FunctionBuilder, lhs: VarId, rhs: VarId) BuildError!VarId {
        return self.emit(.maximum, &.{ lhs, rhs }, &.{});
    }

    pub fn dot(self: *FunctionBuilder, lhs: VarId, rhs: VarId) BuildError!VarId {
        return self.emit(.dot, &.{ lhs, rhs }, &.{});
    }

    pub fn reshape(self: *FunctionBuilder, operand: VarId, out_dims: []const usize) BuildError!VarId {
        const a = self.alloc();
        const out_shape = try a.dupe(usize, out_dims);
        return self.emit(.reshape, &.{operand}, &.{.{ .out_shape = out_shape }});
    }

    pub fn broadcast_in_dim(self: *FunctionBuilder, operand: VarId, out_dims: []const usize, broadcast_dimensions: []const i64) BuildError!VarId {
        const a = self.alloc();
        const out_shape = try a.dupe(usize, out_dims);
        const bd_copy = try a.dupe(i64, broadcast_dimensions);
        return self.emit(.broadcast_in_dim, &.{operand}, &.{
            .{ .out_shape = out_shape },
            .{ .broadcast_dimensions = bd_copy },
        });
    }

    pub fn transpose(self: *FunctionBuilder, operand: VarId, permutation: []const i64) BuildError!VarId {
        const a = self.alloc();
        const perm_copy = try a.dupe(i64, permutation);
        return self.emit(.transpose, &.{operand}, &.{.{ .permutation = perm_copy }});
    }

    pub fn reduce_sum(self: *FunctionBuilder, operand: VarId, axes: []const i64) BuildError!VarId {
        const a = self.alloc();
        const axes_copy = try a.dupe(i64, axes);
        return self.emit(.reduce_sum, &.{operand}, &.{.{ .reduce_axes = axes_copy }});
    }

    pub fn custom_call(self: *FunctionBuilder, target: []const u8, operands: []const VarId, out_like: VarId) BuildError!VarId {
        const a = self.alloc();

        const out_aval = self.avals.items[@intCast(out_like)];
        _ = out_aval.as_tensor() orelse return error.CustomCallTypeMismatch;
        for (operands) |op| _ = try self.tensor_of(op);

        const target_copy = try a.dupe(u8, target);
        return self.emit(.custom_call, operands, &.{
            .{ .call_target_name = target_copy },
            .{ .has_side_effect = false },
            .{ .out_aval = out_aval },
        });
    }

    pub fn call(self: *FunctionBuilder, callee: []const u8, inputs: []const VarId) BuildError![]VarId {
        var callee_func: ?Function = null;
        for (self.program.functions) |func| {
            if (std.mem.eql(u8, func.name, callee)) {
                callee_func = func;
                break;
            }
        }
        const callee_fn = callee_func orelse return error.InvalidParams;

        if (inputs.len != callee_fn.params.len) return error.InvalidEqnArity;
        for (inputs, 0..) |in_id, i| {
            const in_tensor = try self.tensor_of(in_id);
            const callee_tensor = callee_fn.avals[@intCast(callee_fn.params[i])].as_tensor() orelse return error.CallTypeMismatch;
            if (!same_tensor_signature(in_tensor, callee_tensor)) return error.CallTypeMismatch;
        }

        const a = self.alloc();
        const outputs = try a.alloc(VarId, callee_fn.returns.len);
        for (callee_fn.returns, 0..) |ret_id, i| {
            const aval = callee_fn.avals[@intCast(ret_id)];
            outputs[i] = try self.var_with_aval(aval);
        }

        const callee_copy = try a.dupe(u8, callee);
        try self.emit_with_outputs(.call, inputs, outputs, &.{.{ .call_callee = callee_copy }});
        return outputs;
    }

    pub fn finish(self: *FunctionBuilder, returns: []const VarId) BuildError!Function {
        const a = self.alloc();
        const func = Function{
            .name = self.name,
            .params = try self.params.toOwnedSlice(a),
            .returns = try a.dupe(VarId, returns),
            .avals = try self.avals.toOwnedSlice(a),
            .eqns = try self.eqns.toOwnedSlice(a),
            .varids_store = try self.varids_store.toOwnedSlice(a),
            .params_store = try self.params_store.toOwnedSlice(a),
        };
        try validate_function(func);
        return func;
    }
};

test "FunctionBuilder reshape validation" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    try std.testing.expectError(error.ReshapeTypeMismatch, b.reshape(x, &.{4}));
}

test "FunctionBuilder broadcast_in_dim basic" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{3});
    const y = try b.broadcast_in_dim(x, &.{ 2, 3 }, &.{1});
    const func = try b.finish(&.{y});
    try validate_function(func);
}

test "FunctionBuilder transpose validation" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 3, 4 });
    const y = try b.transpose(x, &.{ 2, 0, 1 });
    const func = try b.finish(&.{y});
    try validate_function(func);
}

test "FunctionBuilder reduce_sum basic" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.reduce_sum(x, &.{0});
    const func = try b.finish(&.{y});
    try validate_function(func);
}

test "FunctionBuilder literal scalar" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const one = try b.literal_scalar(.{ .f32 = 1.0 });
    const func = try b.finish(&.{one});
    try validate_function(func);
}
