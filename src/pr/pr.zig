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

    pub fn asTensor(self: Aval) ?Tensor {
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

pub const Eqn = union(enum) {
    literal: LiteralEqn,
    add: Binary,
    subtract: Binary,
    multiply: Binary,
    maximum: Binary,
    dot: Binary,
    reshape: Unary,
    broadcast_in_dim: BroadcastInDim,
    transpose: Transpose,
    custom_call: CustomCall,
};

pub const LiteralEqn = struct {
    value: Literal,
    out: VarId,
};

pub const Binary = struct {
    lhs: VarId,
    rhs: VarId,
    out: VarId,
};

pub const Unary = struct {
    operand: VarId,
    out: VarId,
};

pub const BroadcastInDim = struct {
    operand: VarId,
    out: VarId,
    /// Maps operand dims to output dims (StableHLO `broadcast_dimensions`).
    broadcast_dimensions: []const i64,
};

pub const Transpose = struct {
    operand: VarId,
    out: VarId,
    /// Permutation of [0..rank).
    permutation: []const i64,
};

pub const CustomCall = struct {
    target: []const u8,
    operands: []const VarId,
    out: VarId,
    has_side_effect: bool = false,
};

pub const Function = struct {
    name: []const u8,
    params: []const VarId,
    returns: []const VarId,
    avals: []const Aval,
    eqns: []const Eqn,
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

    pub fn addFunction(self: *Program, func: Function) error{OutOfMemory}!void {
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
    LiteralTypeMismatch,
    AddTypeMismatch,
    SubtractTypeMismatch,
    MultiplyTypeMismatch,
    MaximumTypeMismatch,
    DotTypeMismatch,
    ReshapeTypeMismatch,
    BroadcastInDimTypeMismatch,
    TransposeTypeMismatch,
    CustomCallTypeMismatch,
};

fn expectVarInRange(func: Function, id: VarId) ValidationError!void {
    if (@as(usize, @intCast(id)) >= func.avals.len) return error.InvalidVarId;
}

fn expectTensor(func: Function, id: VarId) ValidationError!Tensor {
    try expectVarInRange(func, id);
    const aval = func.avals[@intCast(id)];
    return aval.asTensor() orelse error.UnsupportedAval;
}

fn sameTensorType(a: Tensor, b: Tensor) bool {
    if (a.dtype != b.dtype) return false;
    if (a.shape.rank() != b.shape.rank()) return false;
    return std.mem.eql(usize, a.shape.dims, b.shape.dims);
}

fn numElements(dims: []const usize) usize {
    var n: usize = 1;
    for (dims) |d| n *= d;
    return n;
}

fn isPermutation(perm: []const i64, rank: usize) bool {
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

fn validateReshapeOp(operand: Tensor, out: Tensor) ValidationError!void {
    if (operand.dtype != out.dtype) return error.ReshapeTypeMismatch;
    if (numElements(operand.shape.dims) != numElements(out.shape.dims)) return error.ReshapeTypeMismatch;
}

fn validateBroadcastInDimOp(operand: Tensor, out: Tensor, broadcast_dimensions: []const i64) ValidationError!void {
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

fn validateTransposeOp(operand: Tensor, out: Tensor, permutation: []const i64) ValidationError!void {
    if (operand.dtype != out.dtype) return error.TransposeTypeMismatch;
    if (!isPermutation(permutation, operand.shape.rank())) return error.TransposeTypeMismatch;
    if (out.shape.rank() != operand.shape.rank()) return error.TransposeTypeMismatch;

    for (permutation, 0..) |p, out_axis| {
        const in_axis: usize = @intCast(p);
        if (out.shape.dims[out_axis] != operand.shape.dims[in_axis]) return error.TransposeTypeMismatch;
    }
}

pub fn validateFunction(func: Function) ValidationError!void {
    for (func.params) |p| try expectVarInRange(func, p);
    for (func.returns) |r| try expectVarInRange(func, r);

    for (func.eqns) |eqn| {
        switch (eqn) {
            .literal => |l| {
                const out = try expectTensor(func, l.out);
                if (out.dtype != l.value.dtype()) return error.LiteralTypeMismatch;
                if (out.shape.rank() != 0) return error.LiteralTypeMismatch;
            },
            .add => |b| {
                const lhs = try expectTensor(func, b.lhs);
                const rhs = try expectTensor(func, b.rhs);
                const out = try expectTensor(func, b.out);
                if (!sameTensorType(lhs, rhs) or !sameTensorType(lhs, out)) return error.AddTypeMismatch;
            },
            .subtract => |b| {
                const lhs = try expectTensor(func, b.lhs);
                const rhs = try expectTensor(func, b.rhs);
                const out = try expectTensor(func, b.out);
                if (!sameTensorType(lhs, rhs) or !sameTensorType(lhs, out)) return error.SubtractTypeMismatch;
            },
            .multiply => |b| {
                const lhs = try expectTensor(func, b.lhs);
                const rhs = try expectTensor(func, b.rhs);
                const out = try expectTensor(func, b.out);
                if (!sameTensorType(lhs, rhs) or !sameTensorType(lhs, out)) return error.MultiplyTypeMismatch;
            },
            .maximum => |b| {
                const lhs = try expectTensor(func, b.lhs);
                const rhs = try expectTensor(func, b.rhs);
                const out = try expectTensor(func, b.out);
                if (!sameTensorType(lhs, rhs) or !sameTensorType(lhs, out)) return error.MaximumTypeMismatch;
            },
            .dot => |b| {
                const lhs = try expectTensor(func, b.lhs);
                const rhs = try expectTensor(func, b.rhs);
                const out = try expectTensor(func, b.out);

                if (lhs.dtype != rhs.dtype or lhs.dtype != out.dtype) return error.DotTypeMismatch;
                if (lhs.shape.rank() != 2 or rhs.shape.rank() != 2 or out.shape.rank() != 2) return error.DotTypeMismatch;
                if (lhs.shape.dims[1] != rhs.shape.dims[0]) return error.DotTypeMismatch;
                if (out.shape.dims[0] != lhs.shape.dims[0] or out.shape.dims[1] != rhs.shape.dims[1]) return error.DotTypeMismatch;
            },
            .reshape => |u| {
                const operand = try expectTensor(func, u.operand);
                const out = try expectTensor(func, u.out);
                try validateReshapeOp(operand, out);
            },
            .broadcast_in_dim => |b| {
                const operand = try expectTensor(func, b.operand);
                const out = try expectTensor(func, b.out);
                try validateBroadcastInDimOp(operand, out, b.broadcast_dimensions);
            },
            .transpose => |t| {
                const operand = try expectTensor(func, t.operand);
                const out = try expectTensor(func, t.out);
                try validateTransposeOp(operand, out, t.permutation);
            },
            .custom_call => |cc| {
                _ = try expectTensor(func, cc.out);
                for (cc.operands) |op| _ = try expectTensor(func, op);
            },
        }
    }
}

pub const BuildError = ValidationError || error{OutOfMemory};

pub const FunctionBuilder = struct {
    program: *Program,
    name: []const u8,
    avals: std.ArrayList(Aval),
    eqns: std.ArrayList(Eqn),
    params: std.ArrayList(VarId),

    pub fn init(program: *Program, name: []const u8) BuildError!FunctionBuilder {
        const a = program.allocator();
        return .{
            .program = program,
            .name = name,
            .avals = try std.ArrayList(Aval).initCapacity(a, 16),
            .eqns = try std.ArrayList(Eqn).initCapacity(a, 16),
            .params = try std.ArrayList(VarId).initCapacity(a, 8),
        };
    }

    pub fn deinit(self: *FunctionBuilder) void {
        const a = self.program.allocator();
        self.avals.deinit(a);
        self.eqns.deinit(a);
        self.params.deinit(a);
    }

    fn alloc(self: *FunctionBuilder) std.mem.Allocator {
        return self.program.allocator();
    }

    fn varTensor(self: *FunctionBuilder, dtype: DType, dims: []const usize) BuildError!VarId {
        const a = self.alloc();
        const dims_copy = try a.dupe(usize, dims);
        const id: VarId = @intCast(self.avals.items.len);
        try self.avals.append(a, .{ .tensor = .{ .dtype = dtype, .shape = .{ .dims = dims_copy } } });
        return id;
    }

    pub fn paramTensor(self: *FunctionBuilder, dtype: DType, dims: []const usize) BuildError!VarId {
        const a = self.alloc();
        const id = try self.varTensor(dtype, dims);
        try self.params.append(a, id);
        return id;
    }

    pub fn literalScalar(self: *FunctionBuilder, value: Literal) BuildError!VarId {
        const a = self.alloc();
        const out = try self.varTensor(value.dtype(), &.{});
        try self.eqns.append(a, .{ .literal = .{ .value = value, .out = out } });
        return out;
    }

    fn tensorOf(self: *FunctionBuilder, id: VarId) ValidationError!Tensor {
        if (@as(usize, @intCast(id)) >= self.avals.items.len) return error.InvalidVarId;
        const aval = self.avals.items[@intCast(id)];
        return aval.asTensor() orelse error.UnsupportedAval;
    }

    pub fn add(self: *FunctionBuilder, lhs: VarId, rhs: VarId) BuildError!VarId {
        const a = self.alloc();
        const lhs_t = try self.tensorOf(lhs);
        const rhs_t = try self.tensorOf(rhs);
        if (!sameTensorType(lhs_t, rhs_t)) return error.AddTypeMismatch;

        const out = try self.varTensor(lhs_t.dtype, lhs_t.shape.dims);
        try self.eqns.append(a, .{ .add = .{ .lhs = lhs, .rhs = rhs, .out = out } });
        return out;
    }

    pub fn subtract(self: *FunctionBuilder, lhs: VarId, rhs: VarId) BuildError!VarId {
        const a = self.alloc();
        const lhs_t = try self.tensorOf(lhs);
        const rhs_t = try self.tensorOf(rhs);
        if (!sameTensorType(lhs_t, rhs_t)) return error.SubtractTypeMismatch;

        const out = try self.varTensor(lhs_t.dtype, lhs_t.shape.dims);
        try self.eqns.append(a, .{ .subtract = .{ .lhs = lhs, .rhs = rhs, .out = out } });
        return out;
    }

    pub fn multiply(self: *FunctionBuilder, lhs: VarId, rhs: VarId) BuildError!VarId {
        const a = self.alloc();
        const lhs_t = try self.tensorOf(lhs);
        const rhs_t = try self.tensorOf(rhs);
        if (!sameTensorType(lhs_t, rhs_t)) return error.MultiplyTypeMismatch;

        const out = try self.varTensor(lhs_t.dtype, lhs_t.shape.dims);
        try self.eqns.append(a, .{ .multiply = .{ .lhs = lhs, .rhs = rhs, .out = out } });
        return out;
    }

    pub fn maximum(self: *FunctionBuilder, lhs: VarId, rhs: VarId) BuildError!VarId {
        const a = self.alloc();
        const lhs_t = try self.tensorOf(lhs);
        const rhs_t = try self.tensorOf(rhs);
        if (!sameTensorType(lhs_t, rhs_t)) return error.MaximumTypeMismatch;

        const out = try self.varTensor(lhs_t.dtype, lhs_t.shape.dims);
        try self.eqns.append(a, .{ .maximum = .{ .lhs = lhs, .rhs = rhs, .out = out } });
        return out;
    }

    pub fn dot(self: *FunctionBuilder, lhs: VarId, rhs: VarId) BuildError!VarId {
        const a = self.alloc();
        const lhs_t = try self.tensorOf(lhs);
        const rhs_t = try self.tensorOf(rhs);

        if (lhs_t.dtype != rhs_t.dtype) return error.DotTypeMismatch;
        if (lhs_t.shape.rank() != 2 or rhs_t.shape.rank() != 2) return error.DotTypeMismatch;
        if (lhs_t.shape.dims[1] != rhs_t.shape.dims[0]) return error.DotTypeMismatch;

        const out_dims = [_]usize{ lhs_t.shape.dims[0], rhs_t.shape.dims[1] };
        const out = try self.varTensor(lhs_t.dtype, &out_dims);
        try self.eqns.append(a, .{ .dot = .{ .lhs = lhs, .rhs = rhs, .out = out } });
        return out;
    }

    pub fn reshape(self: *FunctionBuilder, operand: VarId, out_dims: []const usize) BuildError!VarId {
        const a = self.alloc();
        const operand_t = try self.tensorOf(operand);
        if (numElements(operand_t.shape.dims) != numElements(out_dims)) return error.ReshapeTypeMismatch;

        const out = try self.varTensor(operand_t.dtype, out_dims);
        try self.eqns.append(a, .{ .reshape = .{ .operand = operand, .out = out } });
        return out;
    }

    pub fn broadcastInDim(self: *FunctionBuilder, operand: VarId, out_dims: []const usize, broadcast_dimensions: []const i64) BuildError!VarId {
        const a = self.alloc();
        const operand_t = try self.tensorOf(operand);

        const out = try self.varTensor(operand_t.dtype, out_dims);
        const out_t = try self.tensorOf(out);
        try validateBroadcastInDimOp(operand_t, out_t, broadcast_dimensions);

        const bd_copy = try a.dupe(i64, broadcast_dimensions);
        try self.eqns.append(a, .{
            .broadcast_in_dim = .{
                .operand = operand,
                .out = out,
                .broadcast_dimensions = bd_copy,
            },
        });
        return out;
    }

    pub fn transpose(self: *FunctionBuilder, operand: VarId, permutation: []const i64) BuildError!VarId {
        const a = self.alloc();
        const operand_t = try self.tensorOf(operand);
        if (!isPermutation(permutation, operand_t.shape.rank())) return error.TransposeTypeMismatch;

        const out_dims = try a.alloc(usize, operand_t.shape.rank());
        for (permutation, 0..) |p, i| out_dims[i] = operand_t.shape.dims[@intCast(p)];
        const out = try self.varTensor(operand_t.dtype, out_dims);

        const perm_copy = try a.dupe(i64, permutation);
        try self.eqns.append(a, .{
            .transpose = .{
                .operand = operand,
                .out = out,
                .permutation = perm_copy,
            },
        });

        return out;
    }

    pub fn customCall(self: *FunctionBuilder, target: []const u8, operands: []const VarId, out_like: VarId) BuildError!VarId {
        const a = self.alloc();

        const out_t = try self.tensorOf(out_like);
        for (operands) |op| _ = try self.tensorOf(op);

        const out = try self.varTensor(out_t.dtype, out_t.shape.dims);
        const operands_copy = try a.dupe(VarId, operands);
        const target_copy = try a.dupe(u8, target);

        try self.eqns.append(a, .{
            .custom_call = .{
                .target = target_copy,
                .operands = operands_copy,
                .out = out,
                .has_side_effect = false,
            },
        });

        return out;
    }

    pub fn finish(self: *FunctionBuilder, returns: []const VarId) BuildError!Function {
        const a = self.alloc();
        const func = Function{
            .name = self.name,
            .params = try self.params.toOwnedSlice(a),
            .returns = try a.dupe(VarId, returns),
            .avals = try self.avals.toOwnedSlice(a),
            .eqns = try self.eqns.toOwnedSlice(a),
        };
        try validateFunction(func);
        return func;
    }
};

test "FunctionBuilder reshape validation" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.paramTensor(.f32, &.{ 2, 3 });
    try std.testing.expectError(error.ReshapeTypeMismatch, b.reshape(x, &.{4}));
}

test "FunctionBuilder broadcast_in_dim basic" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.paramTensor(.f32, &.{3});
    const y = try b.broadcastInDim(x, &.{ 2, 3 }, &.{1});
    const func = try b.finish(&.{y});
    try validateFunction(func);
}

test "FunctionBuilder transpose validation" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.paramTensor(.f32, &.{ 2, 3, 4 });
    const y = try b.transpose(x, &.{ 2, 0, 1 });
    const func = try b.finish(&.{y});
    try validateFunction(func);
}

test "FunctionBuilder literal scalar" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const one = try b.literalScalar(.{ .f32 = 1.0 });
    const func = try b.finish(&.{one});
    try validateFunction(func);
}
