/// Op Types - Shared context and types for op implementations
const std = @import("std");
const pr = @import("../pr.zig");
const mlir = @import("../../ffi/mlir/mlir.zig");
const stablehlo = @import("../../ffi/mlir/dialects/stablehlo.zig");

/// Context passed to validation functions
pub const ValidateContext = struct {
    func: pr.Function,
    eqn: pr.Eqn,

    pub fn inputs(self: ValidateContext) []const pr.VarId {
        return self.eqn.inputs.slice(pr.VarId, self.func.varids_store);
    }

    pub fn outputs(self: ValidateContext) []const pr.VarId {
        return self.eqn.outputs.slice(pr.VarId, self.func.varids_store);
    }

    pub fn params(self: ValidateContext) []const pr.Param {
        return self.eqn.params.slice(pr.Param, self.func.params_store);
    }

    pub fn tensor_of(self: ValidateContext, id: pr.VarId) pr.ValidationError!pr.Tensor {
        if (@as(usize, @intCast(id)) >= self.func.avals.len) return error.InvalidVarId;
        const aval = self.func.avals[@intCast(id)];
        return aval.as_tensor() orelse error.UnsupportedAval;
    }
};

/// Context passed to type inference functions
pub const InferContext = struct {
    builder: *pr.FunctionBuilder,
    inputs: []const pr.VarId,
    params: []const pr.Param,

    pub fn tensor_of(self: InferContext, id: pr.VarId) pr.ValidationError!pr.Tensor {
        if (@as(usize, @intCast(id)) >= self.builder.avals.items.len) return error.InvalidVarId;
        const aval = self.builder.avals.items[@intCast(id)];
        return aval.as_tensor() orelse error.UnsupportedAval;
    }

    pub fn alloc(self: InferContext) std.mem.Allocator {
        return self.builder.program.allocator();
    }
};

/// Context passed to lowering functions
pub const LowerContext = struct {
    mlir_ctx: mlir.Context,
    block: mlir.Block,
    loc: mlir.Location,
    value_map: []?mlir.Value,
    func: pr.Function,
    arena: std.mem.Allocator,

    pub fn inputs(self: LowerContext, eqn: pr.Eqn) []const pr.VarId {
        return eqn.inputs.slice(pr.VarId, self.func.varids_store);
    }

    pub fn outputs(self: LowerContext, eqn: pr.Eqn) []const pr.VarId {
        return eqn.outputs.slice(pr.VarId, self.func.varids_store);
    }

    pub fn params(self: LowerContext, eqn: pr.Eqn) []const pr.Param {
        return eqn.params.slice(pr.Param, self.func.params_store);
    }

    pub fn get_value(self: LowerContext, id: pr.VarId) ?mlir.Value {
        return self.value_map[@intCast(id)];
    }

    pub fn set_value(self: LowerContext, id: pr.VarId, value: mlir.Value) void {
        self.value_map[@intCast(id)] = value;
    }

    pub fn tensor_of(self: LowerContext, id: pr.VarId) !pr.Tensor {
        return self.func.avals[@intCast(id)].as_tensor() orelse error.InvalidProgram;
    }

    pub fn tensor_to_mlir_type(self: LowerContext, t: pr.Tensor) !mlir.Type {
        const dims_i64 = try self.arena.alloc(i64, t.shape.dims.len);
        for (t.shape.dims, 0..) |d, i| dims_i64[i] = @intCast(d);
        return mlir.Type.tensor(dims_i64, dtype_to_mlir_type(self.mlir_ctx, t.dtype));
    }
};

/// Context passed to AD backward functions
pub const AdContext = struct {
    builder: *pr.FunctionBuilder,
    primal_map: []?pr.VarId,
    cot_map: []?pr.VarId,
    func: pr.Function,
    allocator: std.mem.Allocator,

    pub fn inputs(self: AdContext, eqn: pr.Eqn) []const pr.VarId {
        return eqn.inputs.slice(pr.VarId, self.func.varids_store);
    }

    pub fn outputs(self: AdContext, eqn: pr.Eqn) []const pr.VarId {
        return eqn.outputs.slice(pr.VarId, self.func.varids_store);
    }

    pub fn params(self: AdContext, eqn: pr.Eqn) []const pr.Param {
        return eqn.params.slice(pr.Param, self.func.params_store);
    }

    pub fn get_primal(self: AdContext, id: pr.VarId) ?pr.VarId {
        return self.primal_map[@intCast(id)];
    }

    pub fn set_primal(self: AdContext, id: pr.VarId, value: pr.VarId) void {
        self.primal_map[@intCast(id)] = value;
    }

    pub fn get_cot(self: AdContext, id: pr.VarId) ?pr.VarId {
        return self.cot_map[@intCast(id)];
    }

    pub fn add_cot(self: AdContext, var_id: pr.VarId, new_cot: pr.VarId) pr.BuildError!void {
        const idx: usize = @intCast(var_id);
        if (self.cot_map[idx]) |existing| {
            self.cot_map[idx] = try self.builder.add(existing, new_cot);
        } else {
            self.cot_map[idx] = new_cot;
        }
    }

    pub fn tensor_of(self: AdContext, id: pr.VarId) pr.Tensor {
        return self.func.avals[@intCast(id)].as_tensor().?;
    }
};

// Helper functions

pub fn dtype_to_mlir_type(ctx: mlir.Context, dt: pr.DType) mlir.Type {
    return switch (dt) {
        .f32 => mlir.Type.float(ctx, .f32),
        .f64 => mlir.Type.float(ctx, .f64),
        .i32 => mlir.Type.int(ctx, .i32),
        .i64 => mlir.Type.int(ctx, .i64),
        .u32 => mlir.Type.int(ctx, .i32),
        .u64 => mlir.Type.int(ctx, .i64),
        .bool => mlir.Type.int(ctx, .i1),
    };
}

pub fn dtype_to_dense_elements_type(dt: pr.DType) mlir.DenseElementsAttributeTypes {
    return switch (dt) {
        .f32 => .f32,
        .f64 => .f64,
        .i32 => .i32,
        .i64 => .i64,
        .u32 => .i32,
        .u64 => .i64,
        .bool => .bool,
    };
}

pub fn same_tensor_type(a: pr.Tensor, b: pr.Tensor) bool {
    if (a.dtype != b.dtype) return false;
    if (a.shape.rank() != b.shape.rank()) return false;
    return std.mem.eql(usize, a.shape.dims, b.shape.dims);
}

pub fn scalar_literal(value_dtype: pr.DType, value: f64) pr.Literal {
    return switch (value_dtype) {
        .f32 => .{ .f32 = @floatCast(value) },
        .f64 => .{ .f64 = value },
        .i32 => .{ .i32 = @intFromFloat(value) },
        .i64 => .{ .i64 = @intFromFloat(value) },
        .u32 => .{ .u32 = @intFromFloat(value) },
        .u64 => .{ .u64 = @intFromFloat(value) },
        .bool => .{ .bool = value != 0.0 },
    };
}

/// Context passed to format functions
pub const FormatContext = struct {
    func: pr.Function,
    eqn: pr.Eqn,

    pub fn inputs(self: FormatContext) []const pr.VarId {
        return self.eqn.inputs.slice(pr.VarId, self.func.varids_store);
    }

    pub fn outputs(self: FormatContext) []const pr.VarId {
        return self.eqn.outputs.slice(pr.VarId, self.func.varids_store);
    }

    pub fn params(self: FormatContext) []const pr.Param {
        return self.eqn.params.slice(pr.Param, self.func.params_store);
    }

    pub fn input_tensor(self: FormatContext, idx: usize) ?pr.Tensor {
        const ins = self.inputs();
        if (idx >= ins.len) return null;
        return self.func.avals[@intCast(ins[idx])].as_tensor();
    }

    pub fn output_tensor(self: FormatContext, idx: usize) ?pr.Tensor {
        const outs = self.outputs();
        if (idx >= outs.len) return null;
        return self.func.avals[@intCast(outs[idx])].as_tensor();
    }
};

pub const Writer = std.Io.Writer;
pub const FormatError = Writer.Error;

// Re-export for convenience
pub const Tensor = pr.Tensor;
pub const Aval = pr.Aval;
pub const VarId = pr.VarId;
pub const Param = pr.Param;
pub const ValidationError = pr.ValidationError;
pub const BuildError = pr.BuildError;
pub const LowerError = error{ InvalidProgram, InvalidMlir, OutOfMemory };
pub const AdError = pr.BuildError || error{ UnsupportedEqn, UnsupportedDType };
