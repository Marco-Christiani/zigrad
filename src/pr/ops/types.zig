/// Context types for op implementations and helpers.
/// TODO: Reconsider this file, probably not the ideal way to organize.
const std = @import("std");
const pr = @import("../pr.zig");

/// Context passed to validation functions.
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

/// Context passed to type inference functions.
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

/// Context passed to AD forward/backward functions.
///
/// Used for both VJP (reverse-mode) and JVP (forward-mode) transforms.
/// For VJP: `tangent_map` is null, `cot_map` holds cotangent accumulation.
/// For JVP: `cot_map` is null, `tangent_map` holds tangent propagation.
/// TODO: we can likely collapse to a single "dual" field depending on how
///  this impacts semantics downstream, would want to check this first, but
///  it seems cleaner from here.
pub const AdContext = struct {
    builder: *pr.FunctionBuilder,
    primal_map: []?pr.VarId,
    /// In reverse-mode (VJP) holds cotangents
    cot_map: ?[]?pr.VarId,
    /// In forward-mode (JVP) holds tangents
    tangent_map: ?[]?pr.VarId,
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
        const cmap = self.cot_map orelse return null;
        return cmap[@intCast(id)];
    }

    pub fn add_cot(self: AdContext, var_id: pr.VarId, new_cot: pr.VarId) pr.BuildError!void {
        const cmap = self.cot_map orelse return;
        const idx: usize = @intCast(var_id);
        if (cmap[idx]) |existing| {
            cmap[idx] = try self.builder.add(existing, new_cot);
        } else {
            cmap[idx] = new_cot;
        }
    }

    pub fn get_tangent(self: AdContext, id: pr.VarId) ?pr.VarId {
        const tmap = self.tangent_map orelse return null;
        return tmap[@intCast(id)];
    }

    pub fn set_tangent(self: AdContext, id: pr.VarId, value: pr.VarId) void {
        self.tangent_map.?[@intCast(id)] = value;
    }

    pub fn tensor_of(self: AdContext, id: pr.VarId) pr.Tensor {
        return self.func.avals[@intCast(id)].as_tensor().?;
    }

    /// Look up tensor info for a VarId created by the VJP builder (e.g. cotangents).
    pub fn builder_tensor_of(self: AdContext, id: pr.VarId) pr.Tensor {
        return self.builder.avals.items[@intCast(id)].as_tensor().?;
    }
};

/// Context passed to format functions.
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

// ============================================================================
// Shared Helpers
// ============================================================================

pub fn same_tensor_type(a: pr.Tensor, b: pr.Tensor) bool {
    if (a.dtype != b.dtype) return false;
    if (a.shape.rank() != b.shape.rank()) return false;
    return std.mem.eql(i64, a.shape.dims, b.shape.dims);
}

fn f32_to_bf16_bits(val: f32) u16 {
    const bits: u32 = @bitCast(val);
    return @intCast(bits >> 16);
}

pub fn scalar_literal(value_dtype: pr.DType, value: f64) pr.Literal {
    return switch (value_dtype) {
        .f16 => .{ .f16 = f32_to_f16_bits(@floatCast(value)) },
        .bf16 => .{ .bf16 = f32_to_bf16_bits(@floatCast(value)) },
        .f32 => .{ .f32 = @floatCast(value) },
        .f64 => .{ .f64 = value },
        .i8 => .{ .i8 = @intFromFloat(value) },
        .u8 => .{ .u8 = @intFromFloat(value) },
        .i32 => .{ .i32 = @intFromFloat(value) },
        .i64 => .{ .i64 = @intFromFloat(value) },
        .u32 => .{ .u32 = @intFromFloat(value) },
        .u64 => .{ .u64 = @intFromFloat(value) },
        .bool => .{ .bool = value != 0.0 },
    };
}

fn f32_to_f16_bits(val: f32) u16 {
    const bits: u32 = @bitCast(val);
    const sign: u16 = @intCast((bits >> 16) & 0x8000);
    const exp_f32: i32 = @intCast((bits >> 23) & 0xFF);
    const mant: u32 = bits & 0x7FFFFF;
    if (exp_f32 == 0xFF) return sign | 0x7C00 | if (mant != 0) @as(u16, 1) else 0;
    const exp_f16 = exp_f32 - 127 + 15;
    if (exp_f16 >= 31) return sign | 0x7C00;
    if (exp_f16 <= 0) return sign;
    return sign | @as(u16, @intCast(exp_f16)) << 10 | @as(u16, @intCast(mant >> 13));
}

// TODO: flagging this for when we reconsider this file
pub const AdError = pr.BuildError || error{ UnsupportedEqn, UnsupportedDType };
