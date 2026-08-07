//! Context types for op implementations and helpers.
const std = @import("std");
const pr = @import("../pr.zig");

/// Context passed to AD forward/backward functions.
///
/// Used for both VJP (reverse-mode) and JVP (forward-mode) transforms.
/// For VJP: `tangent_map` is null, `cot_map` holds cotangent accumulation.
/// For JVP: `cot_map` is null, `tangent_map` holds tangent propagation.
///
/// Maps are indexed by `Var.id` and sized by `func.var_count`.
pub const AdContext = struct {
    builder: *pr.FunctionBuilder,
    primal_map: []?*pr.Var,
    /// In reverse-mode (VJP) holds cotangents.
    cot_map: ?[]?*pr.Var,
    /// In forward-mode (JVP) holds tangents.
    tangent_map: ?[]?*pr.Var,
    allocator: std.mem.Allocator,

    pub fn get_primal(self: AdContext, v: *const pr.Var) ?*pr.Var {
        if (v.id >= self.primal_map.len) return null;
        return self.primal_map[v.id];
    }

    pub fn set_primal(self: AdContext, v: *const pr.Var, value: *pr.Var) void {
        self.primal_map[v.id] = value;
    }

    pub fn get_cot(self: AdContext, v: *const pr.Var) ?*pr.Var {
        const cmap = self.cot_map orelse return null;
        if (v.id >= cmap.len) return null;
        return cmap[v.id];
    }

    pub fn add_cot(self: AdContext, v: *const pr.Var, new_cot: *pr.Var) pr.BuildError!void {
        const cmap = self.cot_map orelse return;
        const idx: usize = v.id;
        if (cmap[idx]) |existing| {
            cmap[idx] = try self.builder.add(existing, new_cot);
        } else {
            cmap[idx] = new_cot;
        }
    }

    pub fn get_tangent(self: AdContext, v: *const pr.Var) ?*pr.Var {
        const tmap = self.tangent_map orelse return null;
        if (v.id >= tmap.len) return null;
        return tmap[v.id];
    }

    pub fn set_tangent(self: AdContext, v: *const pr.Var, value: *pr.Var) void {
        self.tangent_map.?[v.id] = value;
    }
};

pub const Writer = std.Io.Writer;
pub const FormatError = Writer.Error;

pub fn same_tensor_type(a: pr.Tensor, b: pr.Tensor) bool {
    if (a.dtype != b.dtype) return false;
    if (a.shape.rank() != b.shape.rank()) return false;
    return std.mem.eql(i64, a.shape.dims, b.shape.dims);
}

pub const AdError = pr.BuildError || error{
    /// An op cannot reproduce a primal or propagate the active AD value.
    UnsupportedEqn,
    UnsupportedDType,
    /// An index in `VjpOpts.wrt` is out of range for the source function.
    WrtIndexOutOfRange,
    /// A harvested input has no cotangent, or a harvested output has no tangent.
    ///
    /// This usually means a primal-chain op lacks its mode-specific propagation
    ///  handler, or the harvested variable is orphaned.
    MissingDual,
};
