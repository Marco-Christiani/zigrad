//! Inputs, outputs, and operations in a PR region.
//!
//! `describe` computes the member operations, incoming values, and externally
//!  used results for one region in a function.
const std = @import("std");
const pr = @import("../pr.zig");

/// A region's members, inputs, and externally used outputs.
///
/// `deinit` frees the slices with the allocator passed to `describe`.
pub const RegionView = struct {
    name: []const u8,
    ops: []const *pr.Op,
    inputs: []const *pr.Var,
    outputs: []const *pr.Var,

    pub fn op_count(self: RegionView) usize {
        return self.ops.len;
    }

    pub fn deinit(self: RegionView, allocator: std.mem.Allocator) void {
        allocator.free(self.ops);
        allocator.free(self.inputs);
        allocator.free(self.outputs);
    }
};

/// Describes `region` within `func`.
///
/// Inputs are operands whose defining operation is outside the region or absent.
///
/// Outputs are region results used outside the region or returned by `func`.
pub fn describe(
    allocator: std.mem.Allocator,
    func: pr.Function,
    region: pr.Region,
) std.mem.Allocator.Error!RegionView {
    var region_ops_list = try std.ArrayList(*pr.Op).initCapacity(allocator, region.op_ids.len);
    defer region_ops_list.deinit(allocator);
    for (region.op_ids) |op_id| {
        const op = func.op_by_id(op_id) orelse continue;
        try region_ops_list.append(allocator, op);
    }
    const region_ops = try region_ops_list.toOwnedSlice(allocator);
    errdefer allocator.free(region_ops);

    const seen = try allocator.alloc(bool, func.var_count);
    defer allocator.free(seen);

    @memset(seen, false);
    var inputs_list = std.ArrayList(*pr.Var).empty;
    defer inputs_list.deinit(allocator);
    for (region_ops) |op| {
        for (op.inputs) |operand| {
            const v = operand.value;
            if (seen[v.id]) continue;
            seen[v.id] = true;
            const def = v.defining_op orelse {
                try inputs_list.append(allocator, v);
                continue;
            };
            if (!op_in_slice(def, region_ops)) try inputs_list.append(allocator, v);
        }
    }

    @memset(seen, false);
    var outputs_list = std.ArrayList(*pr.Var).empty;
    defer outputs_list.deinit(allocator);
    for (region_ops) |op| {
        for (op.outputs) |out_var| {
            if (seen[out_var.id]) continue;
            if (has_use_outside_view(out_var, region_ops) or var_is_returned(out_var, func)) {
                seen[out_var.id] = true;
                try outputs_list.append(allocator, out_var);
            }
        }
    }

    return .{
        .name = region.name,
        .ops = region_ops,
        .inputs = try inputs_list.toOwnedSlice(allocator),
        .outputs = try outputs_list.toOwnedSlice(allocator),
    };
}

fn op_in_slice(op: *const pr.Op, ops: []const *pr.Op) bool {
    for (ops) |candidate| {
        if (candidate == op) return true;
    }
    return false;
}

fn has_use_outside_view(value: *const pr.Var, region_ops: []const *pr.Op) bool {
    var use = value.first_use;
    while (use) |operand| : (use = operand.next) {
        if (!op_in_slice(operand.owner, region_ops)) return true;
    }
    return false;
}

fn var_is_returned(value: *const pr.Var, func: pr.Function) bool {
    for (func.returns) |returned| {
        if (returned == value) return true;
    }
    return false;
}
