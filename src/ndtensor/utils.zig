const std = @import("std");
const Graph = @import("../graph.zig");
const NodeType = Graph.NodeType;
const zg = @import("../zigrad.zig");
const DeviceReference = zg.DeviceReference;

const LABEL_SIZE: usize = zg.settings.label_capacity;
pub const Label = std.BoundedArray(u8, LABEL_SIZE);

pub fn as_label(slice: ?[]const u8) Label {
    const l = slice orelse return .{};
    return Label.fromSlice(l) catch @panic(std.fmt.comptimePrint("Label size is too large - max {d} characters", .{LABEL_SIZE}));
}

/// User facing config
pub const TensorConfig = struct {
    device: DeviceReference,
    graph: *Graph,
    requires_grad: bool = false,
    acquired: bool = false,
    attached: bool = true,
    label: ?[]const u8 = null,
};

pub const MaxAlongOptions = struct {
    dim: usize,
    keep_dims: bool = false,
};

/// These Op enums are not really functional but are convenient for the user (i.e. when traversing the graph or debugging)
/// There are many more ops supported than this. This may be deprecated by v1.
/// Zigrad does not switch on the op during backward and ops can be added without existing the existing enum values.
pub const Op = enum {
    ADD,
    SUB,
    MUL,
    DIV,
    POW, // TODO:
    TANH, // TODO:
    MATMUL_AB,
    MATMUL_AtB,
    MATMUL_ABt,
    MATMUL_AtBt,
    DOT,
    MATVEC,
    SUM,
    RESHAPE,
    TRANSPOSE,
    MAX,
    EXP,
    TRANSFER,
    CLAMP,

    pub fn matmul_tag(trans_a: bool, trans_b: bool) Op {
        return if (!trans_a and !trans_b)
            .MATMUL_AB
        else if (trans_a and !trans_b)
            .MATMUL_AtB
        else if (!trans_a and trans_b)
            .MATMUL_ABt
        else
            .MATMUL_AtBt;
    }
};
