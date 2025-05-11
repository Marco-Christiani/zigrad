const std = @import("std");
const GraphManager = @import("../graph_manager.zig");
const NodeType = GraphManager.NodeType;
const NodeHeap = GraphManager.NodeHeap;
const DeviceReference = @import("../zigrad.zig").DeviceReference;

const LABEL_SIZE: usize = 32;
pub const Label = std.BoundedArray(u8, LABEL_SIZE);

pub fn as_label(slice: ?[]const u8) Label {
    const l = slice orelse return .{};
    return Label.fromSlice(l) catch @panic("Label size is too large - max 32 characters");
}

/// User facing config
pub const TensorConfig = struct {
    device: DeviceReference,
    heap: NodeHeap,
    requires_grad: bool = false,
    acquired: bool = false,
    attached: bool = true,
    label: ?[]const u8 = null,
};

pub const TensorFlags = struct {
    const BitSet = std.bit_set.IntegerBitSet(Flags.count());

    pub const empty: TensorFlags = .{
        .bitset = .initEmpty(),
    };

    pub const Flags = enum {
        /// Determine if a tensor is a leaf or internal node
        node_type,
        /// Marking a tensor as acquired signals to the
        /// backwards process that this tensor should
        /// not be freed. Set by using the "acquire" and
        /// "release" functions.
        acquired,
        /// An attached tensor can be traversed through
        /// in the backward process. If the tensor is
        /// unattached, the reversal process will not
        /// continue through that tensor. Set by using
        /// the "attach" and "detach" functions.
        attached,
        /// The requires grad field tells the backwards
        /// process if it ought to initialize a gradient.
        /// This field should not be used directly
        /// because runtime gradients may be deactivated.
        /// Use the "requires_grad" function instead.
        requires_grad,
        /// Determine if a tensor has been cleared
        cleared,

        pub fn count() usize {
            return std.meta.fields(Flags).len;
        }
    };
    bitset: BitSet,

    pub fn init(
        node_type: NodeType,
        config: struct {
            requires_grad: bool,
            acquired: bool,
            attached: bool,
            cleared: bool,
        },
    ) TensorFlags {
        var self: TensorFlags = .empty;
        self.set(.node_type, node_type == .leaf);
        comptime var field_count: usize = 1;
        inline for (std.meta.fields(@TypeOf(config))) |field| {
            const tag = comptime std.meta.stringToEnum(Flags, field.name) orelse continue;

            field_count += 1;

            if (field.type != bool)
                @compileError("Flags must be boolean types.");

            self.set(tag, @field(config, field.name));
        }
        if (comptime field_count != Flags.count())
            @compileError("Missing flags in config struct.");

        return self;
    }

    pub fn set(self: *TensorFlags, flag: Flags, value: bool) void {
        self.bitset.setValue(@intFromEnum(flag), value);
    }
    pub fn get(self: TensorFlags, flag: Flags) bool {
        return self.bitset.isSet(@intFromEnum(flag));
    }
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
