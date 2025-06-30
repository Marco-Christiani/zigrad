const std = @import("std");
const builtin = @import("builtin");
const debug: bool = (builtin.mode == .Debug);
const Graph = @import("../graph.zig");
const zg = @import("../zigrad.zig");

const Node = @This();

/// Pointer to the parent graph's body. Can be promoted
/// to retrieve the original parent graph.
gb: *Graph.Builder,
/// Bitset for tensor flags (see ndtensor/utils.zig)
flags: Flags,
/// versioning ensures that inplace ops do not
/// interfere with the backward pass. Versioning
/// is only enabled in debug mode and only
/// matters if you are doing a backward pass.
version: u8 = 0,
/// Type ID to ensure that casting always targets
/// the original type.
type_id: TypeID,
/// Optional label for naming nodes in the graph.
/// These labels are unmanaged and should not be
/// used to store allocated strings.
label: Label,
/// Collection of callback functions - if we get more
/// of these then consier moving to virtual table.
callbacks: struct {
    /// Opaque object that acts like a closure for
    /// backwards function calls. The backwards context
    /// can allocate state if the arguments provided
    /// exceed its internal buffer.
    bwd: ?BackwardContext = null,
    /// deleter function that calls deinit through
    /// the parent node type. Always the same call
    /// and used for graph teardowns.
    del: *const fn (*Node) void,
},

pub fn init(
    NodeParentType: type,
    builder: *Graph.Builder,
    bwd_context: ?BackwardContext,
    label_bytes: ?[]const u8,
    flag_config: Flags.Config,
) Node {
    return .{
        .gb = builder,
        .flags = Flags.init(flag_config),
        .type_id = TypeID.init(NodeParentType),
        .version = 0,
        .label = as_label(label_bytes),
        .callbacks = .{
            .bwd = bwd_context,
            .del = struct {
                pub fn call(node: *Node) void {
                    node.upcast(NodeParentType).deinit();
                }
            }.call,
        },
    };
}

pub fn deinit(self: *Node) void {
    return self.callbacks.del(self);
}

/// Clear any allocated context state.
/// Should be called by host object that
/// the node is intruding upon.
pub fn deactivate(self: *Node) void {
    if (self.callbacks.bwd) |*bwd| {
        bwd.deinit(self.gb.allocator);
        self.callbacks.bwd = null;
    }
    self.flags.set(.active, false);
}

pub fn backward(self: *Node) anyerror!void {
    const _cb = &(self.callbacks.bwd orelse return);
    return _cb.call(self);
}

pub fn get_label(self: *const Node) ?[]const u8 {
    return if (self.label.len > 0) self.label.slice() else null;
}

pub fn set_label(self: *Node, new_label: []const u8) void {
    self.label = as_label(new_label);
}

pub fn upcast(self: *Node, T: type) *T {
    std.debug.assert(self.type_id == TypeID.init(T));
    return @alignCast(@fieldParentPtr("node", self));
}

pub fn child_iterator(self: *Node) ?ChildIterator {
    const ctx = &(self.callbacks.bwd orelse return null);
    if (ctx.children.len == 0 and ctx.next == null) return null;
    return .{ .ctx = ctx };
}

///////////////////////////////////////////////////////
// Flag Helpers ///////////////////////////////////////
//
// A node instance should not set flags - this behavior
// should be determined by the host object that the
// node is intruding upon. The API is thus read-only.

pub fn requires_grad(self: *const Node) bool {
    return self.flags.get(.requires_grad) and zg.runtime.grad_enabled;
}

pub fn acquired(self: *const Node) bool {
    return self.flags.get(.acquired);
}

pub fn active(self: *const Node) bool {
    return self.flags.get(.active);
}

pub fn attached(self: *const Node) bool {
    return self.flags.get(.attached);
}

//////////////////////////////////////
//////////////////////////////////////

pub const Flags = struct {
    const BitSet = std.bit_set.IntegerBitSet(Values.count());

    pub const Config = struct {
        requires_grad: bool,
        acquired: bool,
        attached: bool,
    };

    pub const empty: Flags = .{
        .bitset = .initEmpty(),
    };

    pub const Values = enum {
        /// Marking a tensor as acquired signals to the
        /// backwards process that this tensor should
        /// not be freed. Set by using the "acquire" and
        /// "release" functions.
        acquired,
        /// This field describes whether `deinit` has been
        /// already called on the parent object.
        /// This makes it easier to handle releasing memory
        /// on errors if intermediate tensors were freed.
        /// The parent object is responsible for setting
        /// this field. It is undefined behavior to use an
        /// object that is attched to an inactive node.
        active,
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
        /// Is set to true if this node is the child of
        /// an operation that requires a gradient. Handles
        /// the case where a node itself does not need a
        /// gradient, but it is not safe to free this node
        /// until other tensors that depend upon it have
        /// collected theirs.
        grad_operand,

        pub fn count() usize {
            return std.meta.fields(Values).len;
        }
    };
    bitset: BitSet,

    pub fn init(config: Config) Flags {
        var self: Flags = .empty;
        self.set(.active, true);
        self.set(.grad_operand, false);
        comptime var field_count: usize = 2;
        inline for (std.meta.fields(@TypeOf(config))) |field| {
            const tag = comptime std.meta.stringToEnum(Values, field.name) orelse continue;

            field_count += 1;

            if (field.type != bool)
                @compileError("Flags must be boolean types.");

            self.set(tag, @field(config, field.name));
        }
        if (comptime field_count != Values.count())
            @compileError("Missing flags in config struct.");

        return self;
    }

    pub fn set(self: *Flags, flag: Values, value: bool) void {
        self.bitset.setValue(@intFromEnum(flag), value);
    }
    pub fn get(self: Flags, flag: Values) bool {
        return self.bitset.isSet(@intFromEnum(flag));
    }
};

pub const Children = struct {
    pub const capacity: u64 = zg.settings.backward_children_capacity;
    buffer: [capacity]*Node = undefined,
    versions: if (debug) [capacity]u8 else void = undefined,
    len: usize = 0,

    pub fn from_slice(slice: []const *Node) Children {
        std.debug.assert(Children.capacity >= slice.len);
        var self: Children = .{};
        self.len = slice.len;
        @memcpy(self.buffer[0..self.len], slice);
        if (comptime debug) {
            for (slice, 0..) |child, i| self.versions[i] = child.version;
        }
        return self;
    }

    fn get(self: *const Children, i: usize) *Node {
        std.debug.assert(i < self.len);
        if (comptime debug) {
            const old_version = self.versions[i];
            const new_version = self.buffer[i].version;
            // TODO: use @src at some point? Would be more helpful...
            if (self.versions[i] != self.buffer[i].version) {
                std.debug.panic("Version mismatch for {s}, {}->{}", .{
                    self.buffer[i].get_label() orelse "<unknown>",
                    old_version,
                    new_version,
                });
            }
        }
        return self.buffer[i];
    }

    pub fn get_upcast(self: *const Children, T: type, i: usize) *T {
        return self.get(i).upcast(T);
    }

    pub fn get_bwd_upcast(self: *const Children, T: type, i: usize) ?*T {
        return if (self.get(i).requires_grad())
            self.get_upcast(T, i)
        else
            null;
    }
};

pub const ChildIterator = struct {
    ctx: ?*BackwardContext,
    index: usize = 0,

    pub fn next(self: *ChildIterator) ?*Node {
        if (self.ctx) |ctx| {
            if (self.index < ctx.children.len) {
                defer self.index += 1;
                return ctx.children.buffer[self.index];
            }
            self.ctx = ctx.next;
            self.index = 0;
            return self.next();
        }
        return null;
    }
};

pub const BackwardContext = struct {
    const SmallBuffer = [4]usize;
    children: Children,
    callable: *const fn (*BackwardContext, *Node) anyerror!void,
    type_id: if (debug) TypeID else void,
    persist: bool = false,
    storage: union(enum) {
        none: void,
        buf: SmallBuffer,
        ptr: ClosurePointer,
        ref: *anyopaque,
    } = .none,
    next: ?*BackwardContext = null,

    pub fn init(
        NodeParentType: type,
        BwdClosureType: type,
        allocator: std.mem.Allocator,
        bwd_instance: BwdClosureType,
        bwd_children: []const *Node,
    ) !BackwardContext {
        const is_ref = (@typeInfo(BwdClosureType) == .pointer);
        const BwdClosureDeep = if (is_ref) std.meta.Child(BwdClosureType) else BwdClosureType;

        if (!@hasDecl(BwdClosureType, "backward"))
            @compileError(@typeName(BwdClosureType) ++ " does not have a 'backward' method");

        const callback = struct {
            pub fn wrapper(_self: *BackwardContext, x: *Node) anyerror!void {
                switch (comptime arity(BwdClosureDeep.backward)) {
                    2 => try BwdClosureDeep.backward(x.upcast(NodeParentType), &_self.children),
                    3 => try BwdClosureDeep.backward(x.upcast(NodeParentType), &_self.children, _self.cast(BwdClosureDeep)),
                    else => @compileError("backward callback must have artiy within [2,3]"),
                }
            }
        }.wrapper;

        if (is_ref) {
            return .{
                .type_id = if (debug) TypeID.init(BwdClosureDeep) else {},
                .callable = callback,
                .children = .from_slice(bwd_children),
                .storage = .{ .ref = bwd_instance },
            };
        }

        var tmp: BackwardContext = .{
            .type_id = if (debug) TypeID.init(BwdClosureDeep) else {},
            .children = .from_slice(bwd_children),
            .callable = callback,
        };

        const arg_size = @sizeOf(BwdClosureDeep);

        if (comptime arg_size == 0) {
            return tmp;
        }

        if (comptime arg_size <= @sizeOf(SmallBuffer)) {
            tmp.storage = .{ .buf = undefined };
            const dst = std.mem.sliceAsBytes(&tmp.storage.buf);
            const src = std.mem.asBytes(&bwd_instance);
            @memcpy(dst[0..arg_size], src);
        } else {
            const ptr = try allocator.create(BwdClosureDeep);
            ptr.* = bwd_instance;
            tmp.storage = .{ .ptr = ClosurePointer.init(BwdClosureDeep, ptr) };
        }

        return tmp;
    }

    pub fn prepend(root: *BackwardContext, new_root: BackwardContext, allocator: std.mem.Allocator) !void {
        const next_ptr = try allocator.create(BackwardContext);
        next_ptr.* = root.*;
        root.* = new_root;
        root.next = next_ptr;
    }

    pub fn deinit(self: *BackwardContext, allocator: std.mem.Allocator) void {
        if (self.next) |next| {
            next.deinit(allocator);
            allocator.destroy(next);
        }
        self.release(allocator);
    }

    pub fn call(self: *BackwardContext, x: *Node) anyerror!void {
        var _this: ?*BackwardContext = self;
        while (_this) |this| : (_this = this.next) {
            try this.callable(this, x);
        }
    }

    pub fn release(self: *BackwardContext, allocator: std.mem.Allocator) void {
        if (self.storage == .ptr) {
            self.storage.ptr.deinit(allocator);
        }
        self.storage = .none;
    }

    pub fn cast(self: *BackwardContext, T: type) *T {
        if (comptime @TypeOf(self.type_id) != void) {
            std.debug.assert(self.type_id == TypeID.init(T));
        }

        const address = switch (self.storage) {
            .buf => |*buf| @intFromPtr(buf),
            .ptr => |ptr| @intFromPtr(ptr.held),
            .ref => |ref| @intFromPtr(ref),
            .none => @panic("Cannot cast from empty storage."),
        };
        return @ptrFromInt(address);
    }
};

pub const TypeID = enum(usize) {
    _,
    pub fn init(T: type) TypeID {
        const __impl__ = struct {
            const held = T;
            var id: u8 = 0;
        };
        return @enumFromInt(@intFromPtr(&__impl__.id));
    }
};

fn arity(comptime callable: anytype) usize {
    return switch (@typeInfo(@TypeOf(callable))) {
        .@"fn" => |f| f.params.len,
        else => @compileError("arity requires function type"),
    };
}

fn ContextPtrType(T: type) type {
    return switch (@typeInfo(T)) {
        .pointer => |ptr| {
            if (ptr.size != .one)
                @compileError("pointer must be singular (aka, *T)");

            return ptr.child;
        },
        else => @compileError("context argument must be a pointer type"),
    };
}

fn ContextArgType(comptime callable: anytype) type {
    switch (@typeInfo(@TypeOf(callable))) {
        .@"fn" => |f| {
            const arg_type = f.params[1].type orelse
                @compileError("backwards contexts do not support generic arguments");

            return ContextPtrType(arg_type);
        },
        else => @compileError("backward context callable must be function type"),
    }
}

const ClosurePointer = struct {
    held: *anyopaque,
    free: *const fn (*anyopaque, std.mem.Allocator) void,
    fn init(T: type, ptr: *T) ClosurePointer {
        const free = struct {
            pub fn impl(_ptr: *anyopaque, allocator: std.mem.Allocator) void {
                var self: *T = @ptrCast(@alignCast(_ptr));
                if (@hasDecl(T, "deinit")) self.deinit();
                allocator.destroy(self);
            }
        }.impl;
        return .{ .held = ptr, .free = free };
    }
    fn deinit(self: *ClosurePointer, allocator: std.mem.Allocator) void {
        self.free(self.held, allocator);
        self.* = undefined;
    }
};

const LABEL_SIZE: usize = zg.settings.label_capacity;
pub const Label = std.BoundedArray(u8, LABEL_SIZE);

pub fn as_label(slice: ?[]const u8) Label {
    const l = slice orelse return .{};
    return Label.fromSlice(l) catch @panic(std.fmt.comptimePrint("Label size is too large - max {d} characters", .{LABEL_SIZE}));
}
