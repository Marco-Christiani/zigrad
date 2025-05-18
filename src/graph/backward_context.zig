const std = @import("std");
const builtin = @import("builtin");
const Node

const zg = @import("zigrad.zig");

const debug: bool = (builtin.mode == .Debug);

//////////////////////////////////////
//////////////////////////////////////

pub fn type_id(T: type) usize {
    const Context = struct {
        const held = T;
        var id: u8 = 0;
    };
    return @intFromPtr(&Context.id);
}

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
                allocator.destroy(@ptrCast(@alignCast(_ptr)));
            }
        }.free;
        return .{ .held = ptr, .free = free };
    }
    fn deinit(self: *ClosurePointer, allocator: std.mem.Allocator) void {
        self.free(self.held, allocator);
        self.* = undefined;
    }
};

pub fn BackwardChildren(ChildType: type) type {
    return struct {
        const Self = @This();
        pub const ChildIterator = struct {};
        pub const capacity: u64 = zg.settings.backward_children_capacity;
        buffer: [capacity]ChildType = undefined,
        versions: if (debug) [capacity]u8 else void = undefined,
        len: usize = 0,

        pub fn init(children: []const ChildType) Self {
            std.debug.assert(Self.capacity >= children.len);
            var self: Self = .{};
            self.len = children.len;
            @memcpy(self.buffer[0..self.len], children);
            if (comptime debug) {
                for (children, 0..) |child, i| self.versions[i] = child._version;
            }
            return self;
        }

        pub fn get(self: *const Self, i: usize) ChildType {
            std.debug.assert(i < self.len);
            if (comptime debug) {
                const old_version = self.versions[i];
                const new_version = self.buffer[i]._version;
                // TODO: use @src at some point? Would be more helpful...
                if (self.versions[i] != self.buffer[i]._version) {
                    std.debug.panic("Version mismatch for {s}, {}->{}", .{
                        self.buffer[i].get_label() orelse "<unknown>",
                        old_version,
                        new_version,
                    });
                }
            }
            return self.buffer[i];
        }

        pub fn get_bwd(self: *const Self, i: usize) ?ChildType {
            return if (self.get(i).requires_grad()) self.buffer[i] else null;
        }

        /// Remove all elements from the slice.
        pub fn clear(self: *Self) void {
            self.len = 0;
        }
    };
}

pub fn Iterator(ContextType: type) type {
    return struct {
        const Self = @This();
        node: ?*ContextType,

        pub fn next(self: *Self) ?*ContextType {
            defer if (self.node) |node| {
                self.node = node.next;
            };
            return self.node;
        }
    };
}

pub fn BackwardContext(primary_type: type) type {
    return struct {
        const Self = @This();
        pub const PrimaryType = primary_type;
        pub const ContextPtr = ?*anyopaque;
        pub const ContextFunction = *const fn (*Self, *PrimaryType) anyerror!void;
        pub const Children = BackwardChildren(*PrimaryType);
        pub const SmallBuffer = [4]usize;

        debug_id: if (debug) usize else void,
        callable: ContextFunction,
        children: Children,
        persist: bool = false,
        storage: union(enum) {
            none: void,
            buf: SmallBuffer,
            ptr: ClosurePointer,
            ref: *anyopaque,
        } = .none,
        next: ?*Self = null,

        pub fn init(
            CtxType: type,
            context: CtxType,
            allocator: std.mem.Allocator,
            config: struct {
                children: ?[]const *PrimaryType = null,
                persist: bool = false,
            },
        ) !Self {
            const is_ref = (@typeInfo(CtxType) == .pointer);
            const ArgType = if (is_ref) std.meta.Child(CtxType) else CtxType;

            if (!@hasDecl(ArgType, "callback")) {
                @compileError(@typeName(ArgType) ++ " does not have a 'callback' method");
            }

            const callback = struct {
                pub fn wrapper(_self: *Self, x: *PrimaryType) anyerror!void {
                    switch (comptime arity(ArgType.callback)) {
                        2 => {
                            try ArgType.callback(x, &_self.children);
                        },
                        3 => {
                            try ArgType.callback(x, &_self.children, _self.cast(ArgType));
                        },
                        else => {
                            @compileLog(@typeName(ArgType));
                            @compileError("backward callback must have artiy within [2,3]");
                        },
                    }
                }
            }.wrapper;

            const children: Children = if (config.children) |c| Children.init(c) else .{};

            if (is_ref) {
                return .{
                    .debug_id = if (debug) type_id(ArgType) else {},
                    .callable = callback,
                    .children = children,
                    .storage = .{ .ref = context },
                    .persist = config.persist,
                };
            }

            var tmp: Self = .{
                .debug_id = if (debug) type_id(ArgType) else {},
                .callable = callback,
                .children = children,
                .persist = config.persist,
            };

            const arg_size = @sizeOf(ArgType);

            if (comptime arg_size == 0) {
                return tmp;
            }

            if (comptime arg_size <= @sizeOf(SmallBuffer)) {
                tmp.storage = .{ .buf = undefined };
                const dst = std.mem.sliceAsBytes(&tmp.storage.buf);
                const src = std.mem.asBytes(&context);
                @memcpy(dst[0..arg_size], src);
            } else {
                const ptr = try allocator.create(ArgType);
                ptr.* = context;
                tmp.storage = .{ .ptr = ClosurePointer.init(ArgType, ptr) };
            }

            return tmp;
        }

        pub fn prepend(root: *Self, new_root: Self, allocator: std.mem.Allocator) !void {
            const next_ptr = try allocator.create(Self);
            next_ptr.* = root.*;
            root.* = new_root;
            root.next = next_ptr;
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            if (self.next) |next| {
                next.deinit(allocator);
                allocator.destroy(next);
            }
            self.release(allocator);
        }

        pub fn iterator(self: *Self) Iterator(Self) {
            return .{ .node = self };
        }

        pub const ChildIterator = struct {
            node: ?*Self,
            index: usize = 0,

            pub fn next(self: *ChildIterator) ?*PrimaryType {
                if (self.node) |node| {
                    if (self.index < node.children.len) {
                        defer self.index += 1;
                        return node.children.buffer[self.index];
                    }
                    self.node = node.next;
                    self.index = 0;
                    return self.next();
                }
                return null;
            }
        };

        pub fn call(self: *Self, tensor: *PrimaryType) anyerror!void {
            var _this: ?*Self = self;
            while (_this) |this| : (_this = this.next) {
                try this.callable(this, tensor);
            }
        }

        pub fn release(self: *Self, allocator: std.mem.Allocator) void {
            if (self.storage == .ptr) {
                self.storage.ptr.deinit(allocator);
            }
            self.storage = .none;
        }

        pub fn cast(self: *Self, T: type) *T {
            if (comptime @TypeOf(self.debug_id) != void) {
                std.debug.assert(self.debug_id == type_id(T));
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
}
