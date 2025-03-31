const std = @import("std");
const builtin = @import("builtin");
const zg = @import("zigrad.zig");
const debug: bool = (builtin.mode == .Debug);
const DeviceReference = zg.DeviceReference;

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

pub fn BackwardContext(PrimaryType: type) type {
    return struct {
        const Self = @This();
        const ContextPtr = ?*anyopaque;
        const ContextFunction = *const fn (*Self, *PrimaryType, DeviceReference) anyerror!void;
        // small-buffer-optimized context object
        const SmallBuffer = [4]usize;

        debug_id: if (debug) usize else void,
        callable: ContextFunction,
        storage: union(enum) {
            none: void,
            buf: SmallBuffer,
            ptr: *anyopaque,
            ref: *anyopaque,
        } = .none,

        pub fn init(CtxType: type, context: CtxType, device: DeviceReference) !Self {
            const is_ref = (@typeInfo(CtxType) == .pointer);
            const ArgType = if (is_ref) std.meta.Child(CtxType) else CtxType;

            if (!@hasDecl(ArgType, "callback")) {
                @compileError(@typeName(ArgType) ++ " does not have a 'callback' method");
            }

            const callback = struct {
                pub fn wrapper(_self: *Self, x: *PrimaryType, _device: DeviceReference) anyerror!void {
                    switch (comptime arity(ArgType.callback)) {
                        1 => {
                            try ArgType.callback(x);
                        },
                        2 => {
                            try ArgType.callback(x, _self.cast(ArgType));
                            _self.release(ArgType, _device);
                        },
                        else => {
                            @compileError("backward callback must have artiy within [1,2]");
                        },
                    }
                }
            }.wrapper;

            if (is_ref) {
                return .{
                    .callable = callback,
                    .storage = .{ .ref = context },
                    .debug_id = if (debug) type_id(ArgType) else {},
                };
            }

            var tmp: Self = .{
                .callable = callback,
                .debug_id = if (debug) type_id(ArgType) else {},
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
                const ptr = try device.allocator.create(ArgType);
                ptr.* = context;
                tmp.storage = .{ .ptr = ptr };
            }

            return tmp;
        }

        pub fn deinit(self: *Self) void {
            std.debug.assert(self.storage == .none);
            self.* = undefined;
        }

        pub fn call(self: *Self, tensor: *PrimaryType, device: DeviceReference) anyerror!void {
            return self.callable(self, tensor, device);
        }

        pub fn release(self: *Self, T: type, device: DeviceReference) void {
            if (self.storage == .ptr) {
                device.allocator.destroy(self.cast(T));
            }
            self.storage = .none;
        }

        pub fn cast(self: *Self, T: type) *T {
            if (comptime @TypeOf(self.debug_id) != void) {
                std.debug.assert(self.debug_id == type_id(T));
            }

            const address = switch (self.storage) {
                .buf => |*buf| @intFromPtr(buf),
                .ptr => |ptr| @intFromPtr(ptr),
                .ref => |ref| @intFromPtr(ref),
                .none => @panic("Cannot cast from empty storage."),
            };
            return @ptrFromInt(address);
        }
    };
}
