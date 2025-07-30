const std = @import("std");
const Allocator = std.mem.Allocator;

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

pub const ClosurePointer = struct {
    type_id: TypeID,
    held: *anyopaque,
    free: *const fn (*anyopaque) void,
    pub fn init(ptr: anytype, cleanup: bool) ClosurePointer {
        const T = std.meta.Child(@TypeOf(ptr));

        const free = struct {
            pub fn impl(_ptr: *anyopaque) void {
                const tptr: *T = @ptrCast(@alignCast(_ptr));
                if (comptime @hasDecl(T, "deinit")) {
                    tptr.deinit();
                }
            }
        }.impl;

        const pass = struct {
            pub fn impl(_: *anyopaque) void {}
        }.impl;

        return .{
            .type_id = TypeID.init(T),
            .held = ptr,
            .free = if (cleanup) free else pass,
        };
    }

    pub fn deinit(self: *ClosurePointer) void {
        self.free(self.held);
        self.* = undefined;
    }

    pub fn cast(self: *const ClosurePointer, T: type) *T {
        std.debug.assert(self.type_id == TypeID.init(T));
        return @ptrCast(@alignCast(self.held));
    }
};
