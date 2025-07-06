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
    free: *const fn (*anyopaque, Allocator) void,
    pub fn init(ptr: anytype) ClosurePointer {
        const T = std.meta.Child(@TypeOf(ptr));
        const free = struct {
            pub fn impl(_ptr: *anyopaque, allocator: Allocator) void {
                const tptr: *T = @ptrCast(@alignCast(_ptr));
                if (comptime @hasDecl(T, "deinit")) {
                    tptr.deinit();
                }
                allocator.destroy(tptr);
            }
        }.free;
        return .{
            .type_id = TypeID.init(T),
            .held = ptr,
            .free = free,
        };
    }

    pub fn deinit(self: *ClosurePointer, allocator: Allocator) void {
        self.free(self.held, allocator);
        self.* = undefined;
    }

    pub fn cast(self: *const ClosurePointer, T: type) *T {
        std.debug.assert(self.type_id == TypeID.init(T));
        return @ptrCast(@alignCast(self.held));
    }
};
