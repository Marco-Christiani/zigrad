const std = @import("std");
const Allocator = std.mem.Allocator;
const ArenaUnmanaged = @import("../allocators.zig").ArenaUnmanaged;
const allocator = std.heap.smp_allocator;

const Self = @This();

const Callback = struct {
    fptr: *const fn (*anyopaque, *anyopaque) void,
    ctx: *anyopaque,
    args: *anyopaque,
    pub fn call(self: *const Callback) void {
        self.fptr(self.ctx, self.args);
    }
};

pub const Segment = struct {
    state: *Self,
    head: usize,
    tail: usize,
    pub fn run(self: *const Segment) void {
        for (self.state.callbacks.items[self.head..self.tail]) |*cb|
            cb.call();
    }
};

pub const empty: Self = .{
    .head = null,
    .arena = .empty,
    .callbacks = .empty,
};

head: ?usize = null,
arena: ArenaUnmanaged = .empty,
callbacks: std.ArrayListUnmanaged(Callback) = .empty,

pub fn deinit(self: *Self) void {
    self.arena.deinit(allocator);
    self.callbacks.deinit(allocator);
    self.* = undefined;
}

pub fn reset(self: *Self) void {
    _ = self.arena.reset(allocator, .retain_capacity);
    self.callbacks.clearRetainingCapacity();
}

pub fn is_open(self: *const Self) bool {
    return self.head != null;
}

pub fn open(self: *Self) void {
    if (self.is_open())
        @panic("Cannot open a capture while another is in progress.");

    self.head = self.callbacks.items.len;
}

pub fn close(self: *Self) Segment {
    defer self.head = null;
    return .{
        .state = self,
        .head = self.head orelse @panic("No capture in progress."),
        .tail = self.callbacks.items.len,
    };
}

pub fn record(self: *Self, container_ptr: anytype, opspec_struct: anytype) void {
    const CT = std.meta.Child(@TypeOf(container_ptr));
    const OP = @TypeOf(opspec_struct);

    const fptr = struct {
        pub fn call(ctx: *anyopaque, args: *anyopaque) void {
            const tctx: *CT = @ptrCast(@alignCast(ctx));
            const targs: *OP = @ptrCast(@alignCast(args));
            @field(CT, OP.__name__)(tctx, OP.__type__, targs.*);
        }
    }.call;

    const args = self.arena.create(allocator, OP) catch
        @panic("Failed to allocate when capturing graph.");

    args.* = opspec_struct;

    self.callbacks.append(allocator, .{ .fptr = fptr, .ctx = container_ptr, .args = args }) catch
        @panic("Failled to append callback while capturing.");
}
