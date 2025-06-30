const std = @import("std");
const ArenaUnmanaged = @import("../allocators/arena_unmanaged.zig");
const zg = @import("../zigrad.zig");

/// Sharable graph memory interface for nodes.
///
/// # ADR
///
/// - This abstraction allows us to separately manage leaf and internal tensor shells when building computation graphs.
/// - Leaves and internal nodes have different lifetimes.
const Self = @This();

allocator: std.mem.Allocator,
arena: ArenaUnmanaged = .empty,
map: std.AutoArrayHashMapUnmanaged(usize, Chain) = .empty,

pub fn init(allocator: std.mem.Allocator) Self {
    return .{ .allocator = allocator };
}

pub fn deinit(self: *Self) void {
    self.arena.deinit(self.allocator);
    self.map.deinit(self.allocator);
}

pub fn create_node(self: *Self, T: type) !*T {
    const U = Chain.Intrusive(T);

    cache_block: {
        const chain = self.map.getPtr(@sizeOf(U)) orelse break :cache_block;
        const link = chain.pop() orelse break :cache_block;
        const intrusive: *U = @alignCast(@fieldParentPtr("link", link));
        return &intrusive.data;
    }
    const intrusive = try self.arena.create(self.allocator, U);
    return &intrusive.data;
}

pub fn destroy_node(self: *Self, ptr: anytype) void {
    const U = Chain.Intrusive(std.meta.Child(@TypeOf(ptr)));
    const intrusive: *U = @constCast(@alignCast(@fieldParentPtr("data", ptr)));

    const gop = self.map.getOrPut(self.allocator, @sizeOf(U)) catch
        return self.arena.destroy(intrusive);

    if (!gop.found_existing)
        gop.value_ptr.* = .{};

    gop.value_ptr.prepend(&intrusive.link);
}

pub fn promote(self: *Self) *zg.Graph {
    return @alignCast(@fieldParentPtr("builder", self));
}

const Chain = struct {
    fn Intrusive(T: type) type {
        return struct {
            link: Link,
            data: T,
        };
    }
    const Link = struct {
        next: ?*Link = null,
    };
    head: ?*Link = null,

    pub fn prepend(self: *Chain, link: *Link) void {
        link.next = self.head;
        self.head = link;
    }

    pub fn pop(self: *Chain) ?*Link {
        const head = self.head orelse return null;
        self.head = head.next;
        return head;
    }
};
