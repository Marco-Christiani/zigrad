const std = @import("std");
const ArenaUnmanaged = @import("arena_unmanaged.zig");
const NodeType = @import("node.zig").NodeType;
const Self = @This();

/// Sharable graph memory interface for nodes.
///
/// # ADR
///
/// - This abstraction allows us to separately manage leaf and internal tensor shells when building computation graphs.
/// - Leaves and internal nodes have different lifetimes.
allocator: std.mem.Allocator,
/// arena for all temporary nodes created by ops
internal_node_arena: ArenaUnmanaged,

pub fn create_node(self: *Self, T: type, node_type: NodeType) !*T {
    return switch (node_type) {
        .leaf => self.allocator.create(T),
        .internal => self.internal_node_arena.create(self.allocator, T),
    };
}

pub fn destroy_node(self: *Self, node: anytype, node_type: NodeType) void {
    return switch (node_type) {
        .leaf => self.allocator.destroy(node),
        .internal => {
            // Internal nodes are kept in the arena and are not freed individually.
            // Use reset to clear internal graph nodes.
        },
    };
}
