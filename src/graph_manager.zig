const std = @import("std");
const zg = @import("zigrad.zig");
const settings = zg.settings;
const log = std.log.scoped(.zg_graphmanager);

/// Manages the overall graph, allows for a more memory efficient abstraction
/// where the data structures used for traversing the graph during backprop
/// can be managed independently and reused across training steps
pub fn GraphManager(comptime T: type) type {
    return struct {
        const Self = @This();
        allocator: std.mem.Allocator,
        sorted_nodes: std.ArrayList(*const T),
        visited_nodes: std.AutoHashMap(*const T, void),
        eager_teardown: bool,

        pub const GraphOpts = struct {
            /// Setting this means you _really_ know what you are doing.
            eager_teardown: bool = false,
        };

        pub fn init(allocator: std.mem.Allocator, opts: GraphOpts) Self {
            return Self{
                .allocator = allocator,
                .sorted_nodes = std.ArrayList(*const T).init(allocator),
                .visited_nodes = std.AutoHashMap(*const T, void).init(allocator),
                .eager_teardown = opts.eager_teardown,
            };
        }

        pub fn deinit(self: *Self) void {
            self.sorted_nodes.deinit();
            self.visited_nodes.deinit();
            self.* = undefined;
        }

        fn topo(self: *Self, node: *const T) void {
            const gopr = self.visited_nodes.getOrPut(node) catch unreachable;
            if (!gopr.found_existing) {
                if (node.children) |children| {
                    for (children) |child| {
                        self.topo(child);
                    }
                }
                self.sorted_nodes.append(node) catch unreachable;
            }
        }

        // Must init grad on root node before backprop
        pub fn backward(self: *Self, node: *const T, alloc: std.mem.Allocator) !void {
            self.sorted_nodes.clearRetainingCapacity();
            self.visited_nodes.clearRetainingCapacity();

            self.topo(node);

            const nodes = self.sorted_nodes.items;

            for (0..nodes.len) |i| {
                var curr_node = nodes[nodes.len - i - 1];
                if (curr_node.requiresGrad()) {
                    try curr_node.backward(alloc);
                    // if eager_teardown, immediately destroy node. note that deinit is designed to not cascade recursively,
                    // it just destroys the current tensor and not the children
                    if (!curr_node.acquired and self.eager_teardown) @constCast(curr_node).deinit();
                } else {
                    log.debug("Skipping node {?s}", .{node.label});
                }
            }
        }
    };
}
