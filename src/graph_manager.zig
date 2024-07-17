const std = @import("std");
const zg = @import("root.zig");
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
        eager_teardown: bool = false,
        grad_clip_enabled: bool = settings.grad_clip_enabled,
        grad_clip_max_norm: f32 = settings.grad_clip_max_norm,
        grad_clip_delta: f32 = settings.grad_clip_delta,

        pub const LossConfig = struct {
            eager_teardown: bool = false,
            grad_clip_enabled: bool = settings.grad_clip_enabled,
            grad_clip_max_norm: f32 = settings.grad_clip_max_norm,
            grad_clip_delta: f32 = settings.grad_clip_delta,
        };

        pub fn init(allocator: std.mem.Allocator, opts: LossConfig) Self {
            return Self{
                .allocator = allocator,
                .sorted_nodes = std.ArrayList(*const T).init(allocator),
                .visited_nodes = std.AutoHashMap(*const T, void).init(allocator),
                .grad_clip_enabled = opts.grad_clip_enabled,
                .grad_clip_max_norm = opts.grad_clip_max_norm,
                .grad_clip_delta = opts.grad_clip_delta,
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
                if (curr_node.requires_grad) {
                    log.debug("backward: {?s}", .{curr_node.label});
                    try curr_node.backward(alloc);
                    log.debug("backprop {?s} grad norm is {d} max: {d} min: {d}", .{
                        curr_node.label,
                        curr_node.grad.?.l2_norm(),
                        std.mem.max(@TypeOf(curr_node.data.data[0]), curr_node.grad.?.data),
                        std.mem.min(@TypeOf(curr_node.data.data[0]), curr_node.grad.?.data),
                    });
                    if (self.grad_clip_enabled and curr_node.requires_grad) {
                        if (curr_node.grad) |_| {
                            curr_node.clip_grad_norm_delta(.{ .max_norm = self.grad_clip_max_norm, .delta = self.grad_clip_delta });
                        }
                    }
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
