const std = @import("std");
const zg = @import("zigrad.zig");
const settings = zg.settings;
const DeviceReference = zg.DeviceReference;
const log = std.log.scoped(.zg_graphmanager);

/// Manages the overall graph, allows for a more memory efficient abstraction
/// where the data structures used for traversing the graph during backprop
/// can be managed independently and reused across training steps
pub fn GraphManager(comptime T: type) type {
    return struct {
        const Self = @This();
        allocator: std.mem.Allocator,
        sorted_nodes: std.ArrayList(*T),
        visited_nodes: std.AutoHashMap(*const T, void),
        eager_teardown: bool,

        pub const GraphOpts = struct {
            /// Setting this means you _really_ know what you are doing.
            eager_teardown: bool = false,
        };

        pub fn init(allocator: std.mem.Allocator, opts: GraphOpts) Self {
            return Self{
                .allocator = allocator,
                .sorted_nodes = std.ArrayList(*T).init(allocator),
                .visited_nodes = std.AutoHashMap(*const T, void).init(allocator),
                .eager_teardown = opts.eager_teardown,
            };
        }

        pub fn deinit(self: *Self) void {
            self.sorted_nodes.deinit();
            self.visited_nodes.deinit();
            self.* = undefined;
        }

        fn topo(self: *Self, node: *T) void {
            const gopr = self.visited_nodes.getOrPut(node) catch unreachable;
            if (!gopr.found_existing) {
                const bwd_ctx = &(node._backward_ctx orelse return);
                var children = bwd_ctx.child_iterator();
                while (children.next()) |child| {
                    if (!child.attached) continue;
                    self.topo(child);
                }
                self.sorted_nodes.append(node) catch unreachable;
            }
        }

        // Must init grad on root node before backprop
        pub fn backward(self: *Self, node: *T) !void {
            self.sorted_nodes.clearRetainingCapacity();
            self.visited_nodes.clearRetainingCapacity();
            self.topo(node);
            const nodes = self.sorted_nodes.items;

            for (0..nodes.len) |i| {
                var curr_node = nodes[nodes.len - i - 1];
                if (curr_node.requires_grad()) {
                    try curr_node.backward();
                    // if eager_teardown, immediately destroy node. note that deinit is designed to not cascade recursively,
                    // it just destroys the current tensor and not the children
                    if (!curr_node.acquired and self.eager_teardown) {
                        //log.warn("destroying node {?s}", .{curr_node.get_label()});
                        curr_node.deinit();
                        //log.warn("  ->destroyed", .{});
                    }
                } else {
                    log.debug("Skipping node {?s}", .{node.get_label()});
                }
            }
        }
    };
}

test "GraphManager eager teardown reuse 1" {
    const T = f32;
    const Tensor = zg.NDTensor(T);
    const allocator = std.testing.allocator;

    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();

    const device = cpu.reference();
    zg.rt_grad_enabled = true;

    // This pattern tries to create a graph with multiple valid paths
    // F   B   A
    //  \ / \ /
    //   D   C
    //    \ /
    //     E

    var A = try Tensor.init(&[_]T{2.0}, null, true, device);
    var B = try Tensor.init(&[_]T{3.0}, null, true, device);
    var F = try Tensor.init(&[_]T{1.5}, null, true, device);

    // Acquire leaf tensors
    A.acquire();
    B.acquire();
    F.acquire();

    const C = try B.mul(A);
    const D = try B.add(F);
    const E = try C.mul(D);

    // Setup graph manager with eager teardown
    var gm = GraphManager(Tensor).init(device.allocator, .{ .eager_teardown = true });
    defer gm.deinit();

    // Run backward pass
    try gm.backward(E);

    // The tricky part: B's value is needed for both C and D's backward pass
    // If B is freed too early, we'll get a use-after-free or crash

    // Clean up
    A.release();
    B.release();
    F.release();

    A.deinit();
    B.deinit();
    F.deinit();
}

test "GraphManager eager teardown reuse 2" {
    const T = f32;
    const Tensor = zg.NDTensor(T);
    const allocator = std.testing.allocator;
    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();
    zg.rt_grad_enabled = true;

    // Create a case where a grad needs to accumulate multiple times
    //   C = A * B
    //   D = A * C  (reuses A)
    //   E = D + C  (reuses C)
    //
    //       A  B
    //     / \ /
    //    |   C
    //    \  / \
    //     D   |
    //      \ /
    //       E

    var A = try Tensor.init(&[_]T{2.0}, null, true, device);
    const B = try Tensor.init(&[_]T{3.0}, null, true, device);

    // Acquire leaf tensors
    A.acquire();
    B.acquire();

    const C = try A.mul(B);
    const D = try A.mul(C);
    const E = try D.add(C);

    var gm = GraphManager(Tensor).init(device.allocator, .{ .eager_teardown = true });
    defer gm.deinit();

    try gm.backward(E);

    // Clean up leaves
    A.release();
    B.release();
    A.deinit();
    B.deinit();
}

test "GraphManager x*x" {
    const T = f32;
    const Tensor = zg.NDTensor(T);
    const allocator = std.testing.allocator;
    zg.rt_grad_enabled = true;

    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();

    const A = try Tensor.init(&[_]T{2.0}, null, true, device);
    const B = try Tensor.init(&[_]T{3.0}, null, true, device);

    const C = try A.mul(B);
    const E = try C.mul(C);

    // Acquire leaf tensors
    A.acquire();
    B.acquire();

    var gm = GraphManager(Tensor).init(allocator, .{ .eager_teardown = true });
    defer gm.deinit();

    try gm.backward(E);

    // Clean up leaves
    A.release();
    B.release();
    A.deinit();
    B.deinit();
}

test "GraphManager subgraphs/detach" {
    const T = f32;
    const Tensor = zg.NDTensor(T);
    const allocator = std.testing.allocator;
    zg.rt_grad_enabled = true;

    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();

    const device = cpu.reference();

    // subgraph 1
    const a = try Tensor.init(&.{2.0}, null, true, device);
    defer a.deinit();

    const b = try Tensor.init(&.{3.0}, null, true, device);
    defer b.deinit();

    const c = try a.add(b);
    defer c.deinit();

    c.detach();

    // subgraph 2
    const d = try Tensor.init(&.{4.0}, null, true, device);
    defer d.deinit();

    const e = try c.add(d);
    defer e.deinit();

    /////////////////////////////
    //
    //      a   b
    //       \ /
    //        c
    //        .
    //        .  <----- detached
    //        .
    //        c   d
    //         \ /
    //          e

    var gm = GraphManager(Tensor).init(allocator, .{ .eager_teardown = false });
    defer gm.deinit();

    try gm.backward(e);
    // gradients should be collected by all
    // children that require a gradient
    try std.testing.expect(e.grad != null);
    try std.testing.expect(d.grad != null);
    try std.testing.expect(c.grad != null);
    // traversal should have stopped at the
    // detached child node (c)
    try std.testing.expect(a.grad == null);
    try std.testing.expect(b.grad == null);

    try gm.backward(c);
    // traversal should happen to the atached
    // children of the parent node.
    try std.testing.expect(a.grad != null);
    try std.testing.expect(b.grad != null);
}
