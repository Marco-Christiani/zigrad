const std = @import("std");
const builtin = @import("builtin");
const debug: bool = (builtin.mode == .Debug);

const zg = @import("zigrad.zig");
const ArenaUnmanaged = @import("allocators/arena_unmanaged.zig");
pub const Node = @import("graph/node.zig");
pub const Builder = @import("graph/builder.zig");

const Graph = @This();

const PathInfo = struct {
    pending: u32 = 0,
    visited: bool = false,
};

pub const Opts = struct {
    eager_teardown: bool = false,
};

/// Manages the overall graph, allows for a more memory efficient abstraction where the data structures used for
/// traversing the graph during backprop can be managed independently and reused across training steps
builder: Builder,
/// stores the output result of topological sort
sorted_nodes: std.AutoArrayHashMapUnmanaged(*Node, PathInfo) = .empty,
/// stores children for the backwards pass
backward_node_stack: std.ArrayListUnmanaged(*Node) = .empty,
/// frees unacquired tensors on reverse
eager_teardown: bool,

pub fn init(allocator: std.mem.Allocator, opts: Opts) Graph {
    return .{
        .builder = .{ .allocator = allocator },
        .eager_teardown = opts.eager_teardown,
    };
}

pub fn deinit(self: *Graph) void {
    self.sorted_nodes.deinit(self.builder.allocator);
    self.backward_node_stack.deinit(self.builder.allocator);
    self.builder.deinit();
    self.* = undefined;
}

pub fn topological_sort(self: *Graph, node: *Node) void {
    const gopr = self.sorted_nodes.getOrPut(self.builder.allocator, node) catch unreachable;

    if (gopr.found_existing) {
        gopr.value_ptr.pending += 1;
        return;
    }

    gopr.value_ptr.* = PathInfo{};

    var children = node.child_iterator() orelse return;
    while (children.next()) |child| {
        if (!child.attached()) continue;
        self.topological_sort(child);
    }
}

// Must init grad on root node before backprop
pub fn backward(self: *Graph, root: *Node) !void {
    // do not reset the arena - that contains the graph that we are currently attempting to reverse over
    self.sorted_nodes.clearRetainingCapacity();
    self.backward_node_stack.clearRetainingCapacity();

    self.topological_sort(root);
    self.backward_node_stack.append(self.builder.allocator, root) catch unreachable;

    outer: while (self.backward_node_stack.pop()) |parent| {
        defer if (self.eager_teardown and !parent.acquired()) {
            parent.deinit();
        };

        if (!parent.requires_grad()) continue :outer;

        try parent.backward();

        var children = parent.child_iterator() orelse continue :outer;

        inner: while (children.next()) |child| {
            const info = self.sorted_nodes.getPtr(child) orelse continue :inner;
            if (info.pending == 0 and !info.visited) {
                self.backward_node_stack.append(self.builder.allocator, child) catch unreachable;
                info.visited = true;
            } else {
                info.pending -|= 1;
            }
        }
    }
}

/// Calls clear on all nodes in that are attached to the root. This frees object resources
/// that are associated with the nodes. Freeing nodes themselves must be done with reset.
pub fn teardown(self: *Graph, root: *Node) void {
    self.sorted_nodes.clearRetainingCapacity();
    self.topological_sort(root);

    for (self.sorted_nodes.keys()) |node| {
        if (!node.acquired()) node.deinit();
    }
}

/////////////////////////////////////////////////////
// Testing //////////////////////////////////////////

const TensorOpts = @import("ndtensor.zig").TensorOpts;

// limit memory usage for testing
const TestOpts: zg.device.HostDevice.Options = .{
    .max_pool_size = zg.constants.@"1Mb" / 2, // probably still excessive
};

comptime {
    std.testing.refAllDecls(@This());
}

test "Graph eager teardown reuse 1" {
    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{
        .eager_teardown = true,
    });
    defer graph.deinit();

    const opts: TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    zg.runtime.grad_enabled = true;

    const Tensor = zg.NDTensor(f32);

    // This pattern tries to create a graph with multiple valid paths
    // F   B   A
    //  \ / \ /
    //   D   C
    //    \ /
    //     E

    var A = try Tensor.from_slice(device, &.{2.0}, null, opts);
    var B = try Tensor.from_slice(device, &.{3.0}, null, opts);
    var F = try Tensor.from_slice(device, &.{1.5}, null, opts);

    // Acquire leaf tensors
    A.acquire();
    B.acquire();
    F.acquire();

    const C = try B.mul(A);
    const D = try B.add(F);
    const E = try C.mul(D);

    // Run backward pass
    try E.backward();

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

test "Graph eager teardown reuse 2" {
    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{
        .eager_teardown = true,
    });
    defer graph.deinit();

    const opts: TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    zg.runtime.grad_enabled = true;

    const Tensor = zg.NDTensor(f32);

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

    const A = try Tensor.from_slice(device, &.{2.0}, null, opts);
    const B = try Tensor.from_slice(device, &.{3.0}, null, opts);

    // Acquire leaf tensors
    A.acquire();
    B.acquire();

    const C = try A.mul(B);
    const D = try A.mul(C);
    const E = try D.add(C);

    try E.backward();

    // Clean up leaves
    A.release();
    B.release();
    A.deinit();
    B.deinit();
}

test "Graph x*x" {
    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{
        .eager_teardown = true,
    });
    defer graph.deinit();

    const opts: TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const Tensor = zg.NDTensor(f32);

    zg.runtime.grad_enabled = true;

    const A = try Tensor.from_slice(device, &.{2.0}, null, opts);
    const B = try Tensor.from_slice(device, &.{3.0}, null, opts);

    const C = try A.mul(B);
    const E = try C.mul(C);

    // Acquire leaf tensors
    A.acquire();
    B.acquire();

    try E.backward();

    // Clean up leaves
    A.release();
    B.release();
    A.deinit();
    B.deinit();
}

test "Graph subgraphs/detach" {
    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    const opts: TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const Tensor = zg.NDTensor(f32);

    zg.runtime.grad_enabled = true;

    // subgraph 1
    const a = try Tensor.from_slice(device, &.{2.0}, null, opts);
    defer a.deinit();

    const b = try Tensor.from_slice(device, &.{3.0}, null, opts);
    defer b.deinit();

    const c = try a.add(b);
    defer c.deinit();

    c.detach();

    // subgraph 2
    const d = try Tensor.from_slice(device, &.{4.0}, null, opts);
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

    try e.backward();

    // gradients should be collected by all children that require a gradient
    try std.testing.expect(e.grad != null);
    try std.testing.expect(d.grad != null);
    try std.testing.expect(c.grad != null);

    // traversal should have stopped at the detached child node (c)
    try std.testing.expect(a.grad == null);
    try std.testing.expect(b.grad == null);

    try c.backward();

    // traversal should happen to the attached children of the parent node.
    try std.testing.expect(a.grad != null);
    try std.testing.expect(b.grad != null);
}
