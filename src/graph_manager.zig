const std = @import("std");
const builtin = @import("builtin");
const debug: bool = (builtin.mode == .Debug);
const ArenaUnmanaged = @import("arena_unmanaged.zig");
const zg = @import("zigrad.zig");
const TensorConfig = zg.TensorConfig;

const GraphManager = @This();

pub const NodeType = enum { leaf, internal };

/// Manages the overall graph, allows for a more memory efficient abstraction where the data structures used for
/// traversing the graph during backprop can be managed independently and reused across training steps
const NodeMeta = struct {
    pending: u32,
    visited: bool,
    rtti_id: if (debug) usize else void,

    pub fn init(T: type) NodeMeta {
        return .{
            .pending = 0,
            .visited = false,
            .rtti_id = type_id(T),
        };
    }
};

/// Sharable graph heap interface for tensors.
///
/// # ADR
///
/// - This abstraction allows us to separately manage leaf and internal tensor shells when building computation graphs.
/// - Leaves and internal nodes have different lifetimes.
pub const NodeAllocator = struct {
    const Self = @This();
    allocator: std.mem.Allocator,
    /// arena for all temporary nodes created by ops
    internal_node_arena: *ArenaUnmanaged,

    pub fn create_node(self: Self, Tensor: type, node_type: NodeType) !*Tensor {
        return switch (node_type) {
            .leaf => self.allocator.create(Tensor),
            .internal => self.internal_node_arena.create(self.allocator, Tensor),
        };
    }
    pub fn destroy_node(self: Self, node: anytype, node_type: NodeType) void {
        return switch (node_type) {
            .leaf => self.allocator.destroy(node),
            .internal => {
                // Internal nodes are kept in the arena and are not freed individually.
                // Use reset to clear internal graph nodes.
            },
        };
    }
};

allocator: std.mem.Allocator,
/// stores the output result of topological sort
sorted_nodes: std.AutoArrayHashMapUnmanaged(*anyopaque, NodeMeta) = .empty,
/// stores children for the backwards pass
backward_node_stack: std.ArrayListUnmanaged(*anyopaque) = .empty,
/// arena for all temporary nodes created by ops
internal_node_arena: ArenaUnmanaged = .empty,
/// frees unaquired tensors on reverse
eager_teardown: bool,

pub fn init(allocator: std.mem.Allocator, config: struct {
    eager_teardown: bool = false,
}) GraphManager {
    return GraphManager{
        .allocator = allocator,
        .eager_teardown = config.eager_teardown,
    };
}

/// Create a heap interface for op-graph building
pub fn heap(self: *GraphManager) NodeAllocator {
    return .{
        .allocator = self.allocator,
        .internal_node_arena = &self.internal_node_arena,
    };
}

pub fn deinit(self: *GraphManager) void {
    self.sorted_nodes.deinit(self.allocator);
    self.backward_node_stack.deinit(self.allocator);
    self.internal_node_arena.deinit(self.allocator);
    self.* = undefined;
}

pub fn create_node(self: *GraphManager, Tensor: type, node_type: NodeType) !*Tensor {
    return switch (node_type) {
        .leaf => self.allocator.create(Tensor),
        .internal => self.internal_node_arena.create(self.allocator, Tensor),
    };
}

/// Clears and retains memory - can be called in the case of a failed forward operation to destroy the computation
/// graph. Using computed nodes that belong to this graph after calling reset is undefined behavior.
pub fn reset(self: *GraphManager) void {
    self.sorted_nodes.clearRetainingCapacity();
    self.backward_node_stack.clearRetainingCapacity();
    _ = self.internal_node_arena.reset(self.allocator, .retain_capacity);
}

// TODO: This is all strongly typed - at some point, we may
// need to create a type-erased wrapper for graph operat, node_type:! NodeTypeion
// switch (node_types {
// .leaf => {
// return self.allocator.create(Tensor)}};
// to support graphs with heterogenous tensor types.
// .internal => {
// return self.internal_node_arena.create(self.allocator, Tensor)};
pub fn topological_sort(self: *GraphManager, node: anytype) void {
    const gopr = self.sorted_nodes.getOrPut(self.allocator, node) catch unreachable;

    if (gopr.found_existing) {
        gopr.value_ptr.pending += 1;
        return;
    }

    gopr.value_ptr.* = NodeMeta.init(@TypeOf(node));

    var children = node.child_iterator() orelse return;
    while (children.next()) |child| {
        if (!child.attached()) continue;
        self.topological_sort(child);
    }
}

// Must init grad on root node before backprop
pub fn backward(self: *GraphManager, root: anytype) !void {
    const Tensor = @TypeOf(root);
    const rtti_id = type_id(Tensor);

    // do not reset the arena - that contains the graph that we are currently attempting to reverse over
    self.sorted_nodes.clearRetainingCapacity();
    self.backward_node_stack.clearRetainingCapacity();

    self.topological_sort(root);
    self.backward_node_stack.append(self.allocator, root) catch unreachable;

    outer: while (self.backward_node_stack.pop()) |opaque_parent| {
        const parent: Tensor = @ptrCast(@alignCast(opaque_parent));

        defer if (!parent.acquired() and self.eager_teardown) parent.deinit();
        if (!parent.requires_grad()) continue :outer;

        try parent.backward();

        var children = parent.child_iterator() orelse continue :outer;

        inner: while (children.next()) |child| {
            const meta = self.sorted_nodes.getPtr(child) orelse continue :inner;

            if (comptime debug) {
                // enforce homogeneous tensors until we support backwards graphs.
                std.debug.assert(meta.rtti_id == rtti_id);
            }

            if (meta.pending == 0 and !meta.visited) {
                self.backward_node_stack.append(self.allocator, child) catch unreachable;
                meta.visited = true;
            } else {
                meta.pending -|= 1;
            }
        }
    }
}

pub fn teardown(self: *GraphManager, root: anytype) void {
    const Tensor = @TypeOf(root);

    self.sorted_nodes.clearRetainingCapacity();
    self.topological_sort(root);

    var iter = self.sorted_nodes.iterator();

    while (iter.next()) |entry| {
        const node: Tensor = @ptrCast(@alignCast(entry.key_ptr.*));
        if (node.acquired()) continue;
        node.deinit();
    }
}

test "GraphManager eager teardown reuse 1" {
    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();

    var gm = GraphManager.init(std.testing.allocator, .{});
    defer gm.deinit();

    const config: TensorConfig = .{
        .device = cpu.reference(),
        .node_allocator = gm.heap(),
        .requires_grad = true,
    };

    zg.rt_grad_enabled = true;

    const Tensor = zg.NDTensor(f32);

    // This pattern tries to create a graph with multiple valid paths
    // F   B   A
    //  \ / \ /
    //   D   C
    //    \ /
    //     E

    var A = try Tensor.from_slice(&.{2.0}, null, config);
    var B = try Tensor.from_slice(&.{3.0}, null, config);
    var F = try Tensor.from_slice(&.{1.5}, null, config);

    // Acquire leaf tensors
    A.acquire();
    B.acquire();
    F.acquire();

    const C = try B.mul(A);
    const D = try B.add(F);
    const E = try C.mul(D);

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
    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();

    var gm = GraphManager.init(std.testing.allocator, .{});
    defer gm.deinit();

    const config: TensorConfig = .{
        .device = cpu.reference(),
        .node_allocator = gm.heap(),
        .requires_grad = true,
    };

    zg.rt_grad_enabled = true;

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

    var A = try Tensor.from_slice(&.{2.0}, null, config);
    const B = try Tensor.from_slice(&.{3.0}, null, config);

    // Acquire leaf tensors
    A.acquire();
    B.acquire();

    const C = try A.mul(B);
    const D = try A.mul(C);
    const E = try D.add(C);

    try gm.backward(E);

    // Clean up leaves
    A.release();
    B.release();
    A.deinit();
    B.deinit();
}

test "GraphManager x*x" {
    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();

    var gm = GraphManager.init(std.testing.allocator, .{});
    defer gm.deinit();

    const config: TensorConfig = .{
        .device = cpu.reference(),
        .node_allocator = gm.heap(),
        .requires_grad = true,
    };

    const Tensor = zg.NDTensor(f32);

    zg.rt_grad_enabled = true;

    const A = try Tensor.from_slice(&.{2.0}, null, config);
    const B = try Tensor.from_slice(&.{3.0}, null, config);

    const C = try A.mul(B);
    const E = try C.mul(C);

    // Acquire leaf tensors
    A.acquire();
    B.acquire();

    try gm.backward(E);

    // Clean up leaves
    A.release();
    B.release();
    A.deinit();
    B.deinit();
}

test "GraphManager subgraphs/detach" {
    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();

    var gm = GraphManager.init(std.testing.allocator, .{});
    defer gm.deinit();

    const config: TensorConfig = .{
        .device = cpu.reference(),
        .node_allocator = gm.heap(),
        .requires_grad = true,
    };

    const Tensor = zg.NDTensor(f32);

    zg.rt_grad_enabled = true;

    // subgraph 1
    const a = try Tensor.from_slice(&.{2.0}, null, config);
    defer a.deinit();

    const b = try Tensor.from_slice(&.{3.0}, null, config);
    defer b.deinit();

    const c = try a.add(b);
    defer c.deinit();

    c.detach();

    // subgraph 2
    const d = try Tensor.from_slice(&.{4.0}, null, config);
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

    try gm.backward(e);

    // gradients should be collected by all children that require a gradient
    try std.testing.expect(e.grad != null);
    try std.testing.expect(d.grad != null);
    try std.testing.expect(c.grad != null);

    // traversal should have stopped at the detached child node (c)
    try std.testing.expect(a.grad == null);
    try std.testing.expect(b.grad == null);

    try gm.backward(c);

    // traversal should happen to the attached children of the parent node.
    try std.testing.expect(a.grad != null);
    try std.testing.expect(b.grad != null);
}

// At some point, move this to a helper - this also appears
// in the backward context object.
fn type_id(T: type) if (debug) usize else void {
    if (comptime !debug) return {};

    const Context = struct {
        const held = T;
        var id: u8 = 0;
    };
    return @intFromPtr(&Context.id);
}
