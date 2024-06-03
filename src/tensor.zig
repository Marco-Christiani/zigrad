// TODO: implement gradient forward and backward
// TODO: since the comptime ideas were trashed, shape/size should not be comptime
// TODO: implement view(), transpose(), and permute(), where the latter two mutate the shape
const std = @import("std");
const zarray = @import("zarray.zig");
const generateASCIIUUID = @import("grad.zig").generateASCIIUUID;
const c = @cImport(@cInclude("Accelerate/Accelerate.h"));
const NDArray = zarray.NDArray;
const ZarrayError = zarray.ZarrayError;
const root = @import("root");

/// lib-wide options that can be overridden by the root file.
pub const settings: Settings = if (@hasDecl(root, "zigrad_settings")) root.zigrad_settings else .{};

const Settings = struct {
    grad_enabled: bool = true,
    max_dim: usize = 4,
};

pub const Op = enum { ADD, SUB, MUL, DIV, POW, TANH, MATMUL, DOT, MATVEC };

// pub fn ZarrayT(T: type) type {
//     return struct {
//         size: usize,
//         shape: []const usize,
//         data: []anyopaque,
//         init: fn (values: []const T, shape: ?[]const usize, allocator: std.mem.Allocator) *ZarrayT(T),
//         initFill: fn (val: T, len: usize, shape: ?[]const usize, allocator: std.mem.Allocator) *ZarrayT(T),
//         // fill = fn (self: *Self, val: T) void,
//         // reshape = fn (self: *Self, shape: []usize) void,
//         // get = fn (self: Self, indices: [self.shape.len]u32) T,
//         // set = fn (self: *Self, indices: [self.shape.len]u32, value: T) void,
//         // posToIndex = fn (self: Self, indices: [self.shape.len]u32) u32,
//         // indexToPos = fn (self: Self, index: u32) [self.shape.len]u32,
//         // print = fn (self: Self) void,
//         // setOp = fn (self: *Self, op: Op) *Self,
//         // add = fn (self: *const Self, other: *const Self) void,
//         // sub = fn (self: *const Self, other: *const Self) void,
//         // mul = fn (self: *const Self, other: *const Self) void,
//         // div = fn (self: *const Self, other: *const Self) void,
//     };
// }

pub fn NDTensor(comptime T: type) type {
    return struct {
        const dtype = NDArray(T);
        const Self = @This();
        data: *dtype,
        op: ?Op = null,
        children: ?[2]*const Self = null,
        label: ?[]const u8 = null,
        grad: ?*dtype = null,
        requires_grad: bool,

        pub fn init(values: []const T, shape: ?[]const usize, requires_grad: bool, allocator: std.mem.Allocator) !*Self {
            const result = try allocator.create(Self);
            const zeros = try allocator.alloc(T, values.len);
            @memset(zeros, 0);
            result.* = Self{
                .data = try dtype.init(values, shape, allocator),
                .grad = if (requires_grad) try dtype.init(zeros, shape, allocator) else null,
                .requires_grad = requires_grad,
            };
            return result;
        }

        pub fn fromZarray(values: *dtype, requires_grad: bool, allocator: std.mem.Allocator) !*Self {
            const result = try allocator.create(Self);
            const zeros = try allocator.alloc(T, values.data.len);
            @memset(zeros, 0);
            result.* = Self{
                .data = values,
                .grad = if (requires_grad) try dtype.init(zeros, values.shape, allocator) else null,
                .requires_grad = requires_grad,
            };
            return result;
        }

        pub fn reshape(self: *Self, shape: []const usize) !*Self {
            try self.data.reshape(shape);
            if (self.grad) |g| {
                try g.reshape(shape);
            }
            return self;
        }

        pub fn setLabel(self: *Self, comptime label: ?[]const u8) *Self {
            const uid = generateASCIIUUID(4);
            if (label) |l| {
                var buf: [l.len + 4]u8 = undefined;
                _ = std.fmt.bufPrint(&buf, "{?s}{s}", .{ l, uid }) catch {
                    @panic("Failed to set label");
                };
                self.label = &buf;
            } else {
                self.label = uid;
            }
            return self;
        }

        pub fn fill(self: *Self, val: T) void {
            self.data.fill(val);
        }

        pub fn get(self: Self, indices: []const usize) T {
            return self.data.get(indices);
        }

        pub fn set(self: *Self, indices: []const usize, value: T) !void {
            try self.data.set(indices, value);
        }

        fn posToIndex(self: Self, indices: []const usize) usize {
            return self.data.posToIndex(indices);
        }

        fn flexPosToIndex(self: Self, indices: []const usize) ZarrayError.InvalidIndex!usize {
            return self.data.flexPosToIndex(indices);
        }

        fn indexToPos(self: Self, index: usize, allocator: std.mem.Allocator) []const usize {
            return self.data.indexToPos(index, allocator);
        }

        pub fn print(self: *const Self) void {
            std.debug.print("NDTensor<{},{?s}>[", .{ T, if (self.op) |o| @tagName(o) else null });
            std.debug.print("data: ", .{});
            self.data.print();
            if (self.grad) |g| {
                std.debug.print(" grad: ", .{});
                g.print();
            }
            std.debug.print("], requires_grad={}\n", .{self.requires_grad});
        }

        /// Element-wise addition
        pub fn add(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            var result = try Self.fromZarray(try self.data.add(other.data, allocator), self.requires_grad, allocator);
            if (self.requires_grad) result.op = .ADD;
            result.children = .{ self, other };
            return result;
        }

        /// Element-wise subtraction
        pub fn sub(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            var result = try Self.fromZarray(try self.data.sub(other.data, allocator), self.requires_grad, allocator);
            if (self.requires_grad) result.op = .SUB;
            result.children = .{ self, other };
            return result.reshape(self.data.shape) catch unreachable;
        }

        /// Element-wise multiplication
        pub fn mul(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            var result = try Self.fromZarray(try self.data.mul(other.data, allocator), self.requires_grad, allocator);
            if (self.requires_grad) result.op = .MUL;
            result.children = .{ self, other };
            return result;
        }

        /// Element-wise division
        pub fn div(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            var result = try Self.fromZarray(try self.data.div(other.data, allocator), self.requires_grad, allocator);
            if (self.requires_grad) result.op = .DIV;
            result.children = .{ self, other };
            return result.reshape(self.data.shape) catch unreachable;
        }

        pub fn matmul(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            var result = try Self.fromZarray(try self.data.matmul(other.data, allocator, false, false), self.requires_grad, allocator);
            if (self.requires_grad) result.op = .MATMUL;
            result.children = .{ self, other };
            return result;
        }

        pub fn dot(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            var result = try Self.fromZarray(try self.data.dot(other.data, allocator), self.requires_grad, allocator);
            if (self.requires_grad) result.op = .DOT;
            result.children = .{ self, other };
            return result;
        }

        pub fn matvec(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            var result = try Self.fromZarray(try self.data.matvec(other.data, allocator), self.requires_grad, allocator);
            if (self.requires_grad) result.op = .MATVEC;
            result.children = .{ self, other };
            return result;
        }

        pub fn backward(self: *const Self, allocator: std.mem.Allocator) !void {
            if (!self.requires_grad) return;
            if (self.op) |op| {
                switch (op) {
                    .ADD => {
                        if (self.children) |children| {
                            _ = try children[0].grad.?._add(self.grad.?);
                            _ = try children[1].grad.?._add(self.grad.?);
                        }
                    },
                    .SUB => {
                        if (self.children) |children| {
                            _ = try children[0].grad.?._add(self.grad.?);
                            _ = try children[1].grad.?._sub(self.grad.?);
                        }
                    },
                    .MUL => {
                        if (self.children) |children| {
                            _ = try (try children[0].grad.?._add(self.grad.?))._mul(children[1].data);
                            _ = try (try children[1].grad.?._add(self.grad.?))._mul(children[0].data);
                        }
                    },
                    .DIV => {
                        if (self.children) |children| {
                            var a = children[0].data;
                            const b = children[1].data;

                            // Gradient with respect to the numerator
                            _ = try (try children[0].grad.?._add(self.grad.?))._div(b);

                            // Gradient with respect to the denominator
                            const temp = try a._mul(self.grad.?);
                            _ = try (try (try children[1].grad.?._sub(temp))._div(b))._div(b);
                        }
                    },
                    // .DIV => { // TODO
                    //     if (self.children) |children| {
                    //         var a = children[0].data;
                    //         var b = children[1].data;
                    //         // g/b
                    //         var temp = self.grad.?.div(b); // copies
                    //         _ = children[0].grad.?._add(&temp);
                    //         // a / b^2
                    //         const temp2 = @constCast(&(a.div(&b.mul(b))))._mul(self.grad.?); // copies
                    //         // += -(a/b^2) ==> -= a/b^2
                    //         _ = children[1].grad.?._sub(temp2);
                    //     }
                    // },
                    .MATMUL => {
                        if (self.children) |children| {
                            var A = children[0].data;
                            const B = children[1].data;

                            // gradient wrt A: Must be free'd, assumes owner uses an arena for bw and does not return handle
                            const grad_A = try self.grad.?.matmul(B, allocator, false, true);
                            _ = try children[0].grad.?._add(grad_A);

                            // gradient wrt B: Must be free'd, assumes owner uses an arena for bw and does not return handle
                            const grad_B = try A.matmul(self.grad.?, allocator, true, false);
                            _ = try children[1].grad.?._add(grad_B);
                        }
                    },
                    .DOT => {
                        if (self.children) |children| {
                            var a = children[0].data;
                            var b = children[1].data;
                            // Must be free'd, assumes owner uses an arena for bw and does not return handle
                            const grad_a = try b.mul(self.grad.?, allocator);
                            // Must be free'd, assumes owner uses an arena for bw and does not return handle
                            const grad_b = try a.mul(self.grad.?, allocator);
                            _ = try children[0].grad.?._add(grad_a);
                            _ = try children[1].grad.?._add(grad_b);
                        }
                    },
                    .MATVEC => {
                        if (self.children) |children| {
                            var A = children[0].data;
                            const x = children[1].data;
                            // std.debug.print("A.shape={d} x.shape={d}\n", .{ A.shape, x.shape });
                            // const x_shape = x.shape;
                            // const x_shape_new = try allocator.alloc(usize, x_shape.len + 1);
                            // std.mem.copyForwards(usize, x_shape_new, x_shape);
                            // x_shape_new[x_shape_new.len - 1] = 1;
                            // try x.reshape(x_shape_new);
                            // std.debug.print("A.shape={d} x.shape={d}\n", .{ A.shape, x.shape });
                            // std.debug.print("self.data.shape={d} self.grad.shape={d}\n", .{ self.data.shape, self.grad.?.shape });

                            // Gradient with respect to A
                            // Must be free'd, assumes owner uses an arena for bw and does not return handle
                            const grad_A = try self.grad.?.outer(x, allocator, false, true);
                            _ = try children[0].grad.?._add(grad_A);
                            // try x.reshape(x_shape); // reset shape
                            // std.debug.assert(x_shape.ptr != x_shape_new.ptr);

                            // Gradient with respect to x
                            // Must be free'd, assumes owner uses an arena for bw and does not return handle
                            const grad_x = try A.matmul(self.grad.?, allocator, true, false);
                            _ = try children[1].grad.?._add(grad_x);
                        }
                    },
                    else => @panic("Not yet implemented."),
                }
            }
        }
    };
}

/// Manages the overall graph, allows for a more memory efficient abstraction
/// where the data structures used for traversing the graph during backprop
/// can be managed independently and reused across training steps
pub fn Loss(comptime T: type) type {
    return struct {
        const Self = @This();
        sortedNodes: std.ArrayList(*const T),
        visitedNodes: std.AutoHashMap(*const T, void),

        pub fn init(alloc: std.mem.Allocator) *Self {
            const self = alloc.create(Self) catch unreachable;
            self.* = Self{
                .sortedNodes = std.ArrayList(*const T).init(alloc),
                .visitedNodes = std.AutoHashMap(*const T, void).init(alloc),
            };
            return self;
        }

        pub fn deinit(self: *Self) void {
            self.sortedNodes.deinit();
            self.visitedNodes.deinit();
        }

        fn topo(self: *Self, node: *const T) void {
            const gopr = self.visitedNodes.getOrPut(node) catch unreachable;
            if (!gopr.found_existing) {
                if (node.children) |children| {
                    for (children) |child| {
                        self.topo(child);
                    }
                }
                self.sortedNodes.append(node) catch unreachable;
                node.print();
            }
        }

        // Must init grad on root node before backprop
        pub fn backward(self: *Self, node: *const T, alloc: std.mem.Allocator) !void {
            self.sortedNodes.clearRetainingCapacity();
            self.visitedNodes.clearRetainingCapacity();
            self.topo(node);
            const nodes = self.sortedNodes.items;
            for (0..nodes.len) |i| {
                try nodes[nodes.len - i - 1].backward(alloc);
            }
        }
    };
}

test "tensor/NDTensor index, add, div" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const shape = &[_]usize{ 2, 3 };
    const Tensor = NDTensor(f32);

    // 1 2 3
    // 4 5 6
    var t1 = try Tensor.init(@constCast(&[_]f32{ 1, 2, 3, 4, 5, 6 }), shape, false, alloc);

    // 1 2 3
    // 4 5 23
    try t1.set(&[_]usize{ 1, 2 }, -5);
    t1.print();
    // std.debug.print("{d}\n", .{t1.get(&[_]usize{ 1, 2 })});
    // std.debug.print("{d}\n", .{t1.indexToPos(5, alloc)});

    const t2 = try Tensor.init(@constCast(&[_]f32{ 10, 20, 30, 40, 50, 60 }), shape, false, alloc);
    t2.print();
    const t3 = try t1.add(t2, alloc);
    t3.print();

    std.debug.print("{d}\n", .{t3.data.data});
    t3.print();
    std.debug.print("{?any}\n", .{t3.children});

    var t4 = try t3.div(t3, alloc);
    t4.print();
}

test "tensor/GraphManager/addback" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const shape = &[_]usize{1};
    const T = f32;
    const Tensor = NDTensor(T);

    var t1 = try Tensor.init(@constCast(&[_]T{2}), shape, true, alloc);
    const t2 = try Tensor.init(@constCast(&[_]T{3}), shape, true, alloc);
    // t3 = t1 + t2;
    // dt3/dt1 = 1, dt3/dt2 = 1
    var t3 = try t1.add(t2, alloc);

    var gm = Loss(Tensor).init(arena.allocator());
    defer gm.deinit();
    t3.grad = try Tensor.dtype.init(@constCast(&[_]T{1}), shape, alloc);
    try gm.backward(t3, alloc);
    try std.testing.expectEqualDeep(&[_]T{1.0}, t1.grad.?.data);
    try std.testing.expectEqualDeep(&[_]T{1.0}, t2.grad.?.data);
}

test "tensor/GraphManager/mulback" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const shape = &[_]usize{1};
    const T = f32;
    const Tensor = NDTensor(T);

    var t1 = try Tensor.init(@constCast(&[_]T{2}), shape, true, alloc);
    const t2 = try Tensor.init(@constCast(&[_]T{3}), shape, true, alloc);
    // t3 = t1 * t2;
    // dt3/dt1 = t2, dt3/dt2 = t1
    var t3 = try t1.mul(t2, alloc);

    var gm = Loss(Tensor).init(arena.allocator());
    defer gm.deinit();
    t3.grad = try Tensor.dtype.init(@constCast(&[_]T{1}), shape, alloc);
    try gm.backward(t3, alloc);
    try std.testing.expectEqualDeep(t2.data.data, t1.grad.?.data);
    try std.testing.expectEqualDeep(t1.data.data, t2.grad.?.data);
}

test "tensor/GraphManager/moreback" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const shape = &[_]usize{2};
    const T = f32;
    const Tensor = NDTensor(T);

    var w = try Tensor.init(@constCast(&[_]f32{ 3, 2 }), shape, true, alloc);
    const b = try Tensor.init(@constCast(&[_]f32{ 1, 1 }), shape, true, alloc);
    const x = try Tensor.init(@constCast(&[_]f32{ 4, 4 }), shape, true, alloc);
    // const y = try Tensor.init(@constCast(&[_]f32{12, 10}), true, alloc).reshape(shape);
    // h = w*x + b
    // dh/dw = x, dh/db = 1
    var temp = try w.mul(x, alloc);
    var h = try temp.add(b, alloc);

    var backprop_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const backprop_alloc = backprop_arena.allocator();
    defer backprop_arena.deinit();

    var gm = Loss(Tensor).init(arena.allocator());
    defer gm.deinit();
    h.grad = try Tensor.dtype.init(@constCast(&[_]T{ 1, 1 }), shape, alloc);
    try gm.backward(h, backprop_alloc);
    if (!backprop_arena.reset(.retain_capacity)) @panic("reset failed.\n");
    try std.testing.expectEqualSlices(T, x.data.data, w.grad.?.data);
    try std.testing.expect(std.mem.allEqual(T, b.grad.?.data, 1));

    // 2 x 1
    const shape2 = &[_]usize{ 2, 1 };
    w.grad.?.fill(0);
    b.grad.?.fill(0);
    x.grad.?.fill(0);
    _ = try w.reshape(shape2);
    _ = try b.reshape(shape2);
    _ = try x.reshape(shape2);
    // h = w*x + b
    // dh/dw = x, dh/db = 1
    temp = try w.mul(x, alloc);
    h = try temp.add(b, alloc);

    var gm2 = Loss(Tensor).init(arena.allocator());
    defer gm2.deinit();
    h.grad.?.fill(1);
    try gm.backward(h, alloc);
    if (!backprop_arena.reset(.retain_capacity)) @panic("reset failed.\n");
    try std.testing.expectEqualSlices(T, x.data.data, w.grad.?.data);
    try std.testing.expect(std.mem.allEqual(T, b.grad.?.data, 1));
    h.print();
}

test "NDTensor/div_backward" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f32;
    const Tensor = NDTensor(T);
    const shape = &[_]usize{2};

    var t1 = try Tensor.init(@constCast(&[_]T{ 4, 9 }), shape, true, alloc);
    const t2 = try Tensor.init(@constCast(&[_]T{ 2, 3 }), shape, true, alloc);
    var t3 = try t1.div(t2, alloc);

    var gm = Loss(Tensor).init(arena.allocator());
    defer gm.deinit();
    t3.grad = try Tensor.dtype.init(@constCast(&[_]T{ 1, 1 }), shape, alloc);
    try gm.backward(t3, alloc);

    // Expected gradients for t1 and t2
    const expected_grad_t1 = &[_]T{ 1.0 / 2.0, 1.0 / 3.0 }; // 1 / b
    const expected_grad_t2 = &[_]T{ -4.0 / 4.0, -9.0 / 9.0 }; // -a / b^2

    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);
}

test "NDTensor/matmul_backward" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f32;
    const Tensor = NDTensor(T);
    const shape = &[_]usize{ 2, 2 };

    var t1 = try Tensor.init(@constCast(&[_]T{ 1, 2, 3, 4 }), shape, true, alloc);
    const t2 = try Tensor.init(@constCast(&[_]T{ 1, 0, 0, 1 }), shape, true, alloc);
    var t3 = try t1.matmul(t2, alloc);

    var gm = Loss(Tensor).init(arena.allocator());
    defer gm.deinit();
    t3.grad = try Tensor.dtype.init(@constCast(&[_]T{ 1, 1, 1, 1 }), shape, alloc);
    try gm.backward(t3, alloc);

    const expected_grad_t1 = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2 = &[_]T{ 4, 4, 6, 6 };

    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);
}

test "NDTensor/matvec_backward" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f32;
    const Tensor = NDTensor(T);
    const shape_mat = &[_]usize{ 2, 2 };
    const shape_vec = &[_]usize{2};

    var t1 = try Tensor.init(@constCast(&[_]T{ 1, 2, 3, 4 }), shape_mat, true, alloc);
    const t2 = try Tensor.init(@constCast(&[_]T{ 1, 1 }), shape_vec, true, alloc);
    var t3 = try t1.matvec(t2, alloc);

    var gm = Loss(Tensor).init(arena.allocator());
    defer gm.deinit();
    t3.grad = try Tensor.dtype.init(@constCast(&[_]T{ 1, 1 }), shape_vec, alloc);
    try gm.backward(t3, alloc);

    const expected_grad_t1 = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2 = &[_]T{ 4, 6 };

    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);
}

test "NDTensor/dot_backward" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f32;
    const Tensor = NDTensor(T);
    const shape = &[_]usize{3};

    var t1 = try Tensor.init(@constCast(&[_]T{ 1, 2, 3 }), shape, true, alloc);
    const t2 = try Tensor.init(@constCast(&[_]T{ 4, 5, 6 }), shape, true, alloc);
    var t3 = try t1.dot(t2, alloc);

    var gm = Loss(Tensor).init(arena.allocator());
    defer gm.deinit();
    t3.grad = try Tensor.dtype.init(@constCast(&[_]T{1}), &[_]usize{1}, alloc);
    try gm.backward(t3, alloc);

    const expected_grad_t1 = &[_]T{ 4, 5, 6 };
    const expected_grad_t2 = &[_]T{ 1, 2, 3 };

    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);
}
// pub fn main() !void {
//     var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
//     const alloc = arena.allocator();
//     defer arena.deinit();
//     const shape1 = &[_]usize{ 2, 3 };
//     const shape2 = &[_]usize{ 3, 2 };
//     const Tensor = NDTensor(.cpu, f32, 6);
//
//     // var A = try Tensor.init(@constCast(&[_]f32{ 3, 1, 1, 2 }), true, alloc).reshape(shape);
//     // const B = try Tensor.init(@constCast(&[_]f32{ 1, 1, 1, 1 }), true, alloc).reshape(shape);
//     // var C = A.mul(B, alloc);
//     //
//     // var gm = Loss(Tensor).init(arena.allocator());
//     // defer gm.deinit();
//     // C.print();
//     // multiply a 2x3 and a 3x2 to get a 2x2
//     var A = try Tensor.init(@constCast(&[_]f32{ 1, 2, 3, 4, 5, 6 }), true, alloc).reshape(shape1);
//     const B = try Tensor.init(@constCast(&[_]f32{ 1, 0, 0, 1, 0, 0 }), true, alloc).reshape(shape2);
//     var C = A.mul(B, alloc);
//     C.print();
//
//     // var gm = Loss(Tensor).init(arena.allocator());
//     // defer gm.deinit();
//     // h.grad = try Tensor.dtype.initFill(1, shape, alloc);
//     // gm.backward(h);
//     // try std.testing.expectEqual(x.data.data, w.grad.?.data);
//     // const ones = try Tensor.dtype.initFill(1, shape, alloc);
//     // try std.testing.expectEqual(ones.data, b.grad.?.data);
//
//     // 2 x 1
//     // const shape2 = &[_]usize{ 2, 1 };
//     // w.grad.?.fill(0);
//     // b.grad.?.fill(0);
//     // x.grad.?.fill(0);
//     // _ = try w.reshape(shape2);
//     // _ = try b.reshape(shape2);
//     // _ = try x.reshape(shape2);
//     // h = w*x + b
//     // dh/dw = x, dh/db = 1
//     // temp = w.mul(x, alloc);
//     // h = temp.add(b, alloc);
//     //
//     // var gm2 = Loss(Tensor).init(arena.allocator());
//     // defer gm2.deinit();
//     // h.grad.?.fill(1);
//     // gm.backward(h);
//     // try std.testing.expectEqual(x.data.data, w.grad.?.data);
//     // try std.testing.expectEqual(ones.data, b.grad.?.data);
//     // h.print();
// }
