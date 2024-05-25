// TODO: implement gradient forward and backward
// TODO: since the comptime ideas were trashed, shape/size should not be comptime
// TODO: implement view(), transpose(), and permute(), where the latter two mutate the shape
const std = @import("std");
const Op = @import("grad.zig").Op;
const generateASCIIUUID = @import("grad.zig").generateASCIIUUID;

const root = @import("root");

/// lib-wide options that can be overridden by the root file.
pub const settings: Settings = if (@hasDecl(root, "zigrad_settings")) root.zigrad_settings else .{};

const Settings = struct {
    grad_enabled: bool = true,
    max_dim: usize = 4,
};

const DeviceTag = enum {
    cpu,
};

// const Zarray = union(ZarrayTag) { cpu: NDArray(anyopaque) };

pub fn Zarray(variant: DeviceTag, T: type, size: usize) type {
    return switch (variant) {
        .cpu => NDArray(T, size),
    };
}

pub fn ZTensor(variant: DeviceTag, T: type, size: usize) type {
    return switch (variant) {
        .cpu => NDTensor(T, size),
    };
}

const NDArrayError = IndexError;
const IndexError = error{InvalidShape};

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

pub fn NDArray(comptime T: type, comptime size: usize) type {
    return struct {
        const Self = @This();
        shape: []const usize,
        data: @Vector(size, T),

        pub fn init(values: [size]T, shape: ?[]const usize, allocator: std.mem.Allocator) NDArrayError!*Self {
            if (shape) |s| {
                if (s.len > settings.max_dim) return NDArrayError.InvalidShape;
            }
            const result = allocator.create(Self) catch {
                std.debug.panic("NDArray allocation failed.\n", .{});
            };
            result.* = Self{
                .data = values,
                .shape = if (shape) |s| s else &[_]usize{size},
            };
            return result;
        }

        pub fn initFill(val: T, shape: ?[]const usize, allocator: std.mem.Allocator) NDArrayError!*Self {
            if (shape) |s| {
                if (s.len > settings.max_dim) return NDArrayError.InvalidShape;
            }
            var data: [size]T = undefined;
            for (&data) |*elem| {
                elem.* = val;
            }
            return Self.init(data, shape, allocator);
        }

        pub fn fill(self: *Self, val: T) void {
            for (0..size) |i| {
                self.data[i] = val;
            }
        }

        pub fn reshape(self: *Self, shape: []const usize) NDArrayError!void {
            // TODO: check shape bounds (e.g. the total size)
            if (shape.len > settings.max_dim) return NDArrayError.InvalidShape;
            self.shape = shape;
        }

        pub fn get(self: Self, indices: []const usize) T {
            const index = self.posToIndex(indices);
            return self.data[index];
        }

        pub fn set(self: *Self, indices: []const usize, value: T) NDArrayError!void {
            if (indices.len != self.shape.len) {
                return NDArrayError.InvalidShape;
            }
            std.debug.assert(indices.len == self.shape.len);
            const index = self.posToIndex(indices);
            self.data[index] = value;
        }

        fn posToIndex(self: Self, indices: []const usize) usize {
            std.debug.assert(indices.len == self.shape.len);
            var index: usize = 0;
            var stride: usize = 1;
            for (0..self.shape.len) |i| {
                const dim = self.shape.len - i - 1;
                const dimSize = self.shape[dim];
                const idx = indices[dim];
                std.debug.assert(idx < dimSize);

                index += idx * stride;
                stride *= dimSize;
            }
            return index;
        }

        fn indexToPos(self: Self, index: usize) []const usize {
            var pos: [settings.max_dim]usize = undefined;
            var remainingIndex = index;
            var stride: usize = 1;
            for (0..self.shape.len) |i| {
                stride *= self.shape[i];
            }

            for (0..self.shape.len) |i| {
                const dim = self.shape.len - i - 1;
                stride /= self.shape[dim];
                pos[dim] = remainingIndex / stride;
                remainingIndex %= stride;
            }

            return pos[0..self.shape.len];
        }

        pub fn print(self: *const Self) void {
            const alloc = std.heap.page_allocator;
            var shapeStr: []u8 = alloc.alloc(u8, self.shape.len * 2 - 1) catch unreachable;
            defer alloc.free(shapeStr);
            var j: usize = 0;
            for (self.shape) |s| {
                const b = std.fmt.formatIntBuf(shapeStr[j..shapeStr.len], s, 10, .lower, .{});
                if (j + b < shapeStr.len - 1) shapeStr[j + b] = 'x';
                j += 2;
            }

            std.debug.print("NDArray<{any},{s}>", .{ T, shapeStr });
            std.debug.print("{d}", .{self.data});
        }

        fn setOp(self: *Self, op: Op) *Self {
            self.op = op;
            return self;
        }

        pub fn add(self: *const Self, other: *const Self) Self {
            return Self{
                .shape = self.shape,
                .data = self.data + other.data,
            };
        }

        pub fn _add(self: *Self, other: *const Self) *Self {
            self.data = self.data + other.data;
            return self;
        }

        pub fn sub(self: *const Self, other: *const Self) Self {
            return Self{
                .shape = self.shape,
                .data = self.data - other.data,
            };
        }

        pub fn _sub(self: *Self, other: *const Self) *Self {
            self.data = self.data - other.data;
            return self;
        }

        pub fn mul(self: *const Self, other: *const Self) Self {
            return Self{
                .shape = self.shape,
                .data = self.data * other.data,
            };
        }

        pub fn _mul(self: *Self, other: *const Self) *Self {
            self.data = self.data * other.data;
            return self;
        }

        pub fn div(self: *const Self, other: *const Self) Self {
            return Self{
                .shape = self.shape,
                .data = self.data / other.data,
            };
        }

        pub fn _div(self: *Self, other: *const Self) *Self {
            self.data = self.data / other.data;
            return self;
        }
    };
}

pub fn NDTensor(comptime variant: DeviceTag, comptime T: type, size: usize) type {
    return struct {
        const dtype = Zarray(variant, T, size);
        const Self = @This();
        data: *dtype,
        op: ?Op = null,
        children: ?[2]*const Self = null,
        label: ?[]const u8 = null,
        grad: ?*dtype = null,
        requires_grad: bool,

        pub fn init(values: *[size]T, requires_grad: bool, allocator: std.mem.Allocator) *Self {
            const result = allocator.create(Self) catch {
                std.debug.panic("NDTensor allocation failed.\n", .{});
            };
            result.* = Self{
                .data = dtype.init(values.*, &[_]usize{values.len}, allocator) catch unreachable,
                .grad = if (requires_grad) dtype.initFill(0, null, allocator) catch unreachable else null,
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

        fn indexToPos(self: Self, index: usize) []const usize {
            return self.data.indexToPos(index);
        }

        pub fn print(self: *const Self) void {
            std.debug.print("NDTensor<{s},{?s}>[", .{ @tagName(variant), if (self.op) |o| @tagName(o) else null });
            std.debug.print("data: ", .{});
            self.data.print();
            if (self.grad) |g| {
                std.debug.print(" grad: ", .{});
                g.print();
            }
            std.debug.print("], requires_grad={}\n", .{self.requires_grad});
        }

        pub fn add(self: *const Self, other: *const Self, allocator: std.mem.Allocator) *Self {
            var result = Self.init(@constCast(&self.data.add(other.data).data), self.requires_grad, allocator);
            if (self.requires_grad) result.op = .ADD;
            result.children = .{ self, other };
            return result.reshape(self.data.shape) catch unreachable;
        }

        pub fn sub(self: *const Self, other: *const Self, allocator: std.mem.Allocator) *Self {
            var result = Self.init(@constCast(&self.data.sub(other.data).data), self.requires_grad, allocator);
            if (self.requires_grad) result.op = .SUB;
            result.children = .{ self, other };
            return result.reshape(self.data.shape) catch unreachable;
        }

        pub fn mul(self: *const Self, other: *const Self, allocator: std.mem.Allocator) *Self {
            var result = Self.init(@constCast(&self.data.mul(other.data).data), self.requires_grad, allocator);
            if (self.requires_grad) result.op = .MUL;
            result.children = .{ self, other };
            return result.reshape(self.data.shape) catch unreachable;
        }

        pub fn div(self: *const Self, other: *const Self, allocator: std.mem.Allocator) *Self {
            var result = Self.init(@constCast(&self.data.div(other.data).data), self.requires_grad, allocator);
            if (self.requires_grad) result.op = .DIV;
            result.children = .{ self, other };
            return result.reshape(self.data.shape) catch unreachable;
        }

        pub fn backward(self: *const Self) void {
            if (!self.requires_grad) return;
            if (self.op) |op| {
                switch (op) {
                    .ADD => {
                        if (self.children) |c| {
                            _ = c[0].grad.?._add(self.grad.?);
                            _ = c[1].grad.?._add(self.grad.?);
                        }
                    },
                    .SUB => {
                        if (self.children) |c| {
                            _ = c[0].grad.?._add(self.grad.?);
                            _ = c[1].grad.?._sub(self.grad.?);
                        }
                    },
                    .MUL => {
                        if (self.children) |c| {
                            _ = c[0].grad.?._add(self.grad.?)._mul(c[1].data);
                            _ = c[1].grad.?._add(self.grad.?)._mul(c[0].data);
                        }
                    },
                    .DIV => {
                        if (self.children) |c| {
                            var a = c[0].data;
                            var b = c[1].data;
                            // g/b
                            var temp = self.grad.?.div(b); // copies
                            _ = c[0].grad.?._add(&temp);
                            // a / b^2
                            const temp2 = @constCast(&(a.div(&b.mul(b))))._mul(self.grad.?); // copies
                            // += -(a/b^2) ==> -= a/b^2
                            _ = c[1].grad.?._sub(temp2);
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
        pub fn backward(self: *Self, node: *const T) void {
            self.sortedNodes.clearRetainingCapacity();
            self.visitedNodes.clearRetainingCapacity();
            self.topo(node);
            const nodes = self.sortedNodes.items;
            for (0..nodes.len) |i| {
                nodes[nodes.len - i - 1].backward();
            }
        }
    };
}

test "tensor/NDArray index, add, div" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const Array = NDArray(f32, 6);

    // 1 2 3
    // 4 5 6
    const shape = [_]usize{ 2, 3 };
    // const data: []const f32 = &.{ 1, 2, 3, 4, 5, 6 };
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var t1 = try Array.init(data, shape[0..shape.len], arena.allocator());
    t1.print();
    //
    // // 1 2 3
    // // 4 5 23
    // t1.set([_]u32{ 1, 2 }, 23);
    // std.debug.print("{d}\n", .{t1.get([_]u32{ 1, 2 })});
    // std.debug.print("{d}\n", .{t1.indexToPos(5)});
    //
    // const t2 = Array.init([]f32{ 10, 20, 30, 40, 50, 60 }, shape, alloc);
    // defer alloc.destroy(t2);
    //
    // var t3 = t1.add(t2, alloc);
    // defer alloc.destroy(t3);
    // t3.print();
    //
    // var t4 = t3.div(t3, alloc);
    // defer alloc.destroy(t4);
    // t4.print();
}

test "tensor/NDTensor index, add, div" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const shape = &[_]usize{ 2, 3 };
    // const Tensor = NDTensor(NDArray(f32));
    const Tensor = NDTensor(.cpu, f32, 6);

    // 1 2 3
    // 4 5 6
    var t1 = Tensor.init(@constCast(&[_]f32{ 1, 2, 3, 4, 5, 6 }), false, alloc);
    _ = try t1.reshape(shape);
    t1.print();

    // 1 2 3
    // 4 5 23
    try t1.set(&[_]usize{ 1, 2 }, 23);
    std.debug.print("{d}\n", .{t1.get(&[_]usize{ 1, 2 })});
    std.debug.print("{d}\n", .{t1.indexToPos(5)});

    const t2 = Tensor.init(@constCast(&[_]f32{ 10, 20, 30, 40, 50, 60 }), false, alloc);

    const t3 = t1.add(t2, alloc);

    // std.debug.print("{d}\n", .{t3.data.data.*});
    t3.print();
    // std.debug.print("{?any}\n", .{t3.children});

    var t4 = t3.div(t3, alloc);
    t4.print();
}

test "tensor/GraphManager/addback" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const shape = &[_]usize{1};
    const Tensor = NDTensor(.cpu, f32, 1);

    var t1 = try Tensor.init(@constCast(&[_]f32{2}), true, alloc).reshape(shape);
    const t2 = try Tensor.init(@constCast(&[_]f32{3}), true, alloc).reshape(shape);
    // t3 = t1 + t2;
    // dt3/dt1 = 1, dt3/dt2 = 1
    var t3 = t1.add(t2, alloc);

    var gm = Loss(Tensor).init(arena.allocator());
    defer gm.deinit();
    t3.grad = try Tensor.dtype.initFill(1, shape, alloc);
    gm.backward(t3);
    try std.testing.expectEqual(.{1}, t1.grad.?.data);
    try std.testing.expectEqual(.{1}, t2.grad.?.data);
}

test "tensor/GraphManager/mulback" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const shape = &[_]usize{1};
    const Tensor = NDTensor(.cpu, f32, 1);

    var t1 = try Tensor.init(@constCast(&[_]f32{2}), true, alloc).reshape(shape);
    const t2 = try Tensor.init(@constCast(&[_]f32{3}), true, alloc).reshape(shape);
    // t3 = t1 * t2;
    // dt3/dt1 = t2, dt3/dt2 = t1
    var t3 = t1.mul(t2, alloc);

    var gm = Loss(Tensor).init(arena.allocator());
    defer gm.deinit();
    t3.grad = try Tensor.dtype.initFill(1, shape, alloc);
    gm.backward(t3);
    try std.testing.expectEqual(t2.data.data, t1.grad.?.data);
    try std.testing.expectEqual(t1.data.data, t2.grad.?.data);
}

test "tensor/GraphManager/moreback" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const shape = &[_]usize{2};
    const Tensor = NDTensor(.cpu, f32, 2);

    var w = try Tensor.init(@constCast(&[_]f32{ 3, 2 }), true, alloc).reshape(shape);
    const b = try Tensor.init(@constCast(&[_]f32{ 1, 1 }), true, alloc).reshape(shape);
    const x = try Tensor.init(@constCast(&[_]f32{ 4, 4 }), true, alloc).reshape(shape);
    // const y = try Tensor.init(@constCast(&[_]f32{12, 10}), true, alloc).reshape(shape);
    // h = w*x + b
    // dh/dw = x, dh/db = 1
    var temp = w.mul(x, alloc);
    var h = temp.add(b, alloc);

    var gm = Loss(Tensor).init(arena.allocator());
    defer gm.deinit();
    h.grad = try Tensor.dtype.initFill(1, shape, alloc);
    gm.backward(h);
    try std.testing.expectEqual(x.data.data, w.grad.?.data);
    const ones = try Tensor.dtype.initFill(1, shape, alloc);
    try std.testing.expectEqual(ones.data, b.grad.?.data);

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
    temp = w.mul(x, alloc);
    h = temp.add(b, alloc);

    var gm2 = Loss(Tensor).init(arena.allocator());
    defer gm2.deinit();
    h.grad.?.fill(1);
    gm.backward(h);
    try std.testing.expectEqual(x.data.data, w.grad.?.data);
    try std.testing.expectEqual(ones.data, b.grad.?.data);
    h.print();
}
