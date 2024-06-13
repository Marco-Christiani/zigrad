// TODO: implement gradient forward and backward
// TODO: since the comptime ideas were trashed, shape/size should not be comptime
// TODO: implement view(), transpose(), and permute(), where the latter two mutate the shape
const std = @import("std");
const zarray = @import("zarray.zig");
const generateASCIIUUID = @import("grad.zig").generateASCIIUUID;
const c = @cImport(@cInclude("Accelerate/Accelerate.h"));
const root = @import("root");
const Shape = zarray.Shape;
const NDArray = zarray.NDArray;
const ZarrayError = zarray.ZarrayError;

/// lib-wide options that can be overridden by the root file.
pub const settings: Settings = if (@hasDecl(root, "zigrad_settings")) root.zigrad_settings else .{};

const Settings = struct {
    grad_enabled: bool = true,
    max_dim: usize = 4,
};

pub const Op = enum { ADD, SUB, MUL, DIV, POW, TANH, MATMUL, DOT, MATVEC, SUM };

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
        acquired: bool = false,

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

        pub fn deinit(self: *const Self, allocator: std.mem.Allocator) void {
            // TODO: tbd whether to recurse to children w new arch
            if (self.acquired) std.debug.panic("Attempt to deinit an acquired tensor.", .{});
            self.data.deinit(allocator);
            if (self.grad) |g| g.deinit(allocator);
        }

        pub fn acquire(self: *Self) void {
            self.acquired = true;
        }

        pub fn release(self: *Self) void {
            self.acquired = false;
        }

        pub fn fromZarray(values: *dtype, requires_grad: bool, grad_shape: ?Shape, allocator: std.mem.Allocator) !*Self {
            const result = try allocator.create(Self);
            const grad_size = if (grad_shape) grad_shape.?.size() else values.len;
            const zeros = try allocator.alloc(T, grad_size);
            @memset(zeros, 0);
            result.* = Self{
                .data = values,
                .grad = if (requires_grad) try dtype.init(zeros, grad_shape, allocator) else null,
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
            return self.data.posToOffset(indices);
        }

        fn flexPosToIndex(self: Self, indices: []const usize) ZarrayError.InvalidIndex!usize {
            return self.data.flexPosToOffset(indices);
        }

        fn indexToPos(self: Self, index: usize, allocator: std.mem.Allocator) []const usize {
            return self.data.offsetToPos(index, allocator);
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
            const grad_shape = if (self.requires_grad) self.grad.?.shape else null;
            var result = try Self.fromZarray(try self.data.add(other.data, allocator), self.requires_grad, grad_shape, allocator);
            if (self.requires_grad) result.op = .ADD;
            result.children = .{ self, other };
            return result;
        }

        /// Element-wise subtraction
        pub fn sub(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const grad_shape = if (self.requires_grad) self.grad.?.shape else null;
            var result = try Self.fromZarray(try self.data.sub(other.data, allocator), self.requires_grad, grad_shape, allocator);
            if (self.requires_grad) result.op = .SUB;
            result.children = .{ self, other };
            return result.reshape(self.data.shape.shape) catch unreachable;
        }

        /// Element-wise multiplication
        pub fn mul(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const grad_shape = if (self.requires_grad) self.grad.?.shape else null;
            var result = try Self.fromZarray(try self.data.mul(other.data, allocator), self.requires_grad, grad_shape, allocator);
            if (self.requires_grad) result.op = .MUL;
            result.children = .{ self, other };
            return result;
        }

        /// Element-wise division
        pub fn div(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const grad_shape = if (self.requires_grad) self.grad.?.shape else null;
            var result = try Self.fromZarray(try self.data.div(other.data, allocator), self.requires_grad, grad_shape, allocator);
            if (self.requires_grad) result.op = .DIV;
            result.children = .{ self, other };
            return result;
        }

        pub fn matmul(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const grad_shape = if (self.requires_grad) self.grad.?.shape else null;
            var result = try Self.fromZarray(try self.data.matmul(other.data, false, false, allocator), self.requires_grad, grad_shape, allocator);
            if (self.requires_grad) result.op = .MATMUL;
            result.children = .{ self, other };
            return result;
        }

        // pub fn dot(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
        //     var result = try Self.fromZarray(try self.data.dot(other.data, allocator), self.requires_grad, allocator);
        //     if (self.requires_grad) result.op = .DOT;
        //     result.children = .{ self, other };
        //     return result;
        // }

        pub fn dot(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const dot_result = try self.data.dot(other.data, allocator);

            // set the gradient dimensions to match the operands
            const result = try allocator.create(Self);
            result.* = Self{
                .data = dot_result,
                .grad = if (self.requires_grad) try dtype.zerosLike(self.data, allocator) else null,
                .requires_grad = self.requires_grad,
                .op = .DOT,
                .children = .{ self, other },
            };
            return result;
        }

        pub fn matvec(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const grad_shape = if (self.requires_grad) other.grad.?.shape else null;
            var result = try Self.fromZarray(try self.data.matvec(other.data, false, allocator), self.requires_grad, grad_shape, allocator);
            if (self.requires_grad) result.op = .MATVEC;
            result.children = .{ self, other };
            return result;
        }

        pub fn sum(self: *const Self, allocator: std.mem.Allocator) !*Self {
            const ones = try allocator.alloc(T, self.data.data.len);
            @memset(ones, 1);
            const ones_tensor = try Self.init(ones, self.data.shape.shape, self.requires_grad, allocator);
            const result = try self.dot(ones_tensor, allocator);
            // if (self.requires_grad) result.op = .SUM;
            // result.children = .{ self, ones_tensor };
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
                            // _ = try (try children[0].grad.?._add(self.grad.?))._mul(children[1].data);
                            // _ = try (try children[1].grad.?._add(self.grad.?))._mul(children[0].data);
                            _ = try children[0].grad.?._add(try self.grad.?.mul(children[1].data, allocator));
                            _ = try children[1].grad.?._add(try self.grad.?.mul(children[0].data, allocator));
                        }
                    },
                    .DIV => {
                        if (self.children) |children| {
                            var a = children[0].data;
                            const b = children[1].data;

                            // wrt numerator
                            _ = try (try children[0].grad.?._add(self.grad.?))._div(b);

                            // wrt denominator
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
                            const grad_A = try self.grad.?.matmul(B, false, true, allocator);
                            _ = try children[0].grad.?._add(grad_A);

                            // gradient wrt B: Must be free'd, assumes owner uses an arena for bw and does not return handle
                            const grad_B = try A.matmul(self.grad.?, true, false, allocator);
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

                            // Gradient with respect to A
                            // Must be free'd, assumes owner uses an arena for bw and does not return handle
                            //  L(y), y = Ax, dL/dA = (dL/dy)(dy/dA) = (dL/dy)x'
                            const grad_A = try self.grad.?.outer(x, allocator);
                            _ = try children[0].grad.?._add(grad_A);

                            // Gradient with respect to x
                            // Must be free'd, assumes owner uses an arena for bw and does not return handle
                            //  L(y), y = Ax, dL/dx = (dL/dy)(dy/dx) = A'(dL/dy)
                            const grad_x = try A.matvec(self.grad.?, true, allocator);
                            _ = try children[1].grad.?._add(grad_x);
                        }
                    },
                    // .MATVEC => {
                    //     if (self.children) |children| {
                    //         var A = children[0].data;
                    //         const x = children[1].data;
                    //
                    //         // Gradient with respect to A: grad_A = outer(self.grad.?, x)
                    //         const grad_A = try self.grad.?.outer(x, allocator);
                    //         _ = try children[0].grad.?._add(grad_A);
                    //
                    //         // Gradient with respect to x: grad_x = matvec(transpose(A), self.grad.?)
                    //         const grad_x = try A.matvec(self.grad.?, true, allocator);
                    //         _ = try children[1].grad.?._add(grad_x);
                    //     }
                    // },
                    .SUM => {
                        if (self.children) |children| {
                            const child = children[0];
                            _ = try child.grad.?._add(self.grad.?);
                        }
                        // if (self.children) |children| {
                        //     const child = children[0];
                        //     const grad_shape = child.data.shape;
                        //     const grad_size = child.data.data.len;
                        //     const grad_values = try allocator.alloc(T, grad_size);
                        //     for (0..grad_size) |i| {
                        //         grad_values[i] = self.grad.?.data[0];
                        //     }
                        //     const grad_tensor = try Self.init(grad_values, grad_shape, false, allocator);
                        //     _ = try child.grad.?._add(grad_tensor.data);
                        // }
                    },
                    else => @panic("Not yet implemented."),
                }
            }
        }
    };
}

pub fn NeuronLayer(comptime T: type) type {
    return struct {
        const Self = @This();
        weights: *NDTensor(T),
        biases: *NDTensor(T),
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, input_size: usize, output_size: usize) !*Self {
            // TODO: random init
            const self = try allocator.create(Self);
            const weights_shape = &[_]usize{ output_size, input_size };
            const biases_shape = &[_]usize{output_size};
            const weights = try NDTensor(T).init(try allocator.alloc(T, input_size * output_size), weights_shape, true, allocator);
            const biases = try NDTensor(T).init(try allocator.alloc(T, output_size), biases_shape, true, allocator);
            weights.acquire();
            biases.acquire();
            weights.fill(1);
            biases.fill(1);

            self.* = Self{
                .weights = weights,
                .biases = biases,
                .allocator = allocator,
            };

            return self;
        }

        pub fn forward(self: *const Self, input: *NDTensor(T)) !*NDTensor(T) {
            var result = try self.weights.matvec(input, self.allocator);
            // std.debug.print("\tself.weights.matvec(input) = {d}\n", .{result.data.data});
            result = try result.add(self.biases, self.allocator);
            // std.debug.print("\tresult.add(self.biases) = {d}\n", .{result.data.data});
            return result;
        }

        pub fn zeroGrad(self: *const Self) void {
            self.weights.grad.?.fill(0);
            self.biases.grad.?.fill(0);
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
        eagerTeardown: bool = false,

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
                // node.print();
            }
        }

        // Must init grad on root node before backprop
        pub fn backward(self: *Self, node: *const T, alloc: std.mem.Allocator) !void {
            self.sortedNodes.clearRetainingCapacity();
            self.visitedNodes.clearRetainingCapacity();
            self.topo(node);
            const nodes = self.sortedNodes.items;
            for (0..nodes.len) |i| {
                const curr_node = nodes[nodes.len - i - 1];
                // curr_node.print();
                try curr_node.backward(alloc);
                if (!curr_node.acquired and self.eagerTeardown) curr_node.deinit(alloc);
            }
        }
    };
}

// pub fn mse_loss(T: type, y_pred: *NDTensor(T), y: *NDTensor(T), allocator: std.mem.Allocator) !*NDTensor(T) {
//     var diff = try y_pred.sub(y, allocator);
//     const sq_diff = try diff.mul(diff, allocator);
//     // const loss = try sq_diff.mean(allocator);
//     // const loss = (try sq_diff.sum(allocator))._div();
//     const coef = try allocator.alloc(T, y.data.data.len);
//     @memset(coef, 1.0 / @as(T, @floatFromInt(y.data.data.len)));
//     const coef_tensor = try NDTensor(T).init(coef, null, true, allocator);
//     const loss = coef_tensor.dot(sq_diff, allocator);
//     // const num_elements = try NDTensor(T).init(&[_]T{@floatFromInt(y.data.data.len)}, null, false, allocator);
//     // const mean_sq_diff = try sum_sq_diff.div(num_elements, allocator);
//     return loss;
// }

pub fn mse_loss(T: type, y_pred: *NDTensor(T), y: *NDTensor(T), allocator: std.mem.Allocator) !*NDTensor(T) {
    var diff = try y_pred.sub(y, allocator);
    var sq_diff = try diff.mul(diff, allocator);
    var sum_sq_diff = try sq_diff.sum(allocator);
    const n = @as(T, @floatFromInt(y.data.data.len));
    const num_elements = try NDTensor(T).init(&[_]T{n}, null, true, allocator);
    const mean_sq_diff = try sum_sq_diff.div(num_elements, allocator);
    return mean_sq_diff;
}

pub fn SGD(comptime T: type) type {
    return struct {
        const Self = @This();
        lr: T,

        pub fn step(self: Self, params: []const *NDTensor(T)) void {
            // lol. go to bed.
            for (params) |param| {
                // param.data._sub(param.grad.?._mul(self.lr)); // TODO: really need to implement scalar ops...
                for (0..param.data.data.len) |j| {
                    param.data.data[j] -= self.lr * param.grad.?.data[j];
                }
            }
        }
    };
}

pub fn train(
    layer: *NeuronLayer(f32),
    input: *NDTensor(f32),
    target: *NDTensor(f32),
    loss_manager: *Loss(NDTensor(f32)),
    optimizer: *SGD(f32),
    alloc: std.mem.Allocator,
    num_epochs: usize,
) !void {
    for (0..num_epochs) |epoch| {
        const output = try layer.forward(input);
        const params = [_]*NDTensor(f32){ layer.weights, layer.biases };

        const loss = try mse_loss(f32, output, target, alloc);
        std.debug.print("Epoch {d}, Loss: {d}\n", .{ epoch, loss.data.data });
        std.debug.print("Predictions: ", .{});
        output.print();

        loss.grad.?.fill(1.0);
        layer.zeroGrad();
        try loss_manager.backward(loss, alloc);
        optimizer.step(&params);

        std.debug.print("Params: ", .{});
        for (params) |p| p.print();
        std.debug.print("\n\n", .{});
    }
}

test "NeuronLayer training" {
    std.debug.print("{s} Layer train {s}\n", .{ "-" ** 5, "-" ** 5 });
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f32;

    const layer = try NeuronLayer(T).init(alloc, 2, 2);

    const input_shape = &[_]usize{2};
    const input = try NDTensor(T).init(@constCast(&[_]T{ 1, 1 }), input_shape, true, alloc);
    const target = try NDTensor(T).init(@constCast(&[_]T{ 10, 10 }), input_shape, true, alloc);
    input.acquire();
    target.acquire();

    var gm = Loss(NDTensor(T)).init(alloc);
    gm.eagerTeardown = true;
    defer gm.deinit();
    var optimizer = SGD(T){ .lr = 0.01 };

    try train(layer, input, target, gm, &optimizer, alloc, 10);
}

test "NeuronLayer forward and backward" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f32;

    // input 2, output 2
    var layer = try NeuronLayer(T).init(alloc, 2, 2);

    const input_shape = &[_]usize{2};
    const input = try NDTensor(T).init(@constCast(&[_]T{ 3, 3 }), input_shape, true, alloc);

    const output = try layer.forward(input);
    @memset(output.grad.?.data, 1.0);

    var gm = Loss(NDTensor(T)).init(alloc);
    defer gm.deinit();
    try gm.backward(output, alloc);

    std.debug.print("Weights: ", .{});
    layer.weights.print();
    std.debug.print("Biases: ", .{});
    layer.biases.print();
}

test "MSE" {
    std.debug.print("{s} MSE {s}\n", .{ "-" ** 5, "-" ** 5 });
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f32;
    const Tensor = NDTensor(T);

    const y = try Tensor.init(&[_]T{1}, null, false, alloc);
    const yh = try Tensor.init(&[_]T{1}, null, false, alloc);
    var loss = try mse_loss(T, yh, y, alloc);
    try std.testing.expectEqualSlices(T, &[_]T{0}, loss.data.data);
    // (1-2)**2 == 1
    try yh.set(&[_]usize{0}, 2);
    loss = try mse_loss(T, yh, y, alloc);
    try std.testing.expectEqualSlices(T, &[_]T{1}, loss.data.data);

    // (1-3)**2 == 4
    try yh.set(&[_]usize{0}, 3);
    loss = try mse_loss(T, yh, y, alloc);
    try std.testing.expectEqualSlices(T, &[_]T{4}, loss.data.data);

    // (-1-10)**2 == 121
    try y.set(&[_]usize{0}, -1);
    try yh.set(&[_]usize{0}, 10);
    loss = try mse_loss(T, yh, y, alloc);
    try std.testing.expectEqualSlices(T, &[_]T{121}, loss.data.data);

    // Arrays
    const y_arr = try Tensor.init(&[_]T{ 1, 2, 3 }, null, true, alloc);
    const yh_arr = try Tensor.init(&[_]T{ 1, 2, 3 }, null, true, alloc);
    loss = try mse_loss(T, yh_arr, y_arr, alloc);
    try std.testing.expectEqualSlices(T, &[_]T{0}, loss.data.data);

    // ((1-2)^2 + (2-3)^2 + (3-4)^2) / 3 == (1 + 1 + 1) / 3 == 1
    try yh_arr.set(&[_]usize{0}, 2);
    try yh_arr.set(&[_]usize{1}, 3);
    try yh_arr.set(&[_]usize{2}, 4);
    loss = try mse_loss(T, yh_arr, y_arr, alloc);
    try std.testing.expectEqualSlices(T, &[_]T{1}, loss.data.data);

    // MSE backward
    var gm = Loss(Tensor).init(arena.allocator());
    defer gm.deinit();
    // loss.grad.?.fill(1.0);
    loss.acquire();
    try gm.backward(loss, alloc);
    loss.print();
}

test "sum operation" {
    std.debug.print("{s} sum {s}\n", .{ "-" ** 5, "-" ** 5 });
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f32;
    const Tensor = NDTensor(T);

    const input = try Tensor.init(&[_]T{ 1, 2, 3, 4 }, &[_]usize{4}, true, alloc);
    var sum_result = try input.sum(alloc);

    try std.testing.expectEqualSlices(T, &[_]T{10}, sum_result.data.data);

    // Backward pass
    var gm = Loss(Tensor).init(alloc);
    defer gm.deinit();
    sum_result.grad.?.fill(1.0);
    try gm.backward(sum_result, alloc);

    const expected_grad = &[_]T{ 1, 1, 1, 1 };
    try std.testing.expectEqualSlices(T, expected_grad, input.grad.?.data);
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
    t1.acquire();
    t2.acquire();
    t3.acquire();

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
    t1.acquire();
    t2.acquire();
    t3.acquire();

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
    t1.acquire();
    t2.acquire();
    t3.acquire();

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
    t1.acquire();
    t2.acquire();
    t3.acquire();

    var gm = Loss(Tensor).init(arena.allocator());
    defer gm.deinit();
    t3.grad.?.fill(1.0);
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

    // [1, 2] [1]
    // [3, 4] [1]
    // grad = [1, 1]'
    // dl/dA = grad * [1, 1] = [[2, 2], [2, 2]]
    // dl/dx = A' * grad = [4, 6]'
    var t1 = try Tensor.init(@constCast(&[_]T{ 1, 2, 3, 4 }), shape_mat, true, alloc);
    const t2 = try Tensor.init(@constCast(&[_]T{ 1, 1 }), shape_vec, true, alloc);
    var t3 = try t1.matvec(t2, alloc);
    t1.acquire();
    t2.acquire();
    t3.acquire();

    var gm = Loss(Tensor).init(arena.allocator());
    defer gm.deinit();
    t3.grad.?.fill(1.0);
    try gm.backward(t3, alloc);
    std.debug.print("t1: ", .{});
    t1.print();
    std.debug.print("t2: ", .{});
    t2.print();
    std.debug.print("t3: ", .{});
    t3.print();

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
    t1.acquire();
    t2.acquire();
    t3.acquire();

    var gm = Loss(Tensor).init(arena.allocator());
    defer gm.deinit();
    t3.grad.?.fill(1.0);
    try gm.backward(t3, alloc);

    const expected_grad_t1 = &[_]T{ 4, 5, 6 };
    const expected_grad_t2 = &[_]T{ 1, 2, 3 };

    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);
}

pub fn linear_regression_forward(T: type, W: *NDTensor(T), b: *NDTensor(T), X: *NDTensor(T), allocator: std.mem.Allocator) !*NDTensor(T) {
    var Wx = try W.matvec(X, allocator);
    const y_pred = try Wx.add(b, allocator);
    return y_pred;
}

test "linear regression training" {
    std.debug.print("{s} LR {s}\n", .{ "-" ** 5, "-" ** 5 });
    const T = f32;
    const Tensor = NDTensor(T);
    // one sample
    const X = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const y = [_]f32{ 2.0, 4.0, 6.0, 8.0 };

    const shape_X = &[_]usize{4};
    const shape_y = &[_]usize{4};
    const shape_w = &[_]usize{ 4, 4 };
    const shape_b = &[_]usize{4};

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const W = try Tensor.init(&[_]T{0.0} ** 16, shape_w, true, alloc);
    const b = try Tensor.init(&[_]T{0.0} ** 4, shape_b, true, alloc);

    const X_tensor = try Tensor.init(&X, shape_X, true, alloc);
    const y_tensor = try Tensor.init(&y, shape_y, true, alloc);

    W.acquire();
    b.acquire();
    X_tensor.acquire();
    y_tensor.acquire();

    const learning_rate: T = 0.01;
    var gm = Loss(Tensor).init(arena.allocator());
    defer gm.deinit();

    for (0..100) |epoch| {
        const y_pred = try linear_regression_forward(T, W, b, X_tensor, alloc);

        const loss = try mse_loss(T, y_pred, y_tensor, alloc);
        loss.grad.?.fill(1.0);
        W.grad.?.fill(0.0);
        b.grad.?.fill(0.0);
        try gm.backward(loss, alloc);

        for (0..W.data.data.len) |i| W.data.data[i] -= learning_rate * W.grad.?.data[i];
        for (0..b.data.data.len) |i| b.data.data[i] -= learning_rate * b.grad.?.data[i];

        if (epoch % 10 == 0) std.debug.print("Epoch {}, Loss: {d}\n", .{ epoch, loss.data.data });
    }

    // Evaluation
    const y_pred = try linear_regression_forward(T, W, b, X_tensor, alloc);
    std.debug.print("Predictions:\n", .{});
    y_pred.data.print();
    std.debug.print("\nActual:\n", .{});
    y_tensor.data.print();
    std.debug.print("\nFinal weights:\n", .{});
    std.debug.print("\nW\n", .{});
    W.data.print();
    std.debug.print("\nBias\n", .{});
    b.data.print();
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const shape1 = &[_]usize{ 2, 3 };
    const shape2 = &[_]usize{ 3, 2 };
    const Tensor = NDTensor(.cpu, f32, 6);

    // var A = try Tensor.init(@constCast(&[_]f32{ 3, 1, 1, 2 }), true, alloc).reshape(shape);
    // const B = try Tensor.init(@constCast(&[_]f32{ 1, 1, 1, 1 }), true, alloc).reshape(shape);
    // var C = A.mul(B, alloc);
    //
    // var gm = Loss(Tensor).init(arena.allocator());
    // defer gm.deinit();
    // C.print();
    // multiply a 2x3 and a 3x2 to get a 2x2
    var A = try Tensor.init(@constCast(&[_]f32{ 1, 2, 3, 4, 5, 6 }), true, alloc).reshape(shape1);
    const B = try Tensor.init(@constCast(&[_]f32{ 1, 0, 0, 1, 0, 0 }), true, alloc).reshape(shape2);
    var C = A.mul(B, alloc);
    C.print();

    // var gm = Loss(Tensor).init(arena.allocator());
    // defer gm.deinit();
    // h.grad = try Tensor.dtype.initFill(1, shape, alloc);
    // gm.backward(h);
    // try std.testing.expectEqual(x.data.data, w.grad.?.data);
    // const ones = try Tensor.dtype.initFill(1, shape, alloc);
    // try std.testing.expectEqual(ones.data, b.grad.?.data);

    // 2 x 1
    // const shape2 = &[_]usize{ 2, 1 };
    // w.grad.?.fill(0);
    // b.grad.?.fill(0);
    // x.grad.?.fill(0);
    // _ = try w.reshape(shape2);
    // _ = try b.reshape(shape2);
    // _ = try x.reshape(shape2);
    // h = w*x + b
    // dh/dw = x, dh/db = 1
    // temp = w.mul(x, alloc);
    // h = temp.add(b, alloc);
    //
    // var gm2 = Loss(Tensor).init(arena.allocator());
    // defer gm2.deinit();
    // h.grad.?.fill(1);
    // gm.backward(h);
    // try std.testing.expectEqual(x.data.data, w.grad.?.data);
    // try std.testing.expectEqual(ones.data, b.grad.?.data);
    // h.print();
}
