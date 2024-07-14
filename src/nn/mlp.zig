const std = @import("std");
const NDTensor = @import("tensor.zig").NDTensor;
const Loss = @import("tensor.zig").Loss;
const ops = @import("ops.zig");
const Layer = @import("layer.zig");
const LinearLayer = Layer.LinearLayer;
const readCsv = Layer.readCsv;
const SGD = @import("tensor.zig").SGD;
const dataset = @import("dataset.zig");

pub fn MLP(comptime T: type) type {
    return struct {
        const Self = @This();
        const Tensor = NDTensor(T);

        layers: std.ArrayList(*LinearLayer(T)),
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, layer_sizes: []const usize) !*Self {
            const self = try allocator.create(Self);
            self.* = Self{
                .layers = std.ArrayList(*LinearLayer(T)).init(allocator),
                .allocator = allocator,
            };

            for (0..layer_sizes.len - 1) |i| {
                const layer = try LinearLayer(T).init(allocator, layer_sizes[i], layer_sizes[i + 1]);
                try self.layers.append(layer);
            }

            return self;
        }

        pub fn deinit(self: *Self) void {
            for (self.layers.items) |layer| {
                layer.deinit();
            }
            self.layers.deinit();
            self.allocator.destroy(self);
        }

        pub fn forward(self: *const Self, input: *Tensor) !*Tensor {
            var x = input;
            for (self.layers.items) |layer| {
                const layer_output = try layer.forward(x);
                // if (x != input) {
                //     x.deinit(self.allocator);
                // }
                x = layer_output;

                // Apply ReLU activation to all but the last layer
                if (layer != self.layers.items[self.layers.items.len - 1]) {
                    x = try ops.relu(T, x, self.allocator);
                }
            }
            return x;
        }

        pub fn zeroGrad(self: *const Self) void {
            for (self.layers.items) |layer| {
                layer.zeroGrad();
            }
        }

        pub fn parameters(self: *const Self) ![]*Tensor {
            var params = std.ArrayList(*Tensor).init(self.allocator);
            for (self.layers.items) |layer| {
                params.append(layer.weights) catch unreachable;
                params.append(layer.bias) catch unreachable;
            }
            return try params.toOwnedSlice();
        }
    };
}

pub fn trainMLP(comptime T: type, data: [][]T, layer_sizes: []const usize, alloc: std.mem.Allocator) !void {
    const Tensor = NDTensor(T);

    const mlp = try MLP(T).init(alloc, layer_sizes);
    defer mlp.deinit();

    const params = try mlp.parameters();
    defer alloc.free(params);
    defer for (params) |p| p.release();

    var gm = Loss(NDTensor(T)).init(alloc);
    gm.eager_teardown = false;
    gm.grad_clip_enabled = true;
    gm.grad_clip_delta = 1e-6;
    gm.grad_clip_max_norm = 10.0;
    defer gm.deinit();

    var optimizer = SGD(T){ .lr = 0.01 };
    const lr_epoch_decay = 0.9;

    const batch_size = @min(data.len, 32);
    const input_size = layer_sizes[0];
    const output_size = layer_sizes[layer_sizes.len - 1];

    var input = try Tensor.init(try alloc.alloc(T, input_size * batch_size), &[_]usize{ input_size, batch_size }, true, alloc);
    defer input.deinit(alloc);
    var target = try Tensor.init(try alloc.alloc(T, output_size * batch_size), &[_]usize{ output_size, batch_size }, true, alloc);
    defer target.deinit(alloc);

    const epochs = 10;
    for (0..epochs) |epoch| {
        var total_loss: T = 0;
        var batch_start_i: usize = 0;
        while (batch_start_i < data.len) : (batch_start_i += batch_size) {
            input.grad.?.fill(0.0);
            target.grad.?.fill(0.0);
            const batch_end_i = @min(batch_start_i + batch_size, data.len);
            const batch = data[batch_start_i..batch_end_i];

            for (0..batch.len) |bi| {
                for (0..input_size) |bj| {
                    try input.set(&[_]usize{ bj, bi }, batch[bi][bj]);
                }
                for (0..output_size) |bj| {
                    try target.set(&[_]usize{ bj, bi }, batch[bi][input_size + bj]);
                }
            }
            std.debug.print("\n", .{});
            input.print();
            std.debug.print("\n", .{});
            const output = try mlp.forward(input);
            const loss = try ops.mse_loss(T, output, target, alloc);
            // defer loss.deinit(alloc);
            total_loss += loss.data.data[0];
            loss.print_arrows();

            loss.grad.?.fill(1.0);
            mlp.zeroGrad();
            try gm.backward(loss, alloc);
            optimizer.step(params);
        }
        optimizer.lr *= lr_epoch_decay;
        if (epoch % 1 == 0) {
            std.debug.print("Epoch {d}, Loss: {d:.6}\n", .{ epoch + 1, total_loss / @as(T, @floatFromInt(data.len)) });
        }
    }
}

// test "MLP/train" {
//     var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
//     const alloc = arena.allocator();
//     defer arena.deinit();
//     const T = f32;
//
//     std.debug.print("Reading data\n", .{});
//     const data = try readCsv(T, "/tmp/data.csv", alloc);
//     std.debug.print("data.len={}\n", .{data.len});
//
//     // const layer_sizes = &[_]usize{ 2, 64, 32, 1 };
//     const layer_sizes = &[_]usize{ 2, 64, 32, 1 };
//     try trainMLP(T, data, layer_sizes, alloc);
// }

test "MLP/XOR" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const T = f32;

    std.debug.print("Generating XOR dataset\n", .{});
    const data = try dataset.generateXORDataset(T, 1000, alloc);
    defer dataset.freeDataset(data, alloc);
    std.debug.print("data.len={}\n", .{data.len});

    const layer_sizes = &[_]usize{ 2, 64, 32, 1 };
    try trainMLP(T, data, layer_sizes, alloc);

    // Test the trained model
    const mlp = try MLP(T).init(alloc, layer_sizes);
    defer mlp.deinit();

    const test_cases = [_][3]T{
        .{ 0, 0, 0 },
        .{ 0, 1, 1 },
        .{ 1, 0, 1 },
        .{ 1, 1, 0 },
    };

    for (test_cases) |case| {
        const input = try NDTensor(T).init(case[0..2], &[_]usize{2}, true, alloc);
        defer input.deinit(alloc);
        const output = try mlp.forward(input);
        defer output.deinit(alloc);
        std.debug.print("Input: {d}, {d} - Expected: {d}, Got: {d:.6}\n", .{ case[0], case[1], case[2], output.data.data[0] });
    }
}
