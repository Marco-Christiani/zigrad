const std = @import("std");
const zg = @import("../zigrad.zig");
const NDTensor = zg.NDTensor;
const GraphManager = zg.GraphManager;
const ops = @import("loss.zig");
const Layer = @import("layer.zig").Layer;
const LinearLayer = @import("layer.zig").LinearLayer;
const ReLULayer = @import("layer.zig").ReLULayer;
const read_csv = @import("layer.zig").read_csv;
const SGD = zg.optim.SGD;

pub fn MLP(comptime T: type) type {
    return struct {
        const Self = @This();
        const Tensor = NDTensor(T);

        layers: std.ArrayList(Layer(T)),
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, layer_sizes: []const usize) !*Self {
            const self = try allocator.create(Self);
            self.* = Self{
                .layers = std.ArrayList(Layer(T)).init(allocator),
                .allocator = allocator,
            };

            for (0..layer_sizes.len - 1) |i| {
                const curr_ll = try LinearLayer(T).init(allocator, layer_sizes[i], layer_sizes[i + 1]);
                try self.layers.append(curr_ll.as_layer());
                if (i < layer_sizes.len - 1) try self.layers.append((try ReLULayer(T).init(allocator)).as_layer());
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

        pub fn forward(self: *const Self, input: *Tensor, alloc: std.mem.Allocator) !*Tensor {
            var x = input;
            for (self.layers.items) |layer| {
                const layer_output = try layer.forward(x, alloc);
                // if (x != input) {
                //     x.deinit(self.allocator);
                // }
                x = layer_output;

                // if (layer != self.layers.items[self.layers.items.len - 1]) {
                //     x = try ops.relu(T, x, self.allocator);
                // }
            }
            return x;
        }

        pub fn zero_grad(self: *const Self) void {
            for (self.layers.items) |layer| {
                layer.zero_grad();
            }
        }

        pub fn parameters(self: *const Self) ![]*Tensor {
            var params = std.ArrayList(*Tensor).init(self.allocator);
            for (self.layers.items) |layer| {
                if (layer.get_parameters()) |layer_params| {
                    for (layer_params) |p| try params.append(p);
                }
            }
            return try params.toOwnedSlice();
        }
    };
}

pub fn train_mlp(comptime T: type, data: [][]T, layer_sizes: []const usize, alloc: std.mem.Allocator) !void {
    const Tensor = NDTensor(T);

    const mlp = try MLP(T).init(alloc, layer_sizes);
    defer mlp.deinit();

    const params = try mlp.parameters();
    defer alloc.free(params);
    defer for (params) |p| p.release();

    var gm = GraphManager(NDTensor(T)).init(alloc, .{});
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
    defer input.deinit();
    var target = try Tensor.init(try alloc.alloc(T, output_size * batch_size), &[_]usize{ output_size, batch_size }, true, alloc);
    defer target.deinit();

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
            const output = try mlp.forward(input, alloc);
            const loss = try ops.mse_loss(T, output, target, alloc);
            // defer loss.deinit(alloc);
            total_loss += loss.data.data[0];
            loss.print_arrows();

            loss.grad.?.fill(1.0);
            mlp.zero_grad();
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
//     const data = try read_csv(T, "/tmp/data.csv", alloc);
//     std.debug.print("data.len={}\n", .{data.len});
//
//     // const layer_sizes = &[_]usize{ 2, 64, 32, 1 };
//     const layer_sizes = &[_]usize{ 2, 64, 32, 1 };
//     try train_mlp(T, data, layer_sizes, alloc);
// }

fn generate_xordataset(T: type, n: usize, alloc: std.mem.Allocator) ![][]T {
    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();
    const arr = try alloc.alloc([]T, n);
    for (0..n) |i| {
        arr[i] = try alloc.alloc(T, 3);
        const a: T = if (random.float(T) < 0.5) 0 else 1;
        const b: T = if (random.float(T) < 0.5) 0 else 1;
        arr[i][0] = a;
        arr[i][1] = b;
        arr[i][2] = if ((a == 1 and b == 0) or (a == 0 and b == 1)) 1 else 0;
    }
    return arr;
}

test "MLP/XOR" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const T = f32;

    std.debug.print("Generating XOR dataset\n", .{});
    const data = try generate_xordataset(T, 1000, alloc);
    defer {
        for (0..data.len) |i| alloc.free(data[i]);
        alloc.free(data);
    }
    std.debug.print("data.len={}\n", .{data.len});

    const layer_sizes = &[_]usize{ 2, 64, 32, 1 };
    try train_mlp(T, data, layer_sizes, alloc);

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
        const input = try NDTensor(T).init(case[0..2], &[_]usize{ 1, 2 }, true, alloc);
        defer input.deinit();
        const output = try mlp.forward(input, alloc);
        defer output.deinit();
        std.debug.print("Input: {d}, {d} - Expected: {d}, Got: {d:.6}\n", .{ case[0], case[1], case[2], output.data.data[0] });
    }
}
