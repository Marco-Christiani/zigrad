const std = @import("std");
const NDTensor = @import("tensor.zig").NDTensor;
const Loss = @import("tensor.zig").Loss;
const SGD = @import("tensor.zig").SGD;
const ops = @import("ops.zig");

pub fn LinearLayer(comptime T: type) type {
    return struct {
        const Self = @This();
        const Tensor = NDTensor(T);
        weights: *Tensor,
        bias: *Tensor,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, input_size: usize, output_size: usize) !*Self {
            // TODO: random init and, maybe, stack bias (benchmark)
            const self = try allocator.create(Self);
            const weights_shape = &[_]usize{ output_size, input_size };
            const bias_shape = &[_]usize{output_size};
            const weights = (try Tensor.init(try allocator.alloc(T, input_size * output_size), weights_shape, true, allocator)).setLabel("layer.weights");
            const bias = (try Tensor.init(try allocator.alloc(T, output_size), bias_shape, true, allocator)).setLabel("layer.bias");
            weights.acquire();
            bias.acquire();
            weights.fill(1.0);
            bias.fill(1.0);

            self.* = Self{
                .weights = weights,
                .bias = bias,
                .allocator = allocator,
            };

            return self;
        }

        pub fn forward(self: *const Self, input: *Tensor) !*Tensor {
            var result = (try self.weights.matmul(input, self.allocator)).setLabel("fwd1.1");
            return (try result.add(self.bias, self.allocator)).setLabel("fwd1.2");
        }

        pub fn zeroGrad(self: *const Self) void {
            self.weights.grad.?.fill(0);
            self.bias.grad.?.fill(0);
        }
    };
}

pub fn readCsv(comptime T: type, file_path: []const u8, alloc: std.mem.Allocator) ![][]T {
    var file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    var reader = file.reader();
    const bytes = try reader.readAllAlloc(alloc, 1 << 32);
    var line_iter = std.mem.tokenizeScalar(u8, bytes, '\n');

    var arr = std.ArrayList([]T).init(alloc);
    var row_arr = std.ArrayList(T).init(alloc);

    while (line_iter.next()) |line| {
        var row_iter = std.mem.tokenizeScalar(u8, line, ',');
        while (row_iter.next()) |value| {
            try row_arr.append(try std.fmt.parseFloat(T, value));
        }
        try arr.append(try row_arr.toOwnedSlice());
    }

    return try arr.toOwnedSlice();
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const T = f32;

    std.debug.print("Reading data\n", .{});
    const data = try readCsv(T, "/tmp/data.csv", alloc);
    std.debug.print("data.len={}\n", .{data.len});
    try trainGradAccum(T, data, alloc);
}

fn trainGradAccum(comptime T: type, data: [][]T, alloc: std.mem.Allocator) !void {
    const Tensor = NDTensor(T);
    const input_size = 2;
    const output_size = 1;

    const layer = try LinearLayer(T).init(alloc, input_size, output_size);
    const params = [_]*NDTensor(f32){ layer.weights, layer.bias };

    var gm = Loss(NDTensor(T)).init(alloc);
    gm.eager_teardown = true;
    gm.grad_clip_enabled = true;
    gm.grad_clip_delta = 1e-6;
    gm.grad_clip_max_norm = 10.0;
    defer gm.deinit();
    var optimizer = SGD(T){ .lr = 0.01 };
    const lr_epoch_decay = 0.9;

    const grad_acc_steps = 32;
    var loss: *Tensor = try Tensor.init(&[_]T{0.0}, null, true, alloc);
    // this is a trick to avoid reallocating
    // const input = try Tensor.init(@constCast(&[_]T{0} ** input_size), &[_]usize{input_size}, true, alloc);
    // const target = try Tensor.init(@constCast(&[_]T{0} ** output_size), &[_]usize{output_size}, true, alloc);
    // input.acquire();
    // target.acquire();

    for (0..10) |epoch| {
        var start_i: usize = 0;
        while (start_i < data.len) : (start_i += grad_acc_steps) {
            const end_i = @min(start_i + grad_acc_steps, data.len);
            loss.fill(0);
            loss.grad.?.fill(0);
            for (data[start_i..end_i]) |row| {
                // clever way, but NEED to zero these grads
                // input.data.data = row[0..input_size];
                // target.data.data = row[row.len - output_size .. row.len];
                // this is the more intuitive, less performant method but this keeps all prev ones on the graph which may be req in some cases
                const input = try Tensor.init(row[0..input_size], &[_]usize{ input_size, output_size }, true, alloc);
                const target = try Tensor.init(row[input_size..row.len], null, true, alloc);
                const output = try layer.forward(input);

                // std.debug.print("input:", .{});
                // input.data.print();
                // std.debug.print("\ntarget:", .{});
                // target.data.print();
                // std.debug.print("\noutput:", .{});
                // output.data.print();
                // _ = try output.data.shape.squeeze();
                // _ = try target.data.shape.squeeze();
                // output.grad.?.shape.squeeze();

                // const curr_loss = try ops.mse_loss1(f32, output, target, alloc); // memory error
                const curr_loss = try ops.mse_loss(f32, output, target, alloc); // wont converge
                // var curr_loss: *Tensor = undefined;
                // if (output.get(&[_]usize{ 0, 0 }) > target.get(&[_]usize{0})) {
                //     curr_loss = try output.sub(target, alloc);
                // } else {
                //     curr_loss = try target.sub(output, alloc);
                // }
                _ = try loss.data._add(curr_loss.data);
                curr_loss.grad.?.fill(1.0 / @as(T, grad_acc_steps));
                try gm.backward(curr_loss, alloc);
            }

            optimizer.step(&params);
            layer.zeroGrad();
        }
        optimizer.lr *= lr_epoch_decay;
        std.debug.print("Epoch: {d:<4.4}", .{epoch + 1});
        std.debug.print("AccLoss: {d:<8.4}\t", .{loss.data.data});
        std.debug.print("Weights: {d:>8.4}\t", .{layer.weights.data.data});
        std.debug.print("Bias: {d:.4}\n", .{layer.bias.data.data});
    }
}

test "LinearLayer/trainGradAccum" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const T = f32;

    std.debug.print("Reading data\n", .{});
    const data = try readCsv(T, "/tmp/data.csv", alloc);
    std.debug.print("data.len={}\n", .{data.len});
    try trainGradAccum(T, data, alloc);
}

// TODO: finish implementing trainBatched v1, v2 works though
fn trainBatched(comptime T: type, data: [][]T, alloc: std.mem.Allocator) !void {
    const X = try alloc.alloc([]T, data.len);
    const y = try alloc.alloc(T, data.len);
    const n_features = 2;
    for (0..data.len) |i| {
        X[i] = try alloc.alloc(T, n_features);
        for (0..n_features) |j| {
            X[i][j] = data[i][j];
        }

        y[i] = data[i][n_features];
    }
    const Tensor = NDTensor(T);
    const input_size = 2;
    const output_size = 1;
    const batch_size = 100;

    const layer = try LinearLayer(T).init(alloc, input_size, output_size);
    const params = [_]*NDTensor(T){ layer.weights, layer.bias };

    var gm = Loss(NDTensor(T)).init(alloc);
    gm.eager_teardown = false;
    gm.grad_clip_enabled = true;
    gm.grad_clip_max_norm = 100.0;
    defer gm.deinit();
    var optimizer = SGD(T){ .lr = 0.001 };

    var flattened = try alloc.alloc(T, batch_size * n_features);

    for (0..3) |_| {
        var batch_start_i: usize = 0;
        while (batch_start_i < data.len) : (batch_start_i += batch_size) {
            const batch_end_i = @min(batch_start_i + batch_size, data.len);
            const batchx = X[batch_start_i..batch_end_i];
            const batchy = y[batch_start_i..batch_end_i];

            for (0..batchx.len) |i| {
                for (0..batchx[0].len) |j| flattened[i * n_features + j] = batchx[i][j];
            }

            const input = (try Tensor.init(flattened, &[_]usize{ input_size, batch_size }, true, alloc)).setLabel("input");
            const target = (try Tensor.init(batchy, &[_]usize{ output_size, batch_size }, true, alloc)).setLabel("target");
            const output = (try layer.forward(input)).setLabel("output");
            // output.print();
            // std.debug.print("\n\n", .{});

            // output.data.shape.squeeze();
            // std.debug.print("input shape={d} output shape={d} target shape={d}\n", .{ input.data.shape.shape, output.data.shape.shape, target.data.shape.shape });

            const loss = try ops.mse_loss(T, output, target, alloc);
            loss.grad.?.fill(1.0);

            // var curr_loss = try output.sub(target, alloc);
            // const curr_loss = try diff.mul(diff, alloc);
            // var curr_loss: *Tensor = undefined;
            // if (output.get(&[_]usize{0}) > target.get(&[_]usize{0})) {
            //     curr_loss = try output.sub(target, alloc);
            // } else {
            //     curr_loss = try target.sub(output, alloc);
            // }
            // _ = try loss.data._add(curr_loss.data);
            // curr_loss.grad.?.fill(1.0 / @as(T, batch_size));
            layer.zeroGrad();
            try gm.backward(loss, alloc);
            // loss.print_arrows();
            optimizer.step(&params);
            loss.teardown(alloc);

            std.debug.print("Loss: {d:>10.4}\t", .{loss.data.data});
            std.debug.print("Weights: {d:>10.4}\t", .{layer.weights.data.data});
            std.debug.print("Bias: {d:>10.4}\n", .{layer.bias.data.data});
            // std.debug.print("preds: {d}\n", .{output.data.data});
            // std.debug.print("Epoch {d}, Loss: {d}\n", .{ epoch, loss.data.data });
        }
    }

    std.debug.print("FINAL Weights: {d:>10.4}\t", .{layer.weights.data.data});
    std.debug.print("FINAL Bias: {d:>10.4}\n", .{layer.bias.data.data});
}

fn trainBatched2(comptime T: type, data: [][]T, alloc: std.mem.Allocator) !void {
    const X = try alloc.alloc([]T, data.len);
    const y = try alloc.alloc(T, data.len);
    const n_features = 2;
    for (0..data.len) |i| {
        X[i] = try alloc.alloc(T, n_features);
        for (0..n_features) |j| {
            X[i][j] = data[i][j];
        }

        y[i] = data[i][n_features];
    }
    const Tensor = NDTensor(T);
    const batch_size = 100;
    const input_size = 2;
    const output_size = 1;

    const layer = try LinearLayer(T).init(alloc, input_size, output_size);
    const params = [_]*NDTensor(f32){ layer.weights, layer.bias };
    // try layer.weights.set(&[_]usize{ 0, 0 }, 3);
    // try layer.weights.set(&[_]usize{ 0, 1 }, -0.5);

    // this is a trick to avoid reallocating
    const input = try Tensor.init(@constCast(&[_]T{0} ** (batch_size * input_size)), &[_]usize{ input_size, batch_size }, true, alloc);
    const target = try Tensor.init(@constCast(&[_]T{0} ** (batch_size * output_size)), &[_]usize{ output_size, batch_size }, true, alloc);
    input.acquire();
    target.acquire();

    var gm = Loss(NDTensor(T)).init(alloc);
    gm.eager_teardown = true;
    gm.grad_clip_enabled = true;
    gm.grad_clip_max_norm = 100.0;
    defer gm.deinit();
    var optimizer = SGD(T){ .lr = 0.001 };

    var loss: *Tensor = undefined;

    for (0..3) |e| {
        var batch_start_i: usize = 0;
        while (batch_start_i < y.len) : (batch_start_i += batch_size) {
            input.grad.?.fill(0.0);
            target.grad.?.fill(0.0);
            const batch_end_i = @min(batch_start_i + batch_size, y.len);
            const batch_x = X[batch_start_i..batch_end_i];
            const batch_y = y[batch_start_i..batch_end_i];

            for (0..batch_x.len) |bi| {
                for (0..input_size) |bj| {
                    // std.debug.print("bi * input_size + bj = {d} bi:{d} bj:{d}\n", .{ bi * input_size + bj, bi, bj });
                    try input.set(&[_]usize{ bj, bi }, batch_x[bi][bj]);
                }
            }

            target.data.data = batch_y;
            // this is the more intuitive, less performance method
            // const input = try Tensor.init(row[0..input_size], null, true, alloc);
            // const target = try Tensor.init(row[row.len - output_size .. row.len], null, true, alloc);
            const output = try layer.forward(input);
            // output.print();
            // std.debug.print("\n\n", .{});

            // loss = try ops.mse_loss1(f32, output, target, alloc);
            loss = try ops.mse_loss(f32, output, target, alloc);
            loss.grad.?.fill(1.0);
            layer.zeroGrad();
            try gm.backward(loss, alloc);
            optimizer.step(&params);
            std.debug.print("{s}\nweights: ", .{"-" ** 80});
            layer.weights.data.print();
            std.debug.print("\nbiases: ", .{});
            layer.bias.data.print();
            std.debug.print("\n\tEpoch {d}, Loss: {d}\n", .{ e, loss.data.data });
        }
    }

    std.debug.print("Weights: ", .{});
    layer.weights.print();
    std.debug.print("Biases: ", .{});
    layer.bias.print();
    std.debug.print("\n", .{});
}

test "LinearLayer/trainBatched" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const T = f32;

    std.debug.print("Reading data\n", .{});
    const data = try readCsv(T, "/tmp/data.csv", alloc);
    std.debug.print("data.len={}\n", .{data.len});
    try trainBatched2(T, data, alloc);
    try trainBatched(T, data, alloc);
}

test "LinearLayer forward and backward" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f32;

    // input 2, output 2
    var layer = try LinearLayer(T).init(alloc, 2, 2);

    const input_shape = &[_]usize{ 2, 1 };
    const input = try NDTensor(T).init(@constCast(&[_]T{ 3, 3 }), input_shape, true, alloc);

    const output = try layer.forward(input);
    @memset(output.grad.?.data, 1.0);

    var gm = Loss(NDTensor(T)).init(alloc);
    defer gm.deinit();
    try gm.backward(output, alloc);

    std.debug.print("Weights: ", .{});
    layer.weights.print();
    std.debug.print("bias: ", .{});
    layer.bias.print();
}
