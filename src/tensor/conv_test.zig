const testing = std.testing;
const std = @import("std");
const NDTensor = @import("tensor.zig").NDTensor;
const Conv2DLayer = @import("layer.zig").Conv2DLayer;
const ReLULayer = @import("layer.zig").ReLULayer;
const Model = @import("model.zig").Model;
const Trainer = @import("trainer.zig").Trainer;
const ops = @import("ops.zig");

pub fn main() !void {
    try testConvModel();
}

fn testConvModel() !void {
    // const allocator = std.heap.page_allocator;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        if (gpa.deinit() == .leak) std.debug.print("Leak detected.", .{});
    }
    // Create a simple 1x1x4x4 input (batch_size x channels x height x width)
    const input_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    const input = try NDTensor(f32).init(&input_data, &[_]usize{ 1, 1, 4, 4 }, true, allocator);
    input.label = "input";
    defer input.deinit(allocator);
    std.debug.print("Input tensor shape: {any}\n", .{input.data.shape.shape});

    // Create a simple ConvNet: Conv2D -> ReLU -> Conv2D
    var model = try Model(f32).init(allocator);
    defer model.deinit();

    var conv1 = try Conv2DLayer(f32).init(allocator, 1, 2, 2, 1, 0);
    var relu = try ReLULayer(f32).init(allocator);
    var conv2 = try Conv2DLayer(f32).init(allocator, 2, 1, 2, 1, 0);

    try model.addLayer(conv1.asLayer());
    try model.addLayer(relu.asLayer());
    try model.addLayer(conv2.asLayer());

    // Perform forward pass
    const output = try model.forward(input);
    defer output.deinit(allocator);

    // Create a simple loss and perform backward pass
    const target = try NDTensor(f32).init(&[_]f32{100}, &[_]usize{ 1, 1, 1, 1 }, true, allocator);
    target.label = "target";
    defer target.deinit(allocator);

    const TrainerT = Trainer(f32);
    var trainer = TrainerT.init(model, 0.01, allocator);
    defer trainer.deinit();

    const loss = try trainer.trainStep(input, target);

    // Not checking specific values, just ensuring that the backward pass runs without errors
    // and updates gradients
    try testing.expect(loss > 0);
    try testing.expect(conv1.weights.grad != null);
    try testing.expect(conv1.bias.grad != null);
    try testing.expect(conv2.weights.grad != null);
    try testing.expect(conv2.bias.grad != null);

    std.debug.print("{d}\n", .{output.data.data});
}

fn expectTensorApproxEql(expected: []const f32, actual: []const f32, tolerance: f32) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, a| {
        try testing.expectApproxEqAbs(e, a, tolerance);
    }
}

fn testConvLayer() !void {
    const allocator = testing.allocator;

    // Create a simple 1x3x3 input
    const input_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const input = try NDTensor(f32).init(&input_data, &[_]usize{ 1, 1, 3, 3 }, true, allocator);
    defer input.deinit(allocator);

    // Create a Conv2DLayer with 1 input channel, 1 output channel, 2x2 kernel, stride 1, and no padding
    var conv = try Conv2DLayer(f32).init(allocator, 1, 1, 2, 1, 0);
    defer conv.deinit(allocator);

    // Set weights to a 2x2 identity matrix and bias to 0
    try conv.weights.data.set(&[_]usize{0}, 1);
    try conv.weights.data.set(&[_]usize{1}, 0);
    try conv.weights.data.set(&[_]usize{2}, 0);
    try conv.weights.data.set(&[_]usize{3}, 1);
    conv.bias.data.data[0] = 0;

    // Perform forward pass
    const output = try conv.forward(input, allocator);
    defer output.deinit(allocator);

    // Expected output: 2x2 matrix
    const expected_output = [_]f32{ 6, 8, 12, 14 };

    try expectTensorApproxEql(&expected_output, output.data.data, 1e-6);
}

fn testReluLayer() !void {
    const allocator = testing.allocator;

    // Create a simple 2x2 input
    const input_data = [_]f32{ -1, 2, 3, -4 };
    const input = try NDTensor(f32).init(&input_data, &[_]usize{ 1, 1, 2, 2 }, true, allocator);
    defer input.deinit(allocator);

    var relu = try ReLULayer(f32).init(allocator);
    defer relu.deinit(allocator);

    // Perform forward pass
    const output = try relu.forward(input, allocator);
    defer output.deinit(allocator);

    // Expected output: negative values replaced with 0
    const expected_output = [_]f32{ 0, 2, 3, 0 };

    try expectTensorApproxEql(&expected_output, output.data.data, 1e-6);
}

test "Simple ConvNet forward and backward pass" {
    try testConvModel();
}

test "ReLULayer forward pass" {
    try testReluLayer();
}

test "Conv2DLayer forward pass" {
    try testConvLayer();
}
