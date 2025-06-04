// FIXME: still references the Trainer
const testing = std.testing;
const std = @import("std");

const zg = @import("../zigrad.zig");
const random = zg.random;
const NDTensor = zg.NDTensor;
const Op = zg.Op;
const GraphManager = zg.GraphManager;
const Conv2DLayer = zg.layer.Conv2DLayer;
const ReLULayer = zg.layer.ReLULayer;
const Model = zg.Model;
const Trainer = zg.Trainer;
const mse_loss = zg.loss.mse_loss;

pub const std_options = .{
    .log_level = std.log.Level.debug,
};

pub const zigrad_settings = zg.Settings{
    .grad_clip_enabled = false,
};

pub fn main() !void {
    std.log.warn("zigrad.settings: {}", .{zg.settings});
    // try test_conv_model();
    try test_model_fwd_bwd();
}

fn test_conv_model() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // 1x1x4x4 input (b, c, h, w)
    const input_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    const input = try NDTensor(f32).init(&input_data, &[_]usize{ 1, 1, 4, 4 }, true, allocator);
    _ = input.set_label("input");
    input.acquire(); // so we can do fwd pass and then demo the trainer

    const target = try NDTensor(f32).empty(&[_]usize{ 1, 1, 2, 2 }, true, allocator);
    target.fill(100);
    // const target = try NDTensor(f32).init(&[_]f32{100}, &[_]usize{ 1, 1, 1, 1 }, true, allocator);
    _ = target.set_label("target");
    target.acquire(); // so we can do fwd pass and then demo the trainer

    // Create a simple ConvNet: Conv2D -> ReLU -> Conv2D
    var model = try Model(f32).init(allocator);
    defer model.deinit();

    var conv1 = try Conv2DLayer(f32).init(allocator, 1, 2, 2, 1, 0, 1);
    var relu = try ReLULayer(f32).init(allocator);
    var conv2 = try Conv2DLayer(f32).init(allocator, 2, 1, 2, 1, 0, 1);

    conv1.weights.fill(0.01);
    conv1.bias.fill(0);
    conv2.weights.fill(0.01);
    conv2.bias.fill(0);

    try model.add_layer(conv1.as_layer());
    try model.add_layer(relu.as_layer());
    try model.add_layer(conv2.as_layer());

    // forward pass
    const output = try model.forward(input, allocator);
    std.debug.print("Output shape: {any}\n", .{output.data.shape.shape});
    std.debug.print("Output: {d}\n", .{output.data.data});

    // train step with Trainer

    var trainer = Trainer(f32, .mse).init(model, 0.01, .{});

    const loss = try trainer.train_step(input, target, allocator, allocator);
    defer {
        // destroy results from the manual forward pass, excluding input/target
        output.teardown();

        // now we can get ready to free these
        input.release();
        target.release();

        // tear down the graph created in train_step
        loss.teardown();

        trainer.deinit();
    }

    // TODO: control test against pytorch

    // not checking specific values, testing the backward pass runs without errors grads change
    // try testing.expect(loss > 0);
    try testing.expect(conv1.weights.grad != null);
    try testing.expect(conv1.bias.grad != null);
    try testing.expect(conv2.weights.grad != null);
    try testing.expect(conv2.bias.grad != null);

    std.debug.print("output:\n", .{});
    output.print();
    std.debug.print("\nloss:\n", .{});
    loss.print();
    std.debug.print("\nconv1.weights:\n", .{});
    conv1.weights.print();
    std.debug.print("\nconv1.bias:\n", .{});
    conv1.bias.print();
    std.debug.print("\nconv2.weights:\n", .{});
    conv2.weights.print();
    std.debug.print("\nconv2.bias:\n", .{});
    conv2.bias.print();
}

fn expect_tensor_approx_eql(expected: []const f32, actual: []const f32, tolerance: f32) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, a| {
        try testing.expectApproxEqAbs(e, a, tolerance);
    }
}

fn test_model_fwd_bwd() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    // 1x1x4x4 input (b, c, h, w)
    const input_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    const input = try NDTensor(f32).init(&input_data, &[_]usize{ 1, 1, 4, 4 }, true, allocator);
    _ = input.set_label("input");
    input.acquire(); // so we can do fwd pass and then demo the trainer

    const target = try NDTensor(f32).empty(&[_]usize{ 1, 1, 2, 2 }, true, allocator);
    target.fill(100);
    _ = target.set_label("target");
    target.acquire(); // so we can do fwd pass and then demo the trainer

    var model = try Model(f32).init(allocator);

    var conv1 = try Conv2DLayer(f32).init(allocator, 1, 1, 3, 2, 1, 1);

    conv1.weights.fill(0.01);
    conv1.bias.fill(0);

    try model.add_layer(conv1.as_layer());

    std.debug.print("input:\n", .{});
    input.print();
    std.debug.print("\nconv1.weights:\n", .{});
    conv1.weights.print();
    std.debug.print("\nconv1.bias:\n", .{});
    conv1.bias.print();

    // forward pass
    const output = try model.forward(input);
    std.debug.print("Output: {d}\n", .{output.data.data});
    std.debug.print("Output shape: {any}\n", .{output.data.shape.shape});
    const loss = try mse_loss(f32, output, target, model.allocator);
    std.debug.print("Output shape: {any}\n", .{output.data.shape.shape});
    var gm = GraphManager(NDTensor(f32)).init(allocator, .{ .grad_clip_enabled = false });
    model.zero_grad();
    loss.grad.?.fill(1);
    try gm.backward(loss, allocator);
    defer {
        gm.deinit();
        model.deinit();
        input.release();
        target.release();
        loss.teardown();
    }
    std.debug.print("Output shape: {any}\n", .{output.data.shape.shape});

    // not checking specific values, testing the backward pass runs without errors grads change
    // try testing.expect(loss > 0);
    try testing.expect(conv1.weights.grad != null);
    try testing.expect(conv1.bias.grad != null);

    std.debug.print("input:\n", .{});
    input.print();
    std.debug.print("output:\n", .{});
    output.print();
    std.debug.print("\nloss:\n", .{});
    loss.print();
    std.debug.print("\nconv1.weights:\n", .{});
    conv1.weights.print();
    std.debug.print("\nconv1.bias:\n", .{});
    conv1.bias.print();
}

fn test_conv_layer() !void {
    const allocator = testing.allocator;

    // Create a simple 1x3x3 input
    const input_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const input = try NDTensor(f32).init(&input_data, &[_]usize{ 1, 1, 3, 3 }, true, allocator);
    // defer input.deinit();

    // Create a Conv2DLayer with 1 input channel, 1 output channel, 2x2 kernel, stride 1, and no padding
    var conv = try Conv2DLayer(f32).init(allocator, 1, 1, 2, 1, 0, 1);
    defer conv.deinit();

    // Set weights to a 2x2 identity matrix and bias to 0
    try conv.weights.data.set(&[_]usize{ 0, 0, 0, 0 }, 1);
    try conv.weights.data.set(&[_]usize{ 0, 0, 0, 1 }, 0);
    try conv.weights.data.set(&[_]usize{ 0, 0, 1, 0 }, 0);
    try conv.weights.data.set(&[_]usize{ 0, 0, 1, 1 }, 1);
    conv.bias.data.data[0] = 0;

    // Perform forward pass
    const output = try conv.forward(input, allocator);
    defer {
        std.debug.print("conv2d tearing down\n", .{});
        output.teardown();
    }

    // Expected output: 2x2 matrix
    const expected_output = [_]f32{ 6, 8, 12, 14 };

    try expect_tensor_approx_eql(&expected_output, output.data.data, 1e-6);
}

fn test_relu_layer() !void {
    const allocator = testing.allocator;

    // Create a simple 2x2 input
    const input_data = [_]f32{ -1, 2, 3, -4 };
    const input = try NDTensor(f32).init(&input_data, &[_]usize{ 1, 1, 2, 2 }, true, allocator);
    // defer input.deinit();

    var relu = try ReLULayer(f32).init(allocator);
    defer relu.deinit();

    // Perform forward pass
    const output = try relu.forward(input, allocator);
    defer output.teardown();

    // Expected output: negative values replaced with 0
    const expected_output = [_]f32{ 0, 2, 3, 0 };

    try expect_tensor_approx_eql(&expected_output, output.data.data, 1e-6);
}

test "Simple ConvNet forward and backward pass" {
    try test_conv_model();
}

test "ReLULayer forward pass" {
    try test_relu_layer();
}

test "Conv2DLayer forward pass" {
    try test_conv_layer();
}
