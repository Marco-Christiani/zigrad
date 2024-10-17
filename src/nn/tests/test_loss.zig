const std = @import("std");
const json = std.json;
const zg = @import("../../zigrad.zig");

const Shape = zg.Shape;
const NDArray = zg.NDArray;
const NDTensor = zg.NDTensor;
const settings = zg.settings;
const ag_softmax_1d = zg.loss.ag_softmax_1d;
const softmax = zg.loss.softmax;

const SmceTestCase = struct {
    shape: []usize,
    input: []f32,
    target: []f32,
    loss: f32,
    input_grad: []f32,
};

const SmoothL1TestCase = struct {
    shape: []usize,
    input: []f32,
    target: []f32,
    beta: f32,
    loss: f32,
    input_grad: []f32,
};

const TestCases = struct {
    softmax_crossentropy_1d: SmceTestCase,
    softmax_crossentropy_2d: SmceTestCase,
    smooth_l1: SmoothL1TestCase,
};

fn verifySmceLoss(comptime name: []const u8, case: SmceTestCase, allocator: std.mem.Allocator) !void {
    var gm = zg.GraphManager(NDTensor(f32)).init(allocator, .{});
    defer gm.deinit();
    const input = try NDTensor(f32).init(case.input, case.shape, true, allocator);
    defer input.deinit();

    std.log.info("{s} {d}\n", .{ name, case.shape });
    std.log.info("input: {d}\n", .{input.data.data});
    std.log.info("target: {d}\n", .{case.target});

    const target = try NDTensor(f32).init(case.target, case.shape, true, allocator);
    defer target.deinit();

    const loss = try zg.loss.softmax_cross_entropy_loss(f32, input, target, allocator);
    loss.grad.?.fill(1.0);
    defer loss.deinit();
    try gm.backward(loss, allocator);

    std.log.info("loss: {d}\n", .{loss.data.data});
    std.log.info("expected: {d}\n", .{case.loss});
    std.log.info("input grad      : {d}\n", .{input.grad.?.data});
    std.log.info("expected grad   : {d}\n\n", .{case.input_grad});

    try std.testing.expectApproxEqAbs(case.loss, loss.data.data[0], 1e-4);

    const PRECISION = f16; // precision to compare grad slices at
    const grad_exp_trunc = try allocator.alloc(PRECISION, input.grad.?.data.len);
    const grad_actual_trunc = try allocator.alloc(PRECISION, input.grad.?.data.len);
    defer allocator.free(grad_exp_trunc);
    defer allocator.free(grad_actual_trunc);
    for (case.input_grad, input.grad.?.data, 0..) |e1, e2, i| {
        grad_exp_trunc[i] = @floatCast(e1);
        grad_actual_trunc[i] = @floatCast(e2);
    }
    try std.testing.expectEqualSlices(PRECISION, grad_exp_trunc, grad_actual_trunc);
}

test "softmax_cross_entropy_loss" {
    // const allocator = std.testing.allocator;
    var json_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer json_arena.deinit();
    const file = std.fs.openFileAbsolute("/tmp/loss_test_cases.json", .{}) catch |e| {
        std.log.warn("{s} error opening test file. Skipping `smce` test.", .{@errorName(e)});
        return;
    };
    defer file.close();

    const content = try file.readToEndAlloc(json_arena.allocator(), std.math.maxInt(usize));

    const allocator = json_arena.allocator();
    const test_cases = try json.parseFromSliceLeaky(TestCases, allocator, content, .{});
    try verifySmceLoss("softmax_crossentropy_1d", test_cases.softmax_crossentropy_1d, allocator);
    try verifySmceLoss("softmax_crossentropy_2d", test_cases.softmax_crossentropy_2d, allocator);
}

fn verifySmoothL1Loss(case: SmoothL1TestCase, allocator: std.mem.Allocator) !void {
    var gm = zg.GraphManager(NDTensor(f32)).init(allocator, .{});
    defer gm.deinit();
    const input = try NDTensor(f32).init(case.input, case.shape, true, allocator);
    defer input.deinit();

    const target = try NDTensor(f32).init(case.target, case.shape, true, allocator);
    defer target.deinit();

    std.log.info("Smooth L1 Loss Test", .{});
    std.log.info("Input: {d}", .{input.data.data});
    std.log.info("Target: {d}", .{target.data.data});
    std.log.info("Beta: {d}", .{case.beta});

    const loss = try zg.loss.smooth_l1_loss(f32, input, target, case.beta, allocator);
    loss.grad.?.fill(1.0);
    defer loss.deinit();

    try gm.backward(loss, allocator);

    std.log.info("Calculated loss: {d}n", .{loss.data.data[0]});
    std.log.info("Expected loss: {d}", .{case.loss});
    std.log.info("Calculated input grad: {d}", .{input.grad.?.data});
    std.log.info("Expected input grad: {d}", .{case.input_grad});

    try std.testing.expectApproxEqAbs(case.loss, loss.data.data[0], 1e-4);
    try std.testing.expectEqualSlices(f32, case.input_grad, input.grad.?.data);
}

test "smooth_l1_loss" {
    var json_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer json_arena.deinit();
    const file = std.fs.openFileAbsolute("/tmp/loss_test_cases.json", .{}) catch |e| {
        std.log.warn("{s} error opening test file. Skipping smooth L1 loss test.", .{@errorName(e)});
        return;
    };
    defer file.close();

    const content = try file.readToEndAlloc(json_arena.allocator(), std.math.maxInt(usize));

    const allocator = json_arena.allocator();
    const test_cases = try json.parseFromSliceLeaky(TestCases, allocator, content, .{});

    try verifySmoothL1Loss(test_cases.smooth_l1, allocator);
}
