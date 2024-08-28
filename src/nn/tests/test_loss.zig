const std = @import("std");
const json = std.json;
const zg = @import("../../root.zig");

const Shape = zg.Shape;
const NDArray = zg.NDArray;
const NDTensor = zg.NDTensor;
const settings = zg.settings;
const ag_softmax_1d = zg.loss.ag_softmax_1d;
const softmax = zg.loss.softmax;
const cross_entropy_loss = zg.loss.softmax_cross_entropy_loss;
const smooth_l1_loss = zg.loss.smooth_l1_loss;

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

fn runTest(comptime name: []const u8, case: SmceTestCase, allocator: std.mem.Allocator) !void {
    const input = try NDTensor(f32).init(case.input, case.shape, true, allocator);
    defer input.deinit();

    std.debug.print("{s} {d}\n", .{ name, case.shape });
    std.debug.print("input: {d}\n", .{input.data.data});
    std.debug.print("target: {d}\n", .{case.target});

    const target = try NDTensor(f32).init(case.target, case.shape, true, allocator);
    defer target.deinit();

    const loss = try cross_entropy_loss(f32, input, target, allocator);
    loss.grad.?.fill(1.0);
    defer loss.deinit();

    try std.testing.expectApproxEqAbs(case.loss, loss.data.data[0], 1e-4);
    var gm = zg.GraphManager(NDTensor(f32)).init(allocator, .{});

    try gm.backward(loss, allocator);

    std.debug.print("loss: {d}\n", .{loss.data.data});
    std.debug.print("expected: {d}\n", .{case.loss});
    std.debug.print("input grad      : {d}\n", .{input.grad.?.data});
    std.debug.print("expected grad   : {d}\n\n", .{case.input_grad});
}

test "softmax and cross-entropy tests" {
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
    try runTest("softmax_crossentropy_1d", test_cases.softmax_crossentropy_1d, allocator);
    try runTest("softmax_crossentropy_2d", test_cases.softmax_crossentropy_2d, allocator);
}

fn runSmoothL1Test(case: SmoothL1TestCase, allocator: std.mem.Allocator) !void {
    const input = try NDTensor(f32).init(case.input, case.shape, true, allocator);
    defer input.deinit();

    const target = try NDTensor(f32).init(case.target, case.shape, true, allocator);
    defer target.deinit();

    std.debug.print("Smooth L1 Loss Test\n", .{});
    std.debug.print("Input: {d}\n", .{input.data.data});
    std.debug.print("Target: {d}\n", .{target.data.data});
    std.debug.print("Beta: {d}\n", .{case.beta});

    const loss = try smooth_l1_loss(f32, input, target, case.beta, allocator);
    loss.grad.?.fill(1.0);
    defer loss.deinit();

    var gm = zg.GraphManager(NDTensor(f32)).init(allocator, .{});
    defer gm.deinit();

    try gm.backward(loss, allocator);

    try std.testing.expectApproxEqAbs(case.loss, loss.data.data[0], 1e-4);

    std.debug.print("Calculated loss: {d}\n", .{loss.data.data[0]});
    std.debug.print("Expected loss: {d}\n", .{case.loss});
    std.debug.print("Calculated input grad: {d}\n", .{input.grad.?.data});
    std.debug.print("Expected input grad: {d}\n\n", .{case.input_grad});
}

test "smooth L1 loss" {
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

    try runSmoothL1Test(test_cases.smooth_l1, allocator);
}
