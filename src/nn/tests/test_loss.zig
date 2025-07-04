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

fn verify_smce_loss(comptime name: []const u8, case: SmceTestCase, allocator: std.mem.Allocator) !void {
    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = zg.Graph.init(allocator, .{});
    defer graph.deinit();

    const config: zg.TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const input = try NDTensor(f32).from_slice(device, case.input, case.shape, config);
    defer input.deinit();

    std.log.info("{s} {d}\n", .{ name, case.shape });
    std.log.info("input: {d}\n", .{input.get_data()});
    std.log.info("target: {d}\n", .{case.target});

    const target = try NDTensor(f32).from_slice(device, case.target, case.shape, config);
    defer target.deinit();

    const loss = try zg.loss.softmax_cross_entropy_loss(f32, input, target);
    defer loss.deinit();

    try loss.backward();

    std.log.info("loss: {d}\n", .{loss.get_data()});
    std.log.info("expected: {d}\n", .{case.loss});
    std.log.info("input grad      : {d}\n", .{input.assume_grad_data()});
    std.log.info("expected grad   : {d}\n\n", .{case.input_grad});

    try std.testing.expectApproxEqAbs(case.loss, loss.get(0), 1e-4);

    const PRECISION = f16; // precision to compare grad slices at
    const grad_exp_trunc = try device.mem_alloc(PRECISION, input.assume_grad().size());
    const grad_actual_trunc = try device.mem_alloc(PRECISION, input.assume_grad().size());
    defer device.mem_free(grad_exp_trunc);
    defer device.mem_free(grad_actual_trunc);
    for (case.input_grad, input.assume_grad_data(), 0..) |e1, e2, i| {
        grad_exp_trunc[i] = @floatCast(e1);
        grad_actual_trunc[i] = @floatCast(e2);
    }
    try std.testing.expectEqualSlices(PRECISION, grad_exp_trunc, grad_actual_trunc);
}

test "softmax_cross_entropy_loss" {
    // const allocator = std.testing.allocator;
    var json_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer json_arena.deinit();
    const file = std.fs.openFileAbsolute("/tmp/zg_data/loss_test_cases.json", .{}) catch |e| {
        std.log.warn("{s} error opening test file. Skipping `smce` test.", .{@errorName(e)});
        return;
    };
    defer file.close();

    const content = try file.readToEndAlloc(json_arena.allocator(), std.math.maxInt(usize));

    const allocator = json_arena.allocator();
    const test_cases = try json.parseFromSliceLeaky(TestCases, allocator, content, .{});
    try verify_smce_loss("softmax_crossentropy_1d", test_cases.softmax_crossentropy_1d, allocator);
    try verify_smce_loss("softmax_crossentropy_2d", test_cases.softmax_crossentropy_2d, allocator);
}

fn verify_smooth_l1_loss(case: SmoothL1TestCase, allocator: std.mem.Allocator) !void {
    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = zg.Graph.init(allocator, .{});
    defer graph.deinit();

    const config: zg.TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const input = try NDTensor(f32).from_slice(device, case.input, case.shape, config);
    defer input.deinit();

    const target = try NDTensor(f32).from_slice(device, case.target, case.shape, config);
    defer target.deinit();

    std.log.info("Smooth L1 Loss Test", .{});
    std.log.info("Input: {d}", .{input.get_data()});
    std.log.info("Target: {d}", .{target.get_data()});
    std.log.info("Beta: {d}", .{case.beta});

    const loss = try zg.loss.smooth_l1_loss(f32, input, target, case.beta);
    defer loss.deinit();

    try loss.backward();

    std.log.info("Calculated loss: {d}n", .{loss.get(0)});
    std.log.info("Expected loss: {d}", .{case.loss});
    std.log.info("Calculated input grad: {d}", .{input.assume_grad_data()});
    std.log.info("Expected input grad: {d}", .{case.input_grad});

    try std.testing.expectApproxEqAbs(case.loss, loss.get(0), 1e-4);
    try std.testing.expectEqualSlices(f32, case.input_grad, input.assume_grad_data());
}

test "smooth_l1_loss" {
    var json_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer json_arena.deinit();
    const file = std.fs.openFileAbsolute("/tmp/zg_data/loss_test_cases.json", .{}) catch |e| {
        std.log.warn("{s} error opening test file. Skipping smooth L1 loss test.", .{@errorName(e)});
        return;
    };
    defer file.close();

    const content = try file.readToEndAlloc(json_arena.allocator(), std.math.maxInt(usize));

    const allocator = json_arena.allocator();
    const test_cases = try json.parseFromSliceLeaky(TestCases, allocator, content, .{});

    try verify_smooth_l1_loss(test_cases.smooth_l1, allocator);
}
