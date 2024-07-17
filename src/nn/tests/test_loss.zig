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

const SmceTestCase = struct {
    shape: []usize,
    input: []f32,
    target: []f32,
    loss: f32,
    input_grad: []f32,
};

const SmceTestCases = struct {
    softmax_crossentropy_1d: SmceTestCase,
    softmax_crossentropy_2d: SmceTestCase,
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

    // try std.testing.expectApproxEqAbs(case.loss, loss.data.data[0], 1e-4);
    var gm = zg.GraphManager(NDTensor(f32)).init(allocator, .{});

    try gm.backward(loss, allocator);

    // try expectApproxSlices(f32, case.input_grad, input.grad.?.data);
    std.debug.print("loss: {d}\n", .{loss.data.data});
    std.debug.print("expected: {d}\n", .{case.loss});
    std.debug.print("input grad      : {d}\n", .{input.grad.?.data});
    std.debug.print("expected grad   : {d}\n\n", .{case.input_grad});
}

fn expectApproxSlices(comptime T: type, comptime msg: []const u8, expected: []const T, actual: []const T) !void {
    const stderr = std.io.getStdErr();
    const writer = stderr.writer();
    defer stderr.close();
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, a| {
        std.testing.expectApproxEqAbs(e, a, 1e-4) catch {
            try writer.print("[{s}] Expected: {d}\n", .{ msg, expected });
            try writer.print("[{s}] Actual: {d}\n", .{ msg, actual });
            // return err;
        };
    }
}

test "softmax and cross-entropy tests" {
    // const allocator = std.testing.allocator;
    var json_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer json_arena.deinit();
    const file = try std.fs.cwd().openFile("/tmp/softmax_crossentropy_test_cases.json", .{});
    defer file.close();

    const content = try file.readToEndAlloc(json_arena.allocator(), std.math.maxInt(usize));

    const allocator = json_arena.allocator();
    const test_cases = try json.parseFromSliceLeaky(SmceTestCases, allocator, content, .{});
    try runTest("softmax_crossentropy_1d", test_cases.softmax_crossentropy_1d, allocator);
    try runTest("softmax_crossentropy_2d", test_cases.softmax_crossentropy_2d, allocator);
}

// test "hmm" {
//     var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
//     defer arena.deinit();
//     const allocator = arena.allocator();
//     const input = try NDTensor(f32).init(.{ -0.3917, 0.2034, 0.1089, 0.1066, -0.5150, -0.6448, -0.0028, 0.0944, 0.5641, 0.3965 }, null, true, allocator);
//     defer input.deinit();
//     const loss = try cross_entropy_loss(f32, input, target, allocator);
//     loss.grad.?.fill(1.0);
//     defer loss.deinit();
//
//     // try std.testing.expectApproxEqAbs(case.loss, loss.data.data[0], 1e-4);
//     var gm = zg.GraphManager(NDTensor(f32)).init(allocator, .{});
//
//     try gm.backward(loss, allocator);
//
//     // try expectApproxSlices(f32, case.input_grad, input.grad.?.data);
//     std.debug.print("loss: {d}\n", .{loss.data.data});
//     std.debug.print("expected: {d}\n", .{case.loss});
//     std.debug.print("input grad      : {d}\n", .{input.grad.?.data});
//     std.debug.print("expected grad   : {d}\n\n", .{case.input_grad});
// }
