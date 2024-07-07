const std = @import("std");
const zg = @import("zigrad");

const zarray = zg.zarray;
const Shape = zarray.Shape;
const NDArray = zarray.NDArray;
const ZarrayError = zarray.ZarrayError;
const settings = zg.settings;

const NDTensor = @import("tensor.zig").NDTensor;
const Loss = @import("tensor.zig").Loss;

pub fn simple_mse_loss(T: type, y_pred: NDTensor(T), y: NDTensor(T), allocator: std.mem.Allocator) !NDTensor(T) {
    var diff = (try y_pred.sub(y, allocator)).setLabel("diff");
    const diff2 = try NDTensor(T).init(diff.data.data, diff.data.shape.shape, true, allocator);
    diff2.label = "diff2";
    // const diff2 = (try y_pred.sub(y, allocator)).setLabel("diff2");
    const sq_diff = (try diff.mul(diff2, allocator)).setLabel("sq_diff");
    const sum_sq_diff = (try sq_diff.sum(allocator)).setLabel("sum_sq_diff");
    const coef = @as(T, @floatFromInt(y.data.data.len));
    const coef_tensor = try NDTensor(T).init(&[_]T{coef}, null, true, allocator);
    return (try sum_sq_diff.div(coef_tensor.setLabel("coef"), allocator)).setLabel("mse");
}

pub fn mse_loss(T: type, y_pred: NDTensor(T), y: NDTensor(T), allocator: std.mem.Allocator) !NDTensor(T) {
    const n = @as(T, @floatFromInt(y.data.data.len));
    var sum_sq_diff: T = 0;
    for (y_pred.data.data, y.data.data) |pred, target| {
        const diff = pred - target;
        sum_sq_diff += diff * diff;
    }
    const mse = sum_sq_diff / n;

    const result = try NDTensor(T).init(&[_]T{mse}, &[_]usize{1}, true, allocator);
    result.label = "mse";

    try result.setChildren(@constCast(&[_]NDTensor(T){ y_pred, y }));

    result._backward = struct {
        fn backward(tensor: NDTensor(T), _allocator: std.mem.Allocator) !void {
            _ = _allocator;
            const self_children = tensor.children orelse return error.NoChildren;
            const _y_pred = self_children[0];
            const _y = self_children[1];

            const _n = @as(T, @floatFromInt(_y.data.data.len));
            const scale = @as(T, 2) / _n;

            if (_y_pred.grad) |grad| {
                for (grad.data, _y_pred.data.data, _y.data.data) |*grad_val, pred_val, target_val| {
                    grad_val.* += scale * (pred_val - target_val);
                }
            }
        }
    }.backward;

    return result;
}

pub fn cross_entropy_loss(T: type, y_pred: NDTensor(T), y: NDTensor(T), allocator: std.mem.Allocator) !NDTensor(T) {
    const n = @as(T, @floatFromInt(y.data.data.len));
    var sum_loss: T = 0;
    const epsilon: T = 1e-7; // safe log

    for (y_pred.data.data, y.data.data) |pred, target| {
        const safe_pred = @min(@max(pred, epsilon), 1 - epsilon);
        sum_loss -= target * @log(safe_pred) + (1 - target) * @log(1 - safe_pred);
    }

    const mean_loss = sum_loss / n;
    var result = try NDTensor(T).init(&[_]T{mean_loss}, &[_]usize{1}, true, allocator);
    result.label = "cross_entropy";
    try result.setChildren(@constCast(&[_]NDTensor(T){ y_pred, y }));

    result._backward = struct {
        fn backward(tensor: NDTensor(T), _allocator: std.mem.Allocator) !void {
            _ = _allocator;
            const self_children = tensor.children orelse return error.NoChildren;
            const _y_pred = self_children[0];
            const _y = self_children[1];
            const _n = @as(T, @floatFromInt(_y.data.data.len));
            const scale = @as(T, 1) / _n;

            if (_y_pred.grad) |grad| {
                for (grad.data, _y_pred.data.data, _y.data.data) |*grad_val, pred_val, target_val| {
                    const safe_pred = @min(@max(pred_val, epsilon), 1 - epsilon);
                    const grad_contrib = (safe_pred - target_val) * scale;
                    grad_val.* += grad_contrib;
                }
            }
        }
    }.backward;

    return result;
}

pub fn simple_softmax(T: type, input: NDTensor(T), allocator: std.mem.Allocator) !NDTensor(T) {
    const max_val = try input.max(allocator);
    const exp_input = try (try input.sub(max_val, allocator)).exp(allocator);
    const sum = try exp_input.sum(allocator);
    return try exp_input.div(sum, allocator);
}

pub fn softmax(T: type, input: NDTensor(T), dim: usize, allocator: std.mem.Allocator) !NDTensor(T) {
    const dimsize = input.data.shape.get(dim);
    var result = try input.clone(allocator);
    errdefer result.deinit(allocator);

    // There are a few ways to do this. Could SIMD sum outside the loop with an NDArray method, but accum seems like a solid idea rn.
    for (0..dimsize) |i| { // mutate a view into result
        // slice i in the tgt dim
        const curr_slice = try result.data.sliceUnsafeNoAlloc(dim, i, i + 1);
        const max_val = std.mem.max(T, curr_slice);

        // exp(x-max)
        var curr_sum: T = 0;
        for (curr_slice) |*val| {
            val.* = @exp(val.* - max_val);
            curr_sum += val;
        }
        // scale
        try result.data._scale(1 / curr_sum);
    }

    // TODO: Softmax Backward
    // if (input.requires_grad) {
    //     const _backward = struct {
    //         fn backward(tensor: *NDTensor(T), grad: ?*NDArray(T), _: std.mem.Allocator) !void {
    //             var ctx: []zarray.Range = @ptrCast(@alignCast(tensor._backward_ctx.?));
    //             const self_children = tensor.children orelse return error.NoChildren;
    //             const fwd_input = self_children[0];
    //             if (fwd_input.grad) |input_grad| {
    //                 for (0..batch_size) |i| {
    //                     ctx[0].start = i;
    //                     ctx[0].end = i + 1;
    //
    //                     const softmax_output = try tensor.data.sliceRanges(ctx);
    //                     defer softmax_output.shape.deinit();
    //                     const incoming_grad = try grad.?.sliceRanges(ctx);
    //                     defer incoming_grad.shape.deinit();
    //
    //                     // jvp
    //                     const sum_grad = try softmax_output.dot(incoming_grad, allocator);
    //
    //                     var grad_slice = try softmax_output.mul(incoming_grad, allocator);
    //                     defer grad_slice.deinit(allocator);
    //                     try grad_slice._add_scalar(-sum_grad.data[0]);
    //
    //                     try input_grad.setSliceRanges(ctx, grad_slice);
    //                 }
    //             }
    //         }
    //     }.backward;
    //
    //     try result.setBackward(_backward, ranges);
    //     try result.setChildren(&[_]*const NDTensor(T){input});
    // }

    return result;
}

test "mse_loss" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f32;
    const Tensor = NDTensor(T);

    const y = try Tensor.init(&[_]T{1}, null, true, alloc);
    const yh = try Tensor.init(&[_]T{1}, null, true, alloc);
    var loss = try simple_mse_loss(T, yh, y, alloc);
    try std.testing.expectEqualSlices(T, &[_]T{0}, loss.data.data);
    // (1-2)**2 == 1
    try yh.set(&[_]usize{0}, 2);
    loss = try simple_mse_loss(T, yh, y, alloc);
    try std.testing.expectEqualSlices(T, &[_]T{1}, loss.data.data);

    // (1-3)**2 == 4
    try yh.set(&[_]usize{0}, 3);
    loss = try simple_mse_loss(T, yh, y, alloc);
    try std.testing.expectEqualSlices(T, &[_]T{4}, loss.data.data);

    // (-1-10)**2 == 121
    try y.set(&[_]usize{0}, -1);
    try yh.set(&[_]usize{0}, 10);
    loss = try simple_mse_loss(T, yh, y, alloc);
    try std.testing.expectEqualSlices(T, &[_]T{121}, loss.data.data);

    // Arrays
    const y_arr = try Tensor.init(&[_]T{ 1, 2, 3 }, &[_]usize{3}, true, alloc);
    const yh_arr = try Tensor.init(&[_]T{ 2, 3, 4 }, &[_]usize{3}, true, alloc);
    // FIXME: dim handling not entirely correct if you try adding dims numbers might be wrong, not sure what expected behavior should be
    // const yh_arr = try Tensor.init(&[_]T{ 2, 3, 4 }, &[_]usize{3, 1}, true, alloc);

    // ((1-2)^2 + (2-3)^2 + (3-4)^2) / 3 == (1 + 1 + 1) / 3 == 1
    loss = try simple_mse_loss(T, yh_arr, y_arr, alloc);
    try std.testing.expectEqualSlices(T, &[_]T{1}, loss.data.data);

    // MSE backward
    var gm = Loss(Tensor).init(alloc, .{});
    defer gm.deinit();
    loss.grad.?.fill(1.0);
    loss.acquire();
    try gm.backward(loss, alloc);
    loss.print();
}

fn norm(comptime T: type, comptime n: u32, a: @Vector(n, T)) T {
    return @reduce(.Add, a * a);
}

fn expectApproxSlices(comptime T: type, comptime n: u32, a: []const T, b: []const T) !void {
    const VT = @Vector(n, T);
    const na = norm(T, n, @as(VT, a[0..n].*));
    const va = @as(VT, a[0..n].*) / @as(VT, @splat(na));

    const nb = norm(T, n, @as(VT, b[0..n].*));
    const vb = @as(VT, b[0..n].*) / @as(VT, @splat(nb));
    const result = norm(T, n, va - vb);

    if (result > 0.01) {
        std.debug.print("norm was {d}\n", .{result});
        try std.testing.expectEqualSlices(T, a, b);
    }
}

test "softmax, cross-entropy, 2d" {
    // const allocator = std.testing.allocator;
    const allocator = std.heap.page_allocator;
    const T = f32;

    // Test case 1: 2D input

    const input_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 1.0, 2.0 };
    const input = try NDTensor(T).init(&input_data, &[_]usize{ 2, 3 }, true, allocator);

    const softmax_output = try simple_softmax(T, input, allocator);

    const expected_softmax = [_]T{ 0.09003057, 0.24472848, 0.66524094, 0.8437947, 0.04201007, 0.11419519 };
    const expected_softmax_shape = [_]usize{ 2, 3 };

    // compare softmax outputs
    // for (softmax_output.data.data, expected_softmax) |actual, expected| {
    //     try std.testing.expectApproxEqAbs(expected, actual, 1e-4);
    // }
    try expectApproxSlices(T, 6, &expected_softmax, softmax_output.data.data);
    try std.testing.expectEqualSlices(usize, &expected_softmax_shape, softmax_output.data.shape.shape);
    // cross-entropy
    const target_data = [_]T{ 1.0, 0.0, 0.0, 0.0, 1.0, 0.0 };
    const target = try NDTensor(T).init(&target_data, &[_]usize{ 2, 3 }, true, allocator);

    const loss = try cross_entropy_loss(T, softmax_output, target, allocator);
    // defer loss.teardown();

    const expected_loss: T = 1.4883861541748047;

    try std.testing.expectApproxEqAbs(loss.data.data[0], expected_loss, 1e-4);

    try loss.backward(allocator);

    const expected_grads = [_]T{ -0.18634209, 0.00052114, 0.18582097, 0.26313502, -0.1983895, -0.06474546 };
    // @round(@Vector(input_data.len, T), )
    try expectApproxSlices(T, 6, &expected_grads, input.grad.?.data);
}

test "softmax, cross-entropy, 1d" {
    // const allocator = std.testing.allocator;
    const allocator = std.heap.page_allocator;
    const T = f32;
    const input_data = [_]T{ -1.0, 0.0, 1.0 };
    const input = try NDTensor(T).init(&input_data, &[_]usize{3}, true, allocator);

    const softmax_output = try simple_softmax(T, input, allocator);

    const expected_softmax = [_]T{ 0.09003057, 0.24472848, 0.66524094 };

    try expectApproxSlices(T, 3, &expected_softmax, softmax_output.data.data);

    // cross-entropy
    const target_data = [_]T{ 0.0, 1.0, 0.0 };
    const target = try NDTensor(T).init(&target_data, &[_]usize{3}, true, allocator);

    const loss = try cross_entropy_loss(T, softmax_output, target, allocator);
    // defer loss.teardown();

    const expected_loss: T = 0.8654314875602722;

    try std.testing.expectApproxEqAbs(loss.data.data[0], expected_loss, 1e-4);

    try loss.backward(allocator);

    const expected_grads = [_]T{ 3.8343130e-04, -4.2193821e-01, 4.2155474e-01 };

    try expectApproxSlices(T, 3, &expected_grads, input.grad.?.data);
}
