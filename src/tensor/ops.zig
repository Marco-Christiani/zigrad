const std = @import("std");
const NDTensor = @import("tensor.zig").NDTensor;
const NDArray = @import("zarray.zig").NDArray;
const Loss = @import("tensor.zig").Loss;

pub fn simple_mse_loss(T: type, y_pred: *NDTensor(T), y: *NDTensor(T), allocator: std.mem.Allocator) !*NDTensor(T) {
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

pub fn mse_loss(T: type, y_pred: *NDTensor(T), y: *NDTensor(T), allocator: std.mem.Allocator) !*NDTensor(T) {
    const n = @as(T, @floatFromInt(y.data.data.len));
    var sum_sq_diff: T = 0;
    for (y_pred.data.data, y.data.data) |pred, target| {
        const diff = pred - target;
        sum_sq_diff += diff * diff;
    }
    const mse = sum_sq_diff / n;

    const result = try NDTensor(T).init(&[_]T{mse}, &[_]usize{1}, true, allocator);
    result.label = "mse";

    try result.setChildren(@constCast(&[_]*const NDTensor(T){ y_pred, y }));

    result._backward = struct {
        fn backward(tensor: *NDTensor(T), _allocator: std.mem.Allocator) !void {
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

pub fn cross_entropy_loss(T: type, y_pred: *NDTensor(T), y: *NDTensor(T), allocator: std.mem.Allocator) !*NDTensor(T) {
    const n = @as(T, @floatFromInt(y.data.data.len));
    var sum_loss: T = 0;
    const epsilon: T = 1e-7; // safe log

    for (y_pred.data.data, y.data.data) |pred, target| {
        const safe_pred = @min(@max(pred, epsilon), 1 - epsilon);
        sum_loss -= target * @log(safe_pred) + (1 - target) * @log(1 - safe_pred);
    }

    const mean_loss = sum_loss / n;
    const result = try NDTensor(T).init(&[_]T{mean_loss}, &[_]usize{1}, true, allocator);
    result.label = "cross_entropy";
    try result.setChildren(@constCast(&[_]*const NDTensor(T){ y_pred, y }));

    result._backward = struct {
        fn backward(tensor: *NDTensor(T), _allocator: std.mem.Allocator) !void {
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

pub fn softmax(T: type, input: *NDTensor(T), allocator: std.mem.Allocator) !*NDTensor(T) {
    const max_val = try input.max(allocator);
    const exp_input = try (try input.sub(max_val, allocator)).exp(allocator);
    const sum = try exp_input.sum(allocator);
    return try exp_input.div(sum, allocator);
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
    var gm = Loss(Tensor).init(arena.allocator());
    defer gm.deinit();
    loss.grad.?.fill(1.0);
    loss.acquire();
    try gm.backward(loss, alloc);
    loss.print();
}
