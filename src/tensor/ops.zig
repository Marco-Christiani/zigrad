const std = @import("std");
const NDTensor = @import("tensor.zig").NDTensor;
const Loss = @import("tensor.zig").Loss;

pub fn mse_loss(T: type, y_pred: *NDTensor(T), y: *NDTensor(T), allocator: std.mem.Allocator) !*NDTensor(T) {
    var diff = (try y_pred.sub(y, allocator)).setLabel("diff");
    const sq_diff = (try diff.mul(diff, allocator)).setLabel("sq_diff");
    const sum_sq_diff = (try sq_diff.sum(allocator)).setLabel("sum_sq_diff");
    const coef = @as(T, @floatFromInt(y.data.data.len));
    const coef_tensor = try NDTensor(T).init(&[_]T{coef}, null, true, allocator);
    return (try sum_sq_diff.div(coef_tensor.setLabel("coef"), allocator)).setLabel("mse");
}

test "mse_loss" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f32;
    const Tensor = NDTensor(T);

    const y = try Tensor.init(&[_]T{1}, null, true, alloc);
    const yh = try Tensor.init(&[_]T{1}, null, true, alloc);
    var loss = try mse_loss(T, yh, y, alloc);
    try std.testing.expectEqualSlices(T, &[_]T{0}, loss.data.data);
    // (1-2)**2 == 1
    try yh.set(&[_]usize{0}, 2);
    loss = try mse_loss(T, yh, y, alloc);
    try std.testing.expectEqualSlices(T, &[_]T{1}, loss.data.data);

    // (1-3)**2 == 4
    try yh.set(&[_]usize{0}, 3);
    loss = try mse_loss(T, yh, y, alloc);
    try std.testing.expectEqualSlices(T, &[_]T{4}, loss.data.data);

    // (-1-10)**2 == 121
    try y.set(&[_]usize{0}, -1);
    try yh.set(&[_]usize{0}, 10);
    loss = try mse_loss(T, yh, y, alloc);
    try std.testing.expectEqualSlices(T, &[_]T{121}, loss.data.data);

    // Arrays
    const y_arr = try Tensor.init(&[_]T{ 1, 2, 3 }, &[_]usize{3}, true, alloc);
    const yh_arr = try Tensor.init(&[_]T{ 2, 3, 4 }, &[_]usize{3}, true, alloc);
    // FIXME: dim handling not entirely correct if you try adding dims numbers might be wrong, not sure what expected behavior should be
    // const yh_arr = try Tensor.init(&[_]T{ 2, 3, 4 }, &[_]usize{3, 1}, true, alloc);

    // ((1-2)^2 + (2-3)^2 + (3-4)^2) / 3 == (1 + 1 + 1) / 3 == 1
    loss = try mse_loss(T, yh_arr, y_arr, alloc);
    try std.testing.expectEqualSlices(T, &[_]T{1}, loss.data.data);

    // MSE backward
    var gm = Loss(Tensor).init(arena.allocator());
    defer gm.deinit();
    loss.grad.?.fill(1.0);
    loss.acquire();
    try gm.backward(loss, alloc);
    loss.print();
}
