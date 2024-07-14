// TODO: fused softmax-ce
const std = @import("std");
const zg = @import("../root.zig");

const Shape = zg.Shape;
const NDArray = zg.NDArray;
const settings = zg.settings;
const NDTensor = zg.NDTensor;
const GraphManager = zg.GraphManager;

pub fn simple_mse_loss(T: type, y_pred: *const NDTensor(T), y: *const NDTensor(T), allocator: std.mem.Allocator) !*NDTensor(T) {
    var diff = (try y_pred.sub(y, allocator)).setLabel("diff");
    const diff2 = (try NDTensor(T).init(diff.data.data, diff.data.shape.shape, true, allocator)).setLabel("diff2");
    // const diff2 = (try y_pred.sub(y, allocator)).setLabel("diff2");
    const sq_diff = (try diff.mul(diff2, allocator)).setLabel("sq_diff");
    const sum_sq_diff = (try sq_diff.sum(allocator)).setLabel("sum_sq_diff");
    const coef = @as(T, @floatFromInt(y.data.data.len));
    const coef_tensor = try NDTensor(T).init(&[_]T{coef}, null, true, allocator);
    return (try sum_sq_diff.div(coef_tensor.setLabel("coef"), allocator)).setLabel("mse");
}

pub fn mse_loss(T: type, y_pred: *const NDTensor(T), y: *const NDTensor(T), allocator: std.mem.Allocator) !*NDTensor(T) {
    const n = @as(T, @floatFromInt(y.data.data.len));
    var sum_sq_diff: T = 0;
    for (y_pred.data.data, y.data.data) |pred, target| {
        const diff = pred - target;
        sum_sq_diff += diff * diff;
    }
    const mse = sum_sq_diff / n;

    const bw_fn = struct {
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

    return try NDTensor(T).createDependent(.{
        .data = try NDArray(T).init(&[_]T{mse}, &[_]usize{1}, allocator),
        .children = &[_]*const NDTensor(T){ y_pred, y },
        .label = "mse",
        .requires_grad = true,
        .allocator = allocator,
        ._backward = bw_fn,
    });
}

pub fn cross_entropy_loss(T: type, y_pred: *const NDTensor(T), y: *const NDTensor(T), allocator: std.mem.Allocator) !*NDTensor(T) {
    std.log.info("{d} {d}", .{ y_pred.data.shape.shape, y.data.shape.shape });
    const n = @as(T, @floatFromInt(y.data.data.len));
    var sum_loss: T = 0;
    const epsilon: T = 1e-7; // safe log

    for (y_pred.data.data, y.data.data) |pred, target| {
        const safe_pred = @min(@max(pred, epsilon), 1 - epsilon);
        sum_loss -= target * @log(safe_pred) + (1 - target) * @log(1 - safe_pred);
    }

    const mean_loss = sum_loss / n;

    const bw_fn = struct {
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
    return try NDTensor(T).createDependent(.{
        .data = try NDArray(T).init(&[_]T{mean_loss}, &[_]usize{1}, allocator),
        .op = null,
        .children = &[_]*const NDTensor(T){ y_pred, y },
        .label = "cross_entropy",
        .requires_grad = true,
        .allocator = allocator,
        ._backward = bw_fn,
    });
}

pub fn ag_softmax_1d(T: type, input: *const NDTensor(T), allocator: std.mem.Allocator) !*NDTensor(T) {
    const max_val = try input.max(allocator);
    const exp_input = try (try input.sub(max_val, allocator)).exp(allocator);
    const sum = try exp_input.sum(allocator);
    return try exp_input.div(sum, allocator);
}

pub fn softmax(T: type, input: *const NDTensor(T), dim: usize, allocator: std.mem.Allocator) !*NDTensor(T) {
    const dimsize = try input.data.shape.get(dim);
    var result = try input.clone(allocator);
    errdefer result.deinit();

    // There are a few ways to do this. Could SIMD sum outside the loop with an NDArray method, but accum seems like a solid idea rn.
    for (0..dimsize) |i| { // mutate a view into result by directly operating on the backing ndarray
        // slice i in the tgt dim
        std.debug.print("i:{d}\n", .{i});
        const curr_slice = try result.data.sliceUnsafeNoAlloc(dim, i, i + 1);
        const max_val = std.mem.max(T, curr_slice.data);

        // exp(x-max)
        var curr_sum: T = 0;
        for (curr_slice.data) |*val| {
            val.* = @exp(val.* - max_val);
            curr_sum += val.*;
        }
        // scale
        curr_slice._scale(1 / curr_sum);
        std.debug.print("\ncurr_slice:{d}\n", .{curr_slice.data});
    }

    // TODO: once confirmed can use more ndarray elementwise ops
    const bw_fn = struct {
        fn backward(tensor: NDTensor(T), _allocator: std.mem.Allocator) !void {
            const self_children = tensor.children orelse return error.NoChildren;
            const _input = self_children[0];
            if (_input.grad == null) return;
            const bw_ctx: *usize = @ptrCast(@alignCast(tensor._backward_ctx orelse return error.NoBackwardContext));
            const _dim: usize = bw_ctx.*;
            defer _allocator.destroy(bw_ctx);

            const _dimsize = try tensor.data.shape.get(_dim);
            for (0.._dimsize) |i| {
                const s = try tensor.data.sliceUnsafeNoAlloc(_dim, i, i + 1);
                const dL_ds = try _input.grad.?.sliceUnsafeNoAlloc(_dim, i, i + 1);

                var sum_diff: T = 0;
                for (s.data, dL_ds.data) |s_i, dL_ds_i| {
                    sum_diff += s_i * dL_ds_i;
                }

                for (s.data, dL_ds.data) |s_i, *dL_dx_i| {
                    dL_dx_i.* = s_i * (dL_dx_i.* - sum_diff);
                }
            }
        }
    }.backward;
    const ctx = try allocator.create(usize);
    ctx.* = dim;

    return try NDTensor(T).createDependent(.{
        .data = result.data,
        .children = &[_]*const NDTensor(T){input},
        .label = "softmax",
        .requires_grad = input.requires_grad,
        .allocator = allocator,
        ._backward = bw_fn,
        ._backward_ctx = ctx,
    });
}
