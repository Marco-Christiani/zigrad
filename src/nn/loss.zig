// TODO: fused softmax-ce
const std = @import("std");
const zg = @import("../zigrad.zig");
const DeviceReference = zg.DeviceReference;
const ReduceType = zg.ReduceType;

const Shape = zg.Shape;
const NDArray = zg.NDArray;
const settings = zg.settings;
const NDTensor = zg.NDTensor;
const GraphManager = zg.GraphManager;

// NOTE: Reductions return NaN when "reduce" is set to false.
// This is for c-compatibility and this value is instead traded for null.

// nll entry point
pub fn nll(comptime T: type, comptime config: NLLConfig) NLLType(T, config) {
    return .{};
}

pub const NLLConfig = struct {
    // dimensions of input tensor
    dimensions: usize,
    // nll expecting logits - if true,
    // softmax will be used on input
    input_logits: bool,
    // specifies that the target type
    // will be provided as an index
    target_type: enum { indices, encoding },
    // specifies the reduce type used
    reduce_type: ReduceType,
};

// negative log likelihood
pub fn NLLType(comptime T: type, comptime config: NLLConfig) type {
    return if (config.target_type == .indices) NLLIndex(T, config) else @compileError("Unimplemented");
}

fn NLLEncode(comptime T: type, comptime config: NLLConfig) type {
    return switch (config.dimensions) {
        1 => struct {
            pub fn score(src: *NDTensor(T), trg: *const NDTensor(T), reduce: bool) !*NDTensor {
                const out = try NDTensor(T).empty(&.{1}, src.device);
                src.device.nn.nll_loss_1d_encode_forward(
                    T,
                    src.get_data(),
                    trg.get_data(),
                    out.get_data(),
                    config.input_logits,
                    reduce,
                    config.reduce_type,
                );
                return out;
            }

            //fn backward(src: *zg.NDTensor(T)) anyerror!void {
            //    const trg: usize = @intFromPtr(src._backward_ctx.?) -| 1;
            //    src.device.nn.nll_loss_1d_index_backward(T, src.get_data(), src.grad.?.data, trg, config.reduce_type);
            //}
        },
        else => @compileError("Unsupported Dimensions for NLLIndex"),
    };
}

fn NLLIndex(comptime T: type, comptime config: NLLConfig) type {
    return switch (config.dimensions) {
        1 => struct {
            pub fn score(src: *NDTensor(T), trg_index: usize, reduce: bool) !*NDTensor {
                const out = try NDTensor(T).empty(&.{1}, src.device);
                src.device.nn.nll_loss_1d_index_forward(
                    T,
                    src.get_data(),
                    trg_index,
                    out.get_data(),
                    config.input_logits,
                    reduce,
                    config.reduce_type,
                );
            }

            //fn backward(src: *zg.NDTensor(T)) anyerror!void {
            //    const trg: usize = @intFromPtr(src._backward_ctx.?) -| 1;
            //    src.device.nn.nll_loss_1d_index_backward(T, src.get_data(), src.grad.?.data, trg, config.reduce_type);
            //}
        },
        else => @compileError("Unsupported Dimensions for NLLIndex"),
    };
}

/// Relies on autograd, operates on the flat data.
pub fn ag_mse_1d(T: type, y_pred: *NDTensor(T), y: *NDTensor(T), device: DeviceReference) !*NDTensor(T) {
    var diff = try y_pred.sub(y);
    try diff.set_label("diff");

    const diff2 = try NDTensor(T).init(diff.data.data, diff.data.shape.slice(), true, device);
    try diff2.set_label("diff2");
    // const diff2 = (try y_pred.sub(y, allocator)).set_label("diff2");
    const sq_diff = try diff.mul(diff2);
    try sq_diff.set_label("sq_diff");

    const sum_sq_diff = try sq_diff.sum();
    try sum_sq_diff.set_label("sum_sq_diff");

    const coef = @as(T, @floatFromInt(y.get_size()));
    const coef_tensor = try NDTensor(T).init(&.{coef}, null, true, device);
    try coef_tensor.set_label("coef");

    const out = try sum_sq_diff.div(coef_tensor);
    try out.set_label("mse");

    return out;
}

pub fn mse_loss(T: type, y_pred: *NDTensor(T), y: *NDTensor(T), device: DeviceReference) !*NDTensor(T) {
    const n = @as(T, @floatFromInt(y.get_size()));
    var sum_sq_diff: T = 0;
    for (y_pred.data.data, y.data.data) |pred, target| {
        const diff = pred - target;
        sum_sq_diff += diff * diff;
    }
    const mse = sum_sq_diff / n;

    const bw_fn = struct {
        fn backward(tensor: NDTensor(T), _: DeviceReference) !void {
            const self_children = tensor.get_children() orelse return error.NoChildren;
            const _y_pred = self_children[0];
            const _y = self_children[1];

            const _n = @as(T, @floatFromInt(_y.get_size()));
            const scale = @as(T, 2) / _n;

            if (_y_pred.grad) |grad| {
                for (grad.data, _y_pred.data.data, _y.data.data) |*grad_val, pred_val, target_val| {
                    grad_val.* += scale * (pred_val - target_val);
                }
            }
        }
    }.backward;

    return try NDTensor(T).create_dependent(.{
        .data = try NDArray(T).init(&[_]T{mse}, &.{1}, device),
        .children = &.{ y_pred, y },
        .label = "mse",
        .requires_gradient = true,
        .device = device,
        ._backward = bw_fn,
    });
}

/// Runs over last dim.
pub fn softmax_cross_entropy_loss(T: type, y_pred: *NDTensor(T), y: *NDTensor(T)) !*NDTensor(T) {
    var sum_loss: T = 0;
    const epsilon: T = 1e-7;
    if (y_pred.data.shape.len > 2) return error.NotSupported;
    const batch_size = if (y_pred.data.shape.len > 1) y_pred.data.shape.get(0) else 1;
    const last_dim = if (y_pred.data.shape.len > 1) y_pred.data.shape.len - 1 else 0;
    const sm_preds = try _softmax_fwd(T, y_pred, last_dim);

    for (sm_preds.data.data, 0..) |pred, i| {
        const target = y.data.data[i];
        const safe_pred = @min(@max(pred, epsilon), 1.0 - epsilon);
        sum_loss -= target * @log(safe_pred);
    }
    const mean_loss = sum_loss / @as(T, @floatFromInt(batch_size));

    const bw_fn = struct {
        fn backward(bw_tensor: NDTensor(T)) !void {
            const bw_ctx: *NDTensor(T) = @ptrCast(@alignCast(bw_tensor._backward_ctx orelse return error.NoBackwardContext));
            defer bw_ctx.deinit();
            const bw_self_children = bw_tensor.get_children() orelse return error.NoChildren;
            const bw_y_pred = bw_self_children[0];
            const bw_y = bw_self_children[1];
            const bw_batch_size = if (bw_y_pred.data.shape.len > 1) bw_y_pred.data.shape.get(0) else 1;
            // const bw_n = @as(T, @floatFromInt(bw_y.get_size()));
            if (bw_y_pred.grad) |bw_grad| {
                for (bw_grad.data, bw_ctx.data.data, bw_y.data.data) |*bw_grad_val, bw_sm_val, bw_target_val| {
                    bw_grad_val.* += (bw_sm_val - bw_target_val) / @as(T, @floatFromInt(bw_batch_size));
                }
            }
        }
    }.backward;

    return try NDTensor(T).create_dependent(.{
        .data = try NDArray(T).init(&.{mean_loss}, &.{1}, y_pred.device),
        .op = null,
        .children = &.{ y_pred, y },
        .label = "cross_entropy",
        .device = y_pred.device,
        ._backward = bw_fn,
        ._backward_ctx = sm_preds, // no need to copy
        ._requires_grad = true,
    });
}

/// Relies on autograd, operates on the flat data.
pub fn ag_softmax_1d(T: type, input: *NDTensor(T)) !*NDTensor(T) {
    const max_val = try input.max();
    const exp_input = try (try input.sub(max_val)).exp();
    const sum = try exp_input.sum();
    return try exp_input.div(sum);
}

// There are a few ways to do this. Could SIMD sum outside the loop with an NDArray method, but accum seems like a solid idea rn.
// mutate a view into result by directly operating on the backing ndarray
fn _softmax_fwd(T: type, input: *NDTensor(T), dim: usize) !*NDTensor(T) {
    const shape = input.data.shape.slice();
    if (dim >= shape.len) return error.InvalidDimension;

    const dim_size = shape[dim];
    const total_size = input.get_size();
    const outer_size = @divExact(total_size, dim_size);

    var result = try input.clone();
    errdefer result.deinit();

    // TODO: use shape methods, or prod.
    var strides = try input.device.allocator.alloc(usize, shape.len);
    defer input.device.allocator.free(strides);
    var stride: usize = 1;
    var i: usize = shape.len;
    while (i > 0) {
        i -= 1;
        strides[i] = stride;
        stride *= shape[i];
    }

    // calc softmax
    var outer_idx: usize = 0;
    while (outer_idx < outer_size) : (outer_idx += 1) {
        const base_idx = (outer_idx / strides[dim]) * (strides[dim] * dim_size) + (outer_idx % strides[dim]);

        //  max over slice
        var max_val = result.data.data[base_idx];
        for (1..dim_size) |j| {
            const idx = base_idx + j * strides[dim];
            max_val = @max(max_val, result.data.data[idx]);
        }

        // log-sum-exp
        var sum_exp: T = 0;
        for (0..dim_size) |j| {
            const idx = base_idx + j * strides[dim];
            sum_exp += @exp(result.data.data[idx] - max_val);
        }
        const log_sum_exp = max_val + @log(sum_exp);

        // normalize
        for (0..dim_size) |j| {
            const idx = base_idx + j * strides[dim];
            result.data.data[idx] = @exp(result.data.data[idx] - log_sum_exp);
        }
    }
    return result;
}

pub fn softmax(T: type, input: *const NDTensor(T), dim: usize, device: DeviceReference) !*NDTensor(T) {
    const result = try _softmax_fwd(T, input, dim, device);
    const bw_fn = struct {
        fn backward(bw_tensor: NDTensor(T), bw_device: DeviceReference) !void {
            const bw_self_children = bw_tensor.children orelse return error.NoChildren;
            const bw_input = bw_self_children[0];
            if (bw_input.grad == null) return;
            const bw_ctx: *usize = @ptrCast(@alignCast(bw_tensor._backward_ctx orelse return error.NoBackwardContext));
            const bw_dim = bw_ctx.*;
            defer bw_device.allocator.destroy(bw_ctx);

            const bw_dim_size = bw_tensor.data.shape.get(bw_dim);
            const bw_total_size = bw_tensor.get_size();
            const bw_outer_size = @divExact(bw_total_size, bw_dim_size);

            var bw_outer_idx: usize = 0;
            while (bw_outer_idx < bw_outer_size) : (bw_outer_idx += 1) {
                const bw_base_idx = bw_outer_idx * bw_dim_size;
                var bw_sum_grad: T = 0;
                for (0..bw_dim_size) |bw_j| {
                    const bw_idx = bw_base_idx + bw_j;
                    bw_sum_grad += bw_tensor.data.data[bw_idx] * bw_tensor.grad.?.data[bw_idx];
                }
                for (0..bw_dim_size) |bw_j| {
                    const bw_idx = bw_base_idx + bw_j;
                    const bw_softmax_out = bw_tensor.data.data[bw_idx];
                    bw_input.grad.?.data[bw_idx] += bw_softmax_out * (bw_tensor.grad.?.data[bw_idx] - bw_sum_grad);
                }
            }
        }
    }.backward;

    const ctx = try device.allocator.create(usize);
    ctx.* = dim;

    return try NDTensor(T).create_dependent(.{
        .data = result.data,
        .children = &.{input},
        .label = "softmax",
        .device = device,
        ._backward = bw_fn,
        ._backward_ctx = ctx,
        ._requires_grad = true,
    });
}

pub fn smooth_l1_loss(comptime T: type, y_pred: *NDTensor(T), y: *NDTensor(T), beta: T, device: DeviceReference) !*NDTensor(T) {
    const n = @as(T, @floatFromInt(y.get_size()));
    var sum_loss: T = 0;

    for (y_pred.data.data, y.data.data) |pred, target| {
        const diff: T = pred - target;
        const abs_diff: T = @abs(diff);
        if (abs_diff < beta) {
            sum_loss += 0.5 * (diff * diff) / beta;
        } else {
            sum_loss += abs_diff - (0.5 * beta);
        }
    }
    const loss = sum_loss / n;

    const bw_fn = struct {
        fn backward(tensor: NDTensor(T)) !void {
            const self_children = tensor.get_children() orelse return error.NoChildren;
            const _y_pred = self_children[0];
            const _y = self_children[1];
            const _bw_ctx: *T = @ptrCast(@alignCast(tensor._backward_ctx orelse return error.NoBackwardContext));
            defer tensor.device.mem_destroy(_bw_ctx);
            const _beta = _bw_ctx.*;

            const _n = @as(T, @floatFromInt(_y.get_size()));

            if (_y_pred.grad) |grad| {
                for (grad.data, _y_pred.data.data, _y.data.data) |*grad_val, pred_val, target_val| {
                    const diff = pred_val - target_val;
                    if (@abs(diff) < _beta) {
                        grad_val.* += diff / (_beta * _n);
                    } else {
                        grad_val.* += std.math.sign(diff) / _n;
                    }
                }
            }
        }
    }.backward;

    const beta_ctx = try device.allocator.create(T);
    beta_ctx.* = beta;

    return try NDTensor(T).create_dependent(.{
        .data = try NDArray(T).init(&.{loss}, &.{1}, device),
        .children = &.{ y_pred, y },
        .label = "smooth_l1",
        .device = device,
        ._backward = bw_fn,
        ._backward_ctx = beta_ctx,
        ._requires_grad = true,
    });
}
// TODO: move this since refactor this file was renamed to loss.zig
// pub fn gather(comptime T: type, input: *const NDTensor(T), indices: *const NDTensor(usize), dim: usize, device: DeviceReference) !*NDTensor(T)

// Documented outline for adding new custom operations
// pub fn myop(T: type, input: *const NDTensor(T), dim: usize, device: DeviceReference) !*NDTensor(T) {
//     // implement forward... (could be a separate function)
//     const bw_fn = struct {
//         fn backward(bw_tensor: NDTensor(T), bw_device: DeviceReference) !void {
//             // An example of retreiving data that was saved for backward
//             // in this example, we saved the dim (we cannot cross scope boundary in closure, so this is necessary)
//             // We CAN reuse "T", though in the function signature since this is comptime known
//             // Prefixing variable names with bw_ to avoid collisions with variables in the parent function scope.
//             // We can access the operands for upstream gradients/data.
//             const bw_self_children = bw_tensor.children orelse return error.NoChildren;
//             const bw_input = bw_self_children[0];
//             _ = bw_input; // autofix
//             // We saved a *usize, so we retreive it and reify type information
//             const bw_ctx: *usize = @ptrCast(@alignCast(bw_tensor._backward_ctx orelse return error.NoBackwardContext));
//             const bw_dim = bw_ctx.*;
//             _ = bw_dim; // autofix
//             _ = bw_dim; // autofix
//             defer bw_allocator.destroy(bw_ctx);
//             // ...
//         }
//     }.backward;
//     // save for backward (will be stored internally as a type erased pointer)
//     const ctx = try allocator.create(usize);
//     ctx.* = dim;
//
//     return try NDTensor(T).create_dependent(.{
//         .data = result, // *NDArray
//         .children = &[_]*const NDTensor(T){input},
//         .label = "myop",
//         .requires_gradient = input.requires_gradient,
//         .allocator = allocator,
//         ._backward = bw_fn,
//         ._backward_ctx = ctx,
//     });
// }
