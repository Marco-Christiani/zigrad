const std = @import("std");
const opspec = @import("../opspec.zig");
const HostDevice = @import("../host_device.zig");

// There are a few ways to do this. Could SIMD sum outside the loop with an NDArray method, but accum seems like a solid idea rn.
// mutate a view into result by directly operating on the backing ndarray
fn smax_fwd(_: *const HostDevice, T: type, p: opspec.smax_fwd(T)) void {
    std.debug.assert(p.x.len == p.y.len);

    const shape = p.x_shape;
    const dim_size = shape[p.dim];
    const total_size = p.input.len;
    const outer_size = @divExact(total_size, dim_size);
    const stride = p.x_strides[p.dim];

    // calc softmax
    var outer_idx: usize = 0;
    while (outer_idx < outer_size) : (outer_idx += 1) {
        const base_idx = (outer_idx / stride) * (stride * dim_size) + (outer_idx % stride);

        //  max over slice
        var max_val = p.x[base_idx];
        for (1..dim_size) |j| {
            const idx = base_idx + j * stride;
            max_val = @max(max_val, p.x[idx]);
        }

        // log-sum-exp
        var sum_exp: T = 0;
        for (0..dim_size) |j| {
            const idx = base_idx + j * stride;
            sum_exp += @exp(p.x[idx] - max_val);
        }
        const log_sum_exp = max_val + @log(sum_exp);

        // normalize
        for (0..dim_size) |j| {
            const idx = base_idx + j * stride;
            p.y[idx] = @exp(p.x[idx] - log_sum_exp);
        }
    }
}

pub fn smax_bwd(_: *const HostDevice, T: type, p: opspec.smax_bwd(T)) !void {
    std.debug.assert(p.x_g.len == p.y_g.len);
    std.debug.assert(p.x_g.len == p.y.len);

    const bw_dim_size = p.x_shape[p.dim];
    const bw_total_size = p.x.len;
    const bw_outer_size = @divExact(bw_total_size, bw_dim_size);

    var bw_outer_idx: usize = 0;
    while (bw_outer_idx < bw_outer_size) : (bw_outer_idx += 1) {
        const bw_base_idx = bw_outer_idx * bw_dim_size;
        var bw_sum_grad: T = 0;
        for (0..bw_dim_size) |bw_j| {
            const bw_idx = bw_base_idx + bw_j;
            bw_sum_grad += p.y[bw_idx] * p.y_g[bw_idx];
        }
        for (0..bw_dim_size) |bw_j| {
            const bw_idx = bw_base_idx + bw_j;
            const bw_softmax_out = p.y[bw_idx];
            p.x_g[bw_idx] += bw_softmax_out * (p.y_g[bw_idx] - bw_sum_grad);
        }
    }
}

pub fn relu_fwd(_: *const HostDevice, T: type, p: opspec.relu_fwd(T)) void {
    for (p.x, p.y) |x, *y| y.* = @max(0, x);
}

pub fn relu_bwd(_: *const HostDevice, T: type, p: opspec.relu_bwd(T)) void {
    for (p.x, p.x_g, p.y_g) |x, *x_g, y_g| x_g.* += if (x > 0) y_g else 0;
}

pub fn relu_inplace_bwd(_: *const HostDevice, T: type, p: opspec.relu_inplace_bwd(T)) void {
    for (p.x, p.x_g) |x, *x_g| {
        if (x <= 0) x_g.* = 0;
    }
}

pub fn tanh_fwd(_: *const HostDevice, T: type, p: opspec.tanh_fwd(T)) void {
    for (p.x, p.y) |x, *y| y.* = std.math.tanh(x);
}

pub fn tanh_bwd(_: *const HostDevice, T: type, p: opspec.tanh_bwd(T)) void {
    for (p.x_g, p.y, p.y_g) |*x_g, y, y_g| x_g.* += (1 - (y * y)) * y_g;
}

pub fn tanh_inplace_bwd(_: *const HostDevice, T: type, p: opspec.tanh_inplace_bwd(T)) void {
    for (p.x, p.x_g) |x, *x_g| x_g.* *= (1 - (x * x));
}

pub fn sigm_fwd(_: *const HostDevice, T: type, p: opspec.sigm_fwd(T)) void {
    for (p.x, p.y) |x, *y| y.* = 1 / (1 + @exp(-x));
}

pub fn sigm_bwd(_: *const HostDevice, T: type, p: opspec.sigm_bwd(T)) void {
    for (p.x_g, p.y, p.y_g) |*x_g, y, y_g| x_g.* += y * (1 - y) * y_g;
}

pub fn sigm_inplace_bwd(_: *const HostDevice, T: type, p: opspec.sigm_inplace_bwd(T)) void {
    for (p.x, p.x_g) |x, *x_g| x_g.* *= (x * (1 - x));
}

/// MSE forward: loss = sum((pred - target)^2) / n
pub fn mse_fwd(_: *const HostDevice, T: type, p: opspec.mse_fwd(T)) void {
    var s: T = 0;
    for (p.pred, p.target) |pred_val, target_val| {
        const diff = pred_val - target_val;
        s += diff * diff;
    }
    p.loss[0] = s / @as(T, @floatFromInt(p.n));
}

/// MSE backward: pred_grad += 2 * (pred - target) / n * loss_grad
pub fn mse_bwd(_: *const HostDevice, T: type, p: opspec.mse_bwd(T)) void {
    const _scale = 2.0 / @as(T, @floatFromInt(p.n)) * p.loss_grad[0];
    for (p.pred, p.target, p.pred_grad) |pred_val, target_val, *grad_val| {
        grad_val.* += _scale * (pred_val - target_val);
    }
}
