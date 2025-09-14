// TODO: these ops to be migrated into the new device API design
const std = @import("std");
const builtin = @import("builtin");
const debug: bool = (builtin.mode == .Debug);

const zg = @import("../zigrad.zig");
const DeviceReference = zg.DeviceReference;
const ReduceType = zg.ReduceType;
const Shape = zg.Shape;
const NDArray = zg.NDArray;
const settings = zg.settings;
const NDTensor = zg.NDTensor;
const Graph = zg.Graph;
const Node = Graph.Node;
const opspec = zg.opspec;

/// Direct Mean Squared Error loss.
pub fn mse_loss(T: type, y_pred: *NDTensor(T), y: *NDTensor(T)) !*NDTensor(T) {
    const Tensor = NDTensor(T);

    const MseBwd = struct {
        pub fn backward(_: *Tensor, children: *Node.Children) !void {
            const preds = children.get_bwd_upcast(Tensor, 0) orelse return;
            const label = children.get_upcast(Tensor, 1);
            preds.device.dispatch(opspec.accumulate_scaled_delta(T){
                .x = preds.get_data(),
                .y = label.get_data(),
                .coef = 2.0 / @as(T, @floatFromInt(preds.data.shape.last())),
                .z = try preds.ensure_grad_data(0),
            });
        }
    };

    var output = try Tensor.DataType.empty(&.{1}, y_pred.device);
    errdefer output.deinit(y_pred.device);

    y_pred.device.dispatch(opspec.mse_fwd(T){
        .pred = y_pred.get_data(),
        .target = y.get_data(),
        .loss = output.get_data(),
        .n = y_pred.data.shape.last(),
    });

    return Tensor.create_dependent(MseBwd, .{
        .data = output,
        .children = &.{ &y_pred.node, &y.node },
        .label = "mse",
        .device = y_pred.device,
        .gb = y_pred.node.gb,
        .callback = .{},
    });
}

/// Runs over last dim.
pub fn softmax_cross_entropy_loss(T: type, y_pred: *NDTensor(T), y: *NDTensor(T)) !*NDTensor(T) {
    std.debug.assert(y_pred.get_ndim() <= 2);
    std.debug.assert(y.device.is_compatible(y_pred.device));

    const SceBwd = struct {
        sm_preds: *NDTensor(T),
        pub fn backward(_: *NDTensor(T), children: *Node.Children, ctx: *@This()) !void {
            defer ctx.sm_preds.deinit();
            const preds = children.get_bwd_upcast(NDTensor(T), 0) orelse return;
            const label = children.get_upcast(NDTensor(T), 1);

            preds.device.dispatch(opspec.accumulate_scaled_delta(T){
                .x = ctx.sm_preds.get_data(),
                .y = label.get_data(),
                .coef = 1.0 / @as(T, @floatFromInt(preds.data.shape.last())),
                .z = try preds.ensure_grad_data(0),
            });
        }
    };

    const sm_preds = try NDTensor(T).empty(y.device, y_pred.get_shape(), .{
        .graph = y_pred.node.gb.promote(),
        .label = "softmax_preds",
    });
    errdefer sm_preds.deinit();

    var loss = try NDTensor(T).DataType.empty(&.{1}, y.device);
    errdefer loss.deinit(y.device);

    y.device.dispatch(opspec.softmax_fwd(T){
        .x = y_pred.get_data(),
        .x_shape = y_pred.get_shape(),
        .dim = y_pred.data.shape.last_index(),
        .y = sm_preds.get_data(),
    });

    y.device.dispatch(opspec.nll_fwd(T){
        .y_pred = sm_preds.get_data(),
        .y = y.get_data(),
        .batch_size = sm_preds.data.shape.last(),
        .loss = loss.get_data(),
    });

    if (!y_pred.requires_grad())
        sm_preds.deinit(); // we store this off the typical path

    return NDTensor(T).create_dependent(SceBwd, .{
        .data = loss,
        .children = &.{ &y_pred.node, &y.node },
        .label = "cross_entropy",
        .device = y_pred.device,
        .gb = y_pred.node.gb,
        .callback = .{ .sm_preds = sm_preds },
    });
}

pub fn smooth_l1_loss(T: type, y_pred: *NDTensor(T), y: *NDTensor(T), beta: T) !*NDTensor(T) {
    const Tensor = NDTensor(T);

    std.debug.assert(y_pred.device.is_compatible(y.device));
    const n = @as(T, @floatFromInt(y.get_size()));
    var sum_loss: T = 0;

    for (y_pred.get_data(), y.get_data()) |pred, target| {
        const diff: T = pred - target;
        const abs_diff: T = @abs(diff);
        if (abs_diff < beta) {
            sum_loss += 0.5 * (diff * diff) / beta;
        } else {
            sum_loss += abs_diff - (0.5 * beta);
        }
    }
    const loss = sum_loss / n;

    const Sl1LossBwd = struct {
        beta: T,
        pub fn backward(_: *Tensor, children: *Node.Children, ctx: *@This()) !void {
            const _y_pred = children.get_bwd_upcast(Tensor, 0) orelse return;
            const _y = children.get_upcast(Tensor, 1);
            const _beta = ctx.beta;

            const _n = @as(T, @floatFromInt(_y.get_size()));

            for (try _y_pred.ensure_grad_data(0), _y_pred.get_data(), _y.get_data()) |*grad_val, pred_val, target_val| {
                const diff = pred_val - target_val;
                if (@abs(diff) < _beta) {
                    grad_val.* += diff / (_beta * _n);
                } else {
                    grad_val.* += std.math.sign(diff) / _n;
                }
            }
        }
    };

    return Tensor.create_dependent(Sl1LossBwd, .{
        .data = try NDArray(T).from_slice(&.{loss}, &.{1}, y_pred.device),
        .children = &.{ &y_pred.node, &y.node },
        .label = "smooth_l1",
        .callback = .{ .beta = beta },
        .gb = y_pred.node.gb,
        .device = y_pred.device,
    });
}

/// Naive softmax 1D that uses on autograd.
/// This autograd variant is intended to by used as part of autograd system test and verification
/// dedicated kernels should generally be used for such operations.
pub fn ag_softmax_1d(T: type, input: *NDTensor(T)) !*NDTensor(T) {
    const max_val = try input.max();
    const exp_input = try (try input.sub(max_val)).exp();
    const sum = try exp_input.sum();
    return exp_input.div(sum);
}

/// Naive Mean Squared Error 1D loss that uses on autograd.
/// This autograd variant is intended to by used as part of autograd system test and verification
/// dedicated kernels should generally be used for such operations.
pub fn ag_mse_1d(T: type, y_pred: *NDTensor(T), y: *NDTensor(T), device: DeviceReference) !*NDTensor(T) {
    var diff = try y_pred.sub(y);
    if (debug) try diff.set_label("diff");

    const sq_diff = try diff.pow(2);
    if (debug) try sq_diff.set_label("sq_diff");

    const sum_sq_diff = try sq_diff.sum();
    if (debug) try sum_sq_diff.set_label("sum_sq_diff");

    const coef = @as(T, @floatFromInt(y.get_size()));
    const coef_tensor = try NDTensor(T).from_slice(&.{coef}, null, true, device);
    if (debug) try coef_tensor.set_label("coef");

    const out = try sum_sq_diff.div(coef_tensor);
    if (debug) try out.set_label("mse");

    return out;
}
