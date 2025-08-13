//! Functional Neural Network Ops

const std = @import("std");

const zg = @import("../zigrad.zig");
const opspec = zg.opspec;
const DeviceReference = zg.DeviceReference;
const Node = zg.Graph.Node;

/// Functional Neural Network Operations
pub fn nn(comptime T: type) type {
    return struct {
        const Tensor = zg.NDTensor(T);
        /// Linear transformation: $y = x @ w' + b$
        ///
        /// Returns output tensor of shape [..., out_features]
        pub fn linear(
            /// Input tensor of shape [..., in_features]
            x: *Tensor,
            /// weights: Weight tensor of shape [out_features, in_features]
            weights: *Tensor,
            /// bias: Bias tensor of shape [out_features]
            bias: ?*Tensor,
        ) !*Tensor {
            std.debug.assert(x.device.is_compatible(weights.device));
            if (bias) |b| std.debug.assert(x.device.is_compatible(b.device));

            const batch_dims = x.data.shape.slice()[0 .. x.data.shape.len - 1];
            const in_features = x.data.shape.get(x.data.shape.len - 1);
            const out_features = weights.data.shape.get(0);

            // Verify shapes
            std.debug.assert(weights.data.shape.get(1) == in_features);
            if (bias) |b| std.debug.assert(b.data.shape.get(0) == out_features);
            if (bias) |b| std.debug.assert(b.data.shape.len == 1);

            // x @ weights^T
            const mm_result = try x.bmm(weights, .{ .trans_b = true });
            errdefer mm_result.deinit();
            // output shape -> (...batch_dims, out_features)
            std.debug.assert(std.mem.eql(usize, batch_dims, mm_result.get_shape()[0..batch_dims.len]));
            std.debug.assert(out_features == mm_result.get_shape()[batch_dims.len]);
            std.debug.assert(mm_result.get_shape().len == batch_dims.len + 1);

            // Bias
            // TODO: in-place add leaks a node reference unless we disable grad, this is on our radar
            if (bias) |b| try mm_result._add(b);

            return mm_result;
        }

        /// Rectified Linear Unit activation: y = max(0, x)
        ///
        /// Returns output tensor with same shape as `x`
        pub fn relu(
            /// Input tensor
            x: *Tensor,
        ) !*Tensor {
            const ReluBwd = struct {
                pub fn backward(y: *Tensor, children: *Node.Children) !void {
                    const input = children.get_bwd_upcast(Tensor, 0) orelse return;
                    y.device.dispatch(opspec.relu_bwd(T){
                        .x = input.get_data(),
                        .x_g = try input.ensure_grad_data(0),
                        .y_g = y.assume_grad_data(),
                    });
                }
            };

            const output = try Tensor.DataType.empty(x.get_shape(), x.device);

            x.device.dispatch(opspec.relu_fwd(T){
                .x = x.get_data(),
                .y = output.get_data(),
            });

            return try Tensor.create_dependent(ReluBwd, .{
                .data = output,
                .children = &.{&x.node},
                .device = x.device,
                .gb = x.node.gb,
                .callback = .{},
            });
        }

        /// In-place Rectified Linear Unit activation
        pub fn relu_(
            /// Input tensor - modified in place.
            x: *Tensor,
        ) !void {
            const ReluInplaceBwd = struct {
                version: u8,
                pub fn backward(tensor: *Tensor, _: *Node.Children, ctx: *@This()) !void {
                    std.debug.assert(ctx.version == tensor.node.version);
                    tensor.device.dispatch(opspec.relu_inplace_bwd(T){
                        .x = tensor.get_data(),
                        .x_g = try tensor.ensure_grad_data(0),
                    });
                }
            };

            x.device.dispatch(opspec.relu_fwd(T){
                .x = x.get_data(),
                .y = x.get_data(),
            });

            try Tensor.prepend_dependent(ReluInplaceBwd, x, .{
                .callback = .{ .version = x.node.version +% 1 },
                .children = &.{},
            });
        }

        /// Hyperbolic tangent activation: $y = \tanh(x)$
        ///
        /// Returns output tensor with same shape as input
        pub fn tanh(
            /// Input tensor
            x: *Tensor,
        ) !*Tensor {
            const TanhBwd = struct {
                pub fn backward(y: *Tensor, children: *Node.Children) !void {
                    const input = children.get_bwd_upcast(Tensor, 0) orelse return;
                    y.device.dispatch(opspec.tanh_bwd(T){
                        .x_g = try input.ensure_grad_data(0),
                        .y = y.get_data(),
                        .y_g = y.assume_grad_data(),
                    });
                }
            };

            const output = try Tensor.DataType.empty(x.get_shape(), x.device);

            x.device.dispatch(opspec.tanh_fwd(T){
                .x = x.get_data(),
                .y = output.get_data(),
            });

            return try Tensor.create_dependent(TanhBwd, .{
                .data = output,
                .children = &.{&x.node},
                .device = x.device,
                .gb = x.node.gb,
                .callback = .{},
            });
        }

        /// In-place hyperbolic tangent activation
        pub fn tanh_(
            ///Input tensor - modified in place
            x: *Tensor,
        ) !void {
            const TanhInplaceBwd = struct {
                version: u8,
                pub fn backward(tensor: *Tensor, _: *Node.Children, ctx: *@This()) !void {
                    std.debug.assert(ctx.version == tensor.node.version);
                    tensor.device.dispatch(opspec.tanh_inplace_bwd(T){
                        .x = tensor.get_data(),
                        .x_g = try tensor.ensure_grad_data(0),
                    });
                }
            };

            x.device.dispatch(opspec.tanh_fwd(T){
                .x = x.get_data(),
                .y = x.get_data(),
            });

            try Tensor.prepend_dependent(TanhInplaceBwd, x, .{
                .callback = .{ .version = x.node.version +% 1 },
                .children = &.{},
            });
        }

        /// Sigmoid activation: y = 1 / (1 + exp(-x))
        ///
        /// Returns output tensor with same shape as input
        pub fn sigmoid(
            /// Input tensor
            x: *Tensor,
        ) !*Tensor {
            const SigmoidBwd = struct {
                pub fn backward(y: *Tensor, children: *Node.Children) !void {
                    const input = children.get_bwd_upcast(Tensor, 0) orelse return;
                    y.device.dispatch(opspec.sigm_bwd(T){
                        .x_g = try input.ensure_grad_data(0),
                        .y = y.get_data(),
                        .y_g = y.assume_grad_data(),
                    });
                }
            };

            const output = try Tensor.DataType.empty(x.get_shape(), x.device);

            x.device.dispatch(opspec.sigm_fwd(T){
                .x = x.get_data(),
                .y = output.get_data(),
            });

            return try Tensor.create_dependent(SigmoidBwd, .{
                .data = output,
                .children = &.{&x.node},
                .device = x.device,
                .gb = x.node.gb,
                .callback = .{},
            });
        }

        /// In-place sigmoid activation
        pub fn sigmoid_(
            /// Input tensor - modified in place
            x: *Tensor,
        ) !void {
            const SigmoidInplaceBwd = struct {
                version: u8,
                pub fn backward(tensor: *Tensor, _: *Node.Children, ctx: *@This()) !void {
                    std.debug.assert(ctx.version == tensor.node.version);
                    tensor.device.dispatch(opspec.sigm_inplace_bwd(T){
                        .x = tensor.get_data(),
                        .x_g = try tensor.ensure_grad_data(0),
                    });
                }
            };

            x.device.dispatch(opspec.sigm_fwd(T){
                .x = x.get_data(),
                .y = x.get_data(),
            });

            try Tensor.prepend_dependent(SigmoidInplaceBwd, x, .{
                .callback = .{ .version = x.node.version +% 1 },
                .children = &.{},
            });
        }

        /// Mean Squared Error $\text{loss} = (\hat{y}- y)^2 / n$
        pub fn mse(
            pred: *Tensor,
            target: *Tensor,
        ) !*Tensor {
            std.debug.assert(std.mem.eql(usize, pred.get_shape(), target.get_shape()));

            const MseBwd = struct {
                pub fn backward(loss: *Tensor, children: *Node.Children) !void {
                    const pred_tensor = children.get_bwd_upcast(Tensor, 0) orelse return;
                    const target_tensor = children.get_bwd_upcast(Tensor, 1) orelse return;

                    if (pred_tensor.grad) |_| {
                        loss.device.dispatch(opspec.mse_bwd(T){
                            .pred = pred_tensor.get_data(),
                            .target = target_tensor.get_data(),
                            .pred_grad = try pred_tensor.ensure_grad_data(0),
                            .loss_grad = loss.assume_grad_data(),
                            .n = pred_tensor.get_size(),
                        });
                    }
                }
            };

            const output = try Tensor.DataType.empty(&.{1}, pred.device);

            pred.device.dispatch(opspec.mse_fwd(T){
                .pred = pred.get_data(),
                .target = target.get_data(),
                .loss = output.get_data(),
                .n = pred.get_size(),
            });

            return try Tensor.create_dependent(MseBwd, .{
                .data = output,
                .children = &.{ &pred.node, &target.node },
                .device = pred.device,
                .gb = pred.node.gb,
                .callback = .{},
            });
        }
    };
}

test "functional API" {
    const testing = std.testing;

    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();

    zg.global_graph_init(std.testing.allocator, .{});
    defer zg.global_graph_deinit();

    const device = cpu.reference();

    {
        const input = try zg.NDTensor(f32).random(device, &.{ 32, 784 }, .normal, .{
            .requires_grad = true,
        });
        defer input.deinit();

        // Create and init params
        const weights = try zg.NDTensor(f32).random(device, &.{ 128, 784 }, .uniform, .{ .acquired = true, .requires_grad = true });
        const bias = try zg.NDTensor(f32).zeros(device, &.{128}, .{ .acquired = true, .requires_grad = true });

        // Forward
        const linear_out = try nn(f32).linear(input, weights, bias);
        defer linear_out.deinit();

        const relu_out = try nn(f32).relu(linear_out);
        defer relu_out.deinit();

        // Check shapes
        try testing.expectEqual(32, relu_out.get_dim(0));
        try testing.expectEqual(128, relu_out.get_dim(1));
    }

    {
        const preds = try zg.NDTensor(f32).from_slice(
            device,
            &.{ 2, 3, 5 },
            &.{3},
            .{
                .requires_grad = true,
            },
        );
        defer preds.deinit();

        const targets = try zg.NDTensor(f32).from_slice(
            device,
            &.{ 1, 2, 2 },
            &.{3},
            .{},
        );
        defer targets.deinit();

        // [(1-2)^2 + (2-3)^2 + (5-2)^2] / 3
        // = [1 + 1 + 9] / 3
        const loss = try nn(f32).mse(preds, targets);
        loss.deinit();
        try testing.expectApproxEqAbs(11.0 / 3.0, loss.get(0), 0.001);
    }
}
