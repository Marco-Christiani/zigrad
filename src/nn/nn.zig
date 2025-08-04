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

            // output shape -> (...batch_dims, out_features)
            var output_shape = try std.ArrayList(usize).initCapacity(std.heap.page_allocator, batch_dims.len + 1);
            defer output_shape.deinit();

            try output_shape.appendSlice(batch_dims);
            try output_shape.append(out_features);

            // x @ weights^T
            const mm_result = try x.bmm_acc(weights, output_shape.items, .{ .trans_b = true });
            errdefer mm_result.deinit();

            // Bias
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

/// Parameter initialization utilities
pub const init = struct {
    pub const InitOpts = struct {
        optim: ?zg.Optimizer = null,
        bias_init: ?enum { zeros, ones, normal } = .zeros,
        weight_init: zg.RandType = .normal,
        requires_grad: bool = true,
        /// Comptime-known suffix to add to tensor labels
        label_suffix: []const u8 = "",
    };

    /// Linear layer parameters
    pub fn LinearParams(comptime T: type) type {
        return struct {
            const Tensor = zg.NDTensor(T);
            weights: *Tensor,
            bias: ?*Tensor,

            const Self = @This();

            /// Free weight and bias
            pub fn deinit(self: *Self) void {
                self.weights.release();
                self.weights.deinit();
                if (self.bias) |b| {
                    b.release();
                    b.deinit();
                }
                self.* = undefined;
            }
        };
    }

    /// Convenience method to initialize linear layer parameters
    pub fn init_linear(
        comptime T: type,
        /// Target device
        device: DeviceReference,
        /// Input feature dimension
        in_features: usize,
        /// Output feature dimension
        out_features: usize,
        /// Initialization options
        comptime opts: InitOpts,
    ) !LinearParams(T) {
        const Tensor = zg.NDTensor(T);
        const weights = switch (opts.weight_init) {
            .kaiming => try Tensor.random(
                device,
                &.{ out_features, in_features },
                .{ .kaiming = in_features },
                .{ .label = "linear_weights" ++ opts.label_suffix, .requires_grad = opts.requires_grad, .acquired = true },
            ),
            .uniform => try Tensor.random(
                device,
                &.{ out_features, in_features },
                .uniform,
                .{ .label = "linear_weights" ++ opts.label_suffix, .requires_grad = opts.requires_grad, .acquired = true },
            ),
            .normal => try Tensor.random(
                device,
                &.{ out_features, in_features },
                .normal,
                .{ .label = "linear_weights" ++ opts.label_suffix, .requires_grad = opts.requires_grad, .acquired = true },
            ),
        };
        errdefer weights.deinit();

        const bias = if (opts.bias_init) |bi| switch (bi) {
            .zeros => try Tensor.zeros(
                device,
                &.{out_features},
                .{ .label = "linear_bias" ++ opts.label_suffix, .requires_grad = opts.requires_grad, .acquired = true },
            ),
            .ones => try Tensor.ones(
                device,
                &.{out_features},
                .{ .label = "linear_bias" ++ opts.label_suffix, .requires_grad = opts.requires_grad, .acquired = true },
            ),
            .normal => try Tensor.random(
                device,
                &.{out_features},
                .normal,
                .{ .label = "linear_bias" ++ opts.label_suffix, .requires_grad = opts.requires_grad, .acquired = true },
            ),
        } else null;
        errdefer if (bias) |b| b.deinit();

        if (opts.optim) |optim| {
            try optim.attach(weights);
            if (bias) |b| try optim.attach(b);
        }

        return .{ .weights = weights, .bias = bias };
    }
};

/// Higher-level functional model construction utilities
pub const blocks = struct {
    /// Convenience for building Multi-layer perceptron blocks
    ///
    /// ## Example usage
    /// ```zig
    /// const mlp = blocks.MLP(f32){
    ///     .layer_sizes = &.{ 784, 128, 64, 10 },
    ///     .activation = .relu,
    /// };
    ///
    /// var params = try mlp.init_params(device, .{ .optim = optimizer });
    /// defer params.deinit();
    ///
    /// const output = try mlp.forward(params, input);
    /// ```
    pub fn MLP(comptime T: type) type {
        return struct {
            const Tensor = zg.NDTensor(T);
            layer_sizes: []const usize,
            activation: enum { relu, tanh, sigmoid },

            const Self = @This();

            /// MLP Parameters
            pub const MLPParams = struct {
                layers: []init.LinearParams(T),
                allocator: std.mem.Allocator,

                pub fn deinit(self: *@This()) void {
                    for (self.layers) |*layer| {
                        layer.deinit();
                    }
                    self.allocator.free(self.layers);
                    self.* = undefined;
                }
            };

            /// Initialize MLP parameters
            pub fn init_params(
                self: Self,
                /// Allocator for layer param ptrs
                allocator: std.mem.Allocator,
                /// Target device
                device: DeviceReference,
                /// Initialization options
                comptime opts: struct {
                    optim: ?zg.Optimizer = null,
                },
            ) !MLPParams {
                const layers = try allocator.alloc(init.LinearParams(T), self.layer_sizes.len - 1);
                errdefer allocator.free(layers);

                for (layers, 0..) |*layer, i| {
                    layer.* = try init.init_linear(
                        T,
                        device,
                        self.layer_sizes[i],
                        self.layer_sizes[i + 1],
                        .{ .optim = opts.optim },
                    );
                }

                return .{ .layers = layers, .allocator = allocator };
            }

            /// Forward pass through the MLP
            pub fn forward(
                self: Self,
                /// MLP state
                params: MLPParams,
                /// Input
                x: *Tensor,
            ) !*Tensor {
                var current = x;

                for (params.layers, 0..) |layer, i| {
                    const linear_out = try nn(T).linear(current, layer.weights, layer.bias);

                    if (i < params.layers.len - 1) {
                        defer linear_out.soft_deinit();
                        switch (self.activation) {
                            .relu => try nn(T).relu_(linear_out),
                            .tanh => try nn(T).tanh_(linear_out),
                            .sigmoid => try nn(T).sigmoid_(linear_out),
                        }
                    }

                    current = linear_out;
                }

                return current;
            }
        };
    }
};

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
        const linear_params = try init.init_linear(
            f32,
            device,
            784,
            128,
            .{
                .requires_grad = true,
            },
        );

        // Forward
        const linear_out = try nn(f32).linear(input, linear_params.weights, linear_params.bias);
        defer linear_out.deinit();

        const relu_out = try nn(f32).relu(linear_out);
        defer relu_out.deinit();

        // Check shapes
        try testing.expectEqual(32, relu_out.get_dim(0));
        try testing.expectEqual(128, relu_out.get_dim(1));
    }
    {
        // zg.runtime.grad_enabled = false;
        const input = try zg.NDTensor(f32).random(device, &.{ 32, 784 }, .normal, .{
            .requires_grad = true,
        });
        defer input.deinit();
        const mlp = blocks.MLP(f32){
            .layer_sizes = &.{ 784, 128, 64, 10 },
            .activation = .relu,
        };

        var params = try mlp.init_params(testing.allocator, device, .{ .optim = null });
        defer params.deinit();

        const output = try mlp.forward(params, input);

        try testing.expectEqual(10, output.get_dim(1));
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
