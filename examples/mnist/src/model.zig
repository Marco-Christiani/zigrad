const std = @import("std");
const zg = @import("zigrad");
const opspec = zg.opspec;
const DeviceReference = zg.DeviceReference;
const NDTensor = zg.NDTensor;
const winit = zg.winit;

pub fn MnistModel(comptime T: type, Optimizer: type) type {
    return struct {
        const Self = @This();
        const Tensor = NDTensor(T);
        const Layers = 3;
        linear_layers: [Layers]LinearLayer(T) = undefined,
        flatten: FlattenLayer(T) = .{},
        relu: ReLULayer(T) = .{},

        pub fn init(device: DeviceReference, optim: ?*Optimizer) !Self {
            var self: Self = .{
                .linear_layers = .{
                    try LinearLayer(T).init(device, 28 * 28, 128),
                    try LinearLayer(T).init(device, 128, 64),
                    try LinearLayer(T).init(device, 64, 10),
                },
            };

            if (optim) |_optim| {
                for (&self.linear_layers) |l| {
                    try _optim.attach(l.weights);
                    try _optim.attach(l.bias);
                }
            }

            return self;
        }

        pub fn deinit(self: *Self) void {
            for (&self.linear_layers) |*l| l.deinit();
            self.* = undefined;
        }

        pub fn forward(self: *Self, x: *Tensor) !*Tensor {
            const flat = try self.flatten.forward(x);
            errdefer flat.deinit();

            const f0 = try self.linear_layers[0].forward(flat);
            errdefer f0.deinit();
            const a0 = try self.relu.forward(f0);
            errdefer a0.deinit();

            const f1 = try self.linear_layers[1].forward(a0);
            errdefer f1.deinit();
            const a1 = try self.relu.forward(f1);
            errdefer a1.deinit();

            const f2 = try self.linear_layers[2].forward(a1);
            errdefer f2.deinit();

            return try self.relu.forward(f2);
        }
    };
}

pub fn LinearLayer(comptime T: type) type {
    return struct {
        const Self = @This();
        const Tensor = NDTensor(T);
        weights: *Tensor,
        bias: *Tensor,
        pub fn init(device: DeviceReference, in_features: usize, out_features: usize) !Self {
            var weights = try Tensor.empty(&.{ out_features, in_features }, true, device);
            errdefer weights.deinit();

            var bias = try Tensor.zeros(&.{ 1, out_features }, true, device);
            errdefer bias.deinit();

            try weights.set_label("linear_weights");
            try bias.set_label("linear_bias");

            weights.acquire();
            bias.acquire();

            winit.he_init(T, weights);

            return .{ .weights = weights, .bias = bias };
        }

        pub fn deinit(self: *Self) void {
            self.weights.release();
            self.weights.deinit();

            self.bias.release();
            self.bias.deinit();
        }

        pub fn forward(self: *const Self, x: *Tensor) !*Tensor {
            const batch_size = if (x.data.shape.len > 1) x.get_dim(0) else 1;
            const n_features = self.weights.data.shape.get(0);
            const b_y = try x.bmm_acc(self.weights, &.{ batch_size, n_features }, .{ .trans_b = true });
            return self.bias.add(b_y);
        }
    };
}

pub fn ReLULayer(comptime T: type) type {
    return struct {
        const Self = @This();
        const Tensor = NDTensor(T);

        pub fn forward(_: Self, x: *Tensor) !*Tensor {
            const y = try Tensor.DataType.empty(x.get_shape(), x.device);

            x.device.dispatch(opspec.relu_fwd(T){
                .x = x.get_data(),
                .y = y.data,
            });

            return Tensor.create_dependent(Self, .{
                .data = y,
                .children = &.{x},
                .device = x.device,
                .callback = .{},
                .label = "relu_out",
            });
        }

        pub fn callback(y: *Tensor) !void {
            const x = y.backward_child(0) orelse return;
            y.device.dispatch(opspec.relu_bwd(T){
                .x = x.get_data(),
                .x_g = try x.ensure_grad_data(0),
                .y_g = y.assume_grad_data(),
            });
        }
    };
}

pub fn FlattenLayer(comptime T: type) type {
    return struct {
        const Self = @This();

        pub fn forward(_: Self, input: *NDTensor(T)) !*NDTensor(T) {
            const batch_dim = input.data.shape.get(0);
            const other_dims = input.data.shape.crop(1, 0);
            const flattened_dim = zg.arrayutils.prod(other_dims);

            // view of input tensor with new shape
            const result = try NDTensor(T).create_dependent(Self, .{
                .data = try input.data.copy(input.device), // Reuse the same data
                .children = &.{input},
                .label = "flattened",
                .callback = .{},
                .device = input.device,
            });
            result.data._reshape(&.{ batch_dim, flattened_dim });

            return result;
        }

        pub fn callback(y: *NDTensor(T)) !void {
            const x = y.backward_child(0) orelse return;
            const x_grad = try x.ensure_grad(0);
            y.device.dispatch(opspec.add(T){
                .x = x_grad.data,
                .y = y.assume_grad_data(),
                .z = x_grad.data,
            });
        }
    };
}
