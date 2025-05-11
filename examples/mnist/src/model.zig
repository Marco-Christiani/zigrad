const std = @import("std");
const zg = @import("zigrad");
const opspec = zg.opspec;
const DeviceReference = zg.DeviceReference;
const GraphManager = zg.GraphManager;
const NDTensor = zg.NDTensor;

pub fn MnistModel(comptime T: type, Optimizer: type) type {
    return struct {
        const Self = @This();
        const Tensor = NDTensor(T);
        const Layers = 3;
        linear_layers: [Layers]LinearLayer(T) = undefined,
        flatten: FlattenLayer(T) = .{},
        relu: ReLULayer(T) = .{},

        pub fn init(
            gm: *GraphManager,
            device: DeviceReference,
            optim: ?*Optimizer,
        ) !Self {
            var self: Self = .{
                .linear_layers = .{
                    try LinearLayer(T).init(gm, device, 28 * 28, 128),
                    try LinearLayer(T).init(gm, device, 128, 64),
                    try LinearLayer(T).init(gm, device, 64, 10),
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

            return self.linear_layers[2].forward(a1);
        }
    };
}

pub fn LinearLayer(comptime T: type) type {
    return struct {
        const Self = @This();
        const Tensor = NDTensor(T);
        weights: *Tensor,
        bias: *Tensor,
        pub fn init(gm: *GraphManager, device: DeviceReference, in_features: usize, out_features: usize) !Self {
            const weights = try Tensor.random(&.{ out_features, in_features }, .{ .kaiming = in_features }, .{
                .requires_grad = true,
                .device = device,
                .heap = gm.heap(),
                .label = "linear_weights",
                .acquired = true,
            });
            errdefer weights.deinit();

            const bias = try Tensor.zeros(&.{out_features}, .{
                .requires_grad = true,
                .device = device,
                .heap = gm.heap(),
                .label = "linear_bias",
                .acquired = true,
            });

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
            const y = try x.bmm_acc(self.weights, &.{ batch_size, n_features }, .{ .trans_b = true });
            try self.bias.add_(y);
            return y;
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
                .heap = x._heap,
                .callback = .{},
                .label = "relu_out",
            });
        }

        pub fn callback(y: *Tensor, children: *Tensor.Children) !void {
            const x = children.get_bwd(0) orelse return;
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
        const Tensor = NDTensor(T);

        pub fn forward(_: Self, input: *Tensor) !*Tensor {
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
                .heap = input._heap,
            });
            result.data._reshape(&.{ batch_dim, flattened_dim });
            return result;
        }

        pub fn callback(y: *Tensor, children: *Tensor.Children) !void {
            const x = children.get_bwd(0) orelse return;
            const x_grad = try x.ensure_grad(0);
            y.device.dispatch(opspec.add(T){
                .x = x_grad.data,
                .y = y.assume_grad_data(),
                .z = x_grad.data,
            });
        }
    };
}
