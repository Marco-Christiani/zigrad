const std = @import("std");
const zg = @import("zigrad");
const opspec = zg.opspec;
const DeviceReference = zg.DeviceReference;
const Graph = zg.Graph;
const Node = Graph.Node;
const NDTensor = zg.NDTensor;

pub fn MnistModel(comptime T: type) type {
    return struct {
        const Self = @This();
        const Tensor = NDTensor(T);
        const Layers = 3;
        linear_layers: [Layers]LinearLayer(T) = undefined,
        flatten: FlattenLayer(T) = .{},

        pub fn init(device: DeviceReference, opts: struct {
            optim: ?zg.Optimizer = null,
        }) !Self {
            return .{
                .linear_layers = .{
                    try LinearLayer(T).init(device, 28 * 28, 128, .{ .optim = opts.optim }),
                    try LinearLayer(T).init(device, 128, 64, .{ .optim = opts.optim }),
                    try LinearLayer(T).init(device, 64, 10, .{ .optim = opts.optim }),
                },
            };
        }

        pub fn deinit(self: *Self) void {
            for (&self.linear_layers) |*l| l.deinit();
            self.* = undefined;
        }

        pub fn forward(self: *Self, x: *Tensor) !*Tensor {
            const flat = try self.flatten.forward(x);
            errdefer flat.deinit();

            const z0 = try self.linear_layers[0].forward(flat);
            errdefer z0.deinit();
            try zg.nn.relu_(T, z0);

            if (flat.should_deinit())
                flat.deinit();

            const z1 = try self.linear_layers[1].forward(z0);
            errdefer z1.deinit();
            try zg.nn.relu_(T, z1);

            if (z0.should_deinit())
                z0.deinit();

            const z2 = try self.linear_layers[2].forward(z1);

            if (z1.should_deinit())
                z1.deinit();

            return z2;
        }

        pub fn zero_grad(self: *Self) void {
            for (&self.linear_layers) |*l| {
                if (l.weights.grad) |_| l.weights.setup_grad(0) catch {};
                if (l.bias.grad) |_| l.bias.setup_grad(0) catch {};
            }
        }
    };
}

pub fn LinearLayer(comptime T: type) type {
    return struct {
        const Self = @This();
        const Tensor = NDTensor(T);
        weights: *Tensor,
        bias: *Tensor,
        pub fn init(device: DeviceReference, in_features: usize, out_features: usize, opts: struct {
            optim: ?zg.Optimizer = null,
        }) !Self {
            const weights = try Tensor.random(device, &.{ out_features, in_features }, .{ .kaiming = in_features }, .{
                .label = "linear_weights",
                .requires_grad = true,
                .acquired = true,
            });
            errdefer weights.deinit();

            const bias = try Tensor.zeros(device, &.{out_features}, .{
                .label = "linear_bias",
                .requires_grad = true,
                .acquired = true,
            });
            errdefer bias.deinit();

            if (opts.optim) |optim| {
                try optim.attach(weights);
                try optim.attach(bias);
            }

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

pub fn FlattenLayer(comptime T: type) type {
    return struct {
        const Self = @This();
        const Tensor = NDTensor(T);

        pub fn forward(_: Self, input: *Tensor) !*Tensor {
            const batch_dim = input.data.shape.get(0);
            const other_dims = input.data.shape.crop(1, 0);
            const flattened_dim = zg.arrayutils.prod(other_dims);

            // view of input tensor with new shape
            const result = try Tensor.create_dependent(Self, .{
                .data = try input.data.copy(input.device), // Reuse the same data
                .children = &.{&input.node},
                .label = "flattened",
                .callback = .{},
                .device = input.device,
                .gb = input.node.gb,
            });
            result.data._reshape(&.{ batch_dim, flattened_dim });
            return result;
        }

        pub fn backward(y: *Tensor, children: *zg.Graph.Node.Children) !void {
            const x = children.get_bwd_upcast(Tensor, 0) orelse return;
            const x_grad = try x.ensure_grad(0);
            y.device.dispatch(opspec.add(T){
                .x = x_grad.get_data(),
                .y = y.assume_grad_data(),
                .z = x_grad.get_data(),
            });
        }
    };
}
