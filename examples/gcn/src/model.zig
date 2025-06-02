const std = @import("std");
const zg = @import("zigrad");
const opspec = zg.opspec;
const DeviceReference = zg.DeviceReference;
const NDTensor = zg.NDTensor;
const winit = zg.winit;

pub fn GCN(comptime T: type, Optimizer: type) type {
    return struct {
        const Self = @This();
        const Layers = 3;
        conv1: GraphConvLayer(T),
        conv2: GraphConvLayer(T),
        relu: ReLULayer(T) = .{},

        pub fn init(device: DeviceReference, in_features: usize, out_features: usize, optim: ?*Optimizer) !Self {
            const self: Self = .{
                .conv1 = try GraphConvLayer(T).init(device, in_features, 16),
                .conv2 = try GraphConvLayer(T).init(device, 16, out_features),
            };
            if (optim) |o| {
                try o.attach(self.conv1.lin.weights);
                try o.attach(self.conv1.bias);
                try o.attach(self.conv2.lin.weights);
                try o.attach(self.conv2.bias);
            }
            return self;
        }

        pub fn deinit(self: *Self) void {
            self.conv1.deinit();
            self.conv2.deinit();
        }

        pub fn forward(self: *Self, x: *NDTensor(T), edge_index: *NDTensor(usize)) !*NDTensor(T) {
            const c1 = try self.conv1.forward(x, edge_index);
            errdefer c1.deinit();

            const r1 = try self.relu.forward(c1);
            errdefer r1.deinit();

            const c2 = try self.conv2.forward(r1, edge_index);
            errdefer c2 = c2.deinit();

            return c2;
        }
    };
}

pub fn GraphConvLayer(comptime T: type) type {
    return struct {
        const Self = @This();
        const Tensor = NDTensor(T);

        device: DeviceReference,
        lin: LinearLayer(T),
        bias: *Tensor,

        adj_mat: ?*Tensor = null,

        pub fn init(device: DeviceReference, in_features: usize, out_features: usize) !Self {
            const lin = try LinearLayer(T).init(device, in_features, out_features);

            const bias = try Tensor.zeros(&.{ 1, out_features }, true, device);
            errdefer bias.deinit();
            try bias.set_label("conv_bias");
            bias.acquire();

            return .{
                .device = device,
                .lin = lin,
                .bias = bias,
            };
        }

        pub fn deinit(self: *Self) void {
            self.lin.deinit();

            self.bias.release();
            self.bias.deinit();

            if (self.adj_mat) |adj_mat| {
                adj_mat.deinit();
            }
        }

        pub fn forward(self: *Self, x: *Tensor, edge_index: *NDTensor(usize)) !*Tensor {
            const h = try self.lin.forward(x);
            if (self.adj_mat == null) {
                try self.normalize(x.get_dim(0), edge_index);
            }
            const out = try self.adj_mat.?.bmm_acc(h, h.get_shape(), .{});
            try out.add_(self.bias);
            return out;
        }

        pub fn normalize(self: *Self, n_node: usize, edge_index: *NDTensor(usize)) !void {
            const n_edge = edge_index.get_dim(1);

            var adj_mat = try Tensor.empty(&.{ n_node, n_node }, true, self.device);
            errdefer adj_mat.deinit();

            var deg_map = std.AutoHashMap(usize, T).init(self.device.allocator);
            defer deg_map.deinit();
            for (0..n_edge) |i| {
                const source = edge_index.get(i * 2);
                if (deg_map.get(source)) |v| {
                    try deg_map.put(source, v + 1);
                } else {
                    try deg_map.put(source, 2); // the initial is 1 (self loop) + current 1;
                }
            }

            for (0..n_edge) |i| {
                const source = edge_index.get(i * 2);
                const target = edge_index.get(i * 2 + 1);
                adj_mat.set(source * n_node + target, std.math.pow(T, deg_map.get(source).?, -0.5));
            }
            self.adj_mat = adj_mat;
        }
    };
}

pub fn LinearLayer(comptime T: type) type {
    return struct {
        const Self = @This();
        const Tensor = NDTensor(T);
        weights: *Tensor,
        pub fn init(device: DeviceReference, in_features: usize, out_features: usize) !Self {
            var weights = try Tensor.empty(&.{ out_features, in_features }, true, device);
            errdefer weights.deinit();

            try weights.set_label("linear_weights");

            weights.acquire();

            winit.he_init(T, weights);

            return .{ .weights = weights };
        }

        pub fn deinit(self: *Self) void {
            self.weights.release();
            self.weights.deinit();
        }

        pub fn forward(self: *const Self, x: *Tensor) !*Tensor {
            const batch_size = if (x.data.shape.len > 1) x.get_dim(0) else 1;
            const n_features = self.weights.data.shape.get(0);
            const y = x.bmm_acc(self.weights, &.{ batch_size, n_features }, .{ .trans_b = true });
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

// TODO: I not sure I doing right with autograd or there a easy way to do this
/// this layer Filter unused data base on mask
/// like pytorch out[train_mask]
pub fn MaskLayer(comptime T: type) type {
    return struct {
        const Self = @This();
        const Tensor = NDTensor(T);

        mask: *NDTensor(bool) = undefined,

        pub fn forward(_: Self, x: *Tensor, mask: *NDTensor(bool)) !*Tensor {
            const num_feature = x.get_dim(1);
            const buf = try x.device.allocator.alloc(T, x.get_size());
            defer x.device.allocator.free(buf);
            var count: usize = 0;
            for (mask.get_data(), 0..) |ok, i| {
                if (ok) {
                    const buf_start = count * num_feature;
                    const buf_end = count * num_feature + num_feature;
                    const x_start = i * num_feature;
                    const x_end = i * num_feature + num_feature;
                    std.mem.copyForwards(T, buf[buf_start..buf_end], x.data.data[x_start..x_end]);
                    count += 1;
                }
            }
            const y = try Tensor.DataType.init(
                buf[0 .. count * num_feature],
                &.{ count, num_feature },
                x.device,
            );

            return Tensor.create_dependent(Self, .{
                .data = y,
                .children = &.{x},
                .device = x.device,
                .callback = .{
                    .mask = mask,
                },
                .label = "mask",
            });
        }

        pub fn callback(y: *Tensor, children: *Tensor.Children, self: *Self) !void {
            const x = children.get_bwd(0) orelse return;
            const num_feature = x.get_dim(1);

            var x_grad = try x.ensure_grad_data(0);
            const y_grad = y.assume_grad_data();

            for (self.mask.get_data(), 0..) |ok, i| {
                const start = i * num_feature;
                const end = start + num_feature;
                if (ok) {
                    std.mem.copyForwards(
                        T,
                        x_grad[start..end],
                        y_grad[start..end],
                    );
                }
            }
            return;
        }
    };
}
