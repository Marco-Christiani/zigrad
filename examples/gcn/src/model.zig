const std = @import("std");
const zg = @import("zigrad");
const opspec = zg.opspec;
const Graph = zg.Graph;
const Node = Graph.Node;
const DeviceReference = zg.DeviceReference;
const NDTensor = zg.NDTensor;
const winit = zg.winit;

pub fn GCN(comptime T: type) type {
    return struct {
        const Self = @This();
        const Layers = 3;
        conv1: GraphConvLayer(T),
        conv2: GraphConvLayer(T),

        pub fn init(device: DeviceReference, in_features: usize, out_features: usize, opts: struct {
            optim: ?zg.Optimizer = null,
        }) !Self {
            const self: Self = .{
                .conv1 = try GraphConvLayer(T).init(device, in_features, 16, .{ .optim = opts.optim }),
                .conv2 = try GraphConvLayer(T).init(device, 16, out_features, .{ .optim = opts.optim }),
            };
            return self;
        }

        pub fn deinit(self: *Self) void {
            self.conv1.deinit();
            self.conv2.deinit();
        }

        pub fn forward(self: *Self, x: *NDTensor(T), edge_index: *NDTensor(usize)) !*NDTensor(T) {
            const c1 = try self.conv1.forward(x, edge_index);
            errdefer c1.deinit();
            defer {
                if (c1.should_deinit()) {
                    c1.deinit();
                }
            }
            try zg.nn.relu_(T, c1);
            const c2 = try self.conv2.forward(c1, edge_index);
            errdefer c2 = c2.deinit();

            return c2;
        }

        pub fn zero_grad(self: *Self) void {
            if (self.conv1.weights.grad) |_| self.conv1.weights.setup_grad(0) catch {};
            if (self.conv1.bias.grad) |_| self.conv1.bias.setup_grad(0) catch {};
            if (self.conv2.weights.grad) |_| self.conv2.weights.setup_grad(0) catch {};
            if (self.conv2.bias.grad) |_| self.conv2.bias.setup_grad(0) catch {};
        }
    };
}

pub fn GraphConvLayer(comptime T: type) type {
    return struct {
        const Self = @This();
        const Tensor = NDTensor(T);

        device: DeviceReference,

        weights: *Tensor,
        bias: *Tensor,
        adj_mat: ?*Tensor = null,

        pub fn init(device: DeviceReference, in_features: usize, out_features: usize, opts: struct {
            optim: ?zg.Optimizer = null,
        }) !Self {
            const weights = try Tensor.random(device, &.{ out_features, in_features }, .{ .kaiming = in_features }, .{
                .label = "conv_weights",
                .requires_grad = true,
                .acquired = true,
            });
            errdefer weights.deinit();

            const bias = try Tensor.zeros(device, &.{out_features}, .{
                .label = "conv_bias",
                .requires_grad = true,
                .acquired = true,
            });

            if (opts.optim) |optim| {
                try optim.attach(weights);
                try optim.attach(bias);
            }

            return .{
                .device = device,
                .weights = weights,
                .bias = bias,
            };
        }

        pub fn deinit(self: *Self) void {
            self.weights.release();
            self.weights.deinit();
            self.bias.release();
            self.bias.deinit();

            if (self.adj_mat) |adj_mat| {
                adj_mat.release();
                adj_mat.deinit();
            }
        }

        pub fn forward(self: *Self, x: *Tensor, edge_index: *NDTensor(usize)) !*Tensor {
            const shape: []const usize = &.{ x.get_dim(0), self.weights.get_dim(0) };
            const h = try x.bmm_acc(self.weights, shape, .{ .trans_b = true });
            errdefer h.deinit();
            defer {
                if (h.should_deinit()) {
                    h.deinit();
                }
            }

            const y = try self.propagate(h, edge_index);
            errdefer y.deinit();

            try y._add(self.bias);
            return y;
        }

        pub fn propagate(self: *Self, h: *Tensor, edge_index: *NDTensor(usize)) !*Tensor {
            if (self.adj_mat == null) {
                const n_node = h.get_dim(0);
                const n_edge = edge_index.get_dim(1);
                const edge_index_data = edge_index.get_data();

                var adj_mat = try Tensor.zeros(self.device, &.{ n_node, n_node }, .{
                    .label = "conv_adj_mat",
                    .acquired = true,
                });
                errdefer adj_mat.deinit();
                const adj_mat_data = adj_mat.get_data();

                const deg = try self.device.mem_scratch(T, n_node);
                self.device.mem_fill(T, deg, 1); // the initial is 1 (self loop)
                for (0..n_edge) |i| {
                    const source = edge_index_data[i * 2];
                    deg[source] += 1;
                }

                for (0..n_edge) |i| {
                    const source = edge_index_data[i * 2];
                    const target = edge_index_data[i * 2 + 1];
                    adj_mat_data[source * n_node + target] = std.math.pow(T, deg[source], -0.5);
                }
                self.adj_mat = adj_mat;
            }
            if (self.adj_mat) |a| {
                return a.bmm_acc(h, h.get_shape(), .{});
            } else unreachable;
        }
    };
}

/// this layer filter unused data base on mask
/// like pytorch out[train_mask]
pub fn MaskLayer(comptime T: type) type {
    return struct {
        const Self = @This();
        const Tensor = NDTensor(T);

        mask: *NDTensor(bool),

        pub fn forward(self: Self, x: *Tensor) !*Tensor {
            const device = x.device;
            const num_feature = x.get_dim(1);
            const buf = try device.mem_alloc(T, x.get_size());
            defer device.mem_free(buf);
            var count: usize = 0;
            for (self.mask.get_data(), 0..) |ok, i| {
                if (ok) {
                    const buf_start = count * num_feature;
                    const buf_end = count * num_feature + num_feature;
                    const x_start = i * num_feature;
                    const x_end = i * num_feature + num_feature;
                    device.mem_copy(T, x.data.data[x_start..x_end], buf[buf_start..buf_end]);
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
                .children = &.{&x.node},
                .device = device,
                .callback = self,
                .label = "mask",
                .gb = x.node.gb,
            });
        }

        pub fn backward(y: *Tensor, children: *Node.Children, self: *Self) !void {
            const device = y.device;
            const x = children.get_bwd_upcast(Tensor, 0) orelse return;
            const num_feature = x.get_dim(1);

            var x_grad = try x.ensure_grad_data(0);
            const y_grad = y.assume_grad_data();

            for (self.mask.get_data(), 0..) |ok, i| {
                const start = i * num_feature;
                const end = start + num_feature;
                if (ok) {
                    device.mem_copy(T, y_grad[start..end], x_grad[start..end]);
                }
            }
            return;
        }
    };
}
