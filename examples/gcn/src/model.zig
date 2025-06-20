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
            if (zg.rt_grad_enabled) std.debug.assert(x.requires_grad());
            const c1 = try self.conv1.forward(x, edge_index);
            errdefer c1.deinit();
            defer {
                if (c1.should_deinit()) {
                    c1.deinit();
                }
            }
            try zg.nn.relu_(T, c1);

            if (zg.rt_grad_enabled) std.debug.assert(c1.requires_grad());

            const c2 = try self.conv2.forward(c1, edge_index);
            errdefer c2 = c2.deinit();

            if (zg.rt_grad_enabled) std.debug.assert(c2.requires_grad());

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
            if (zg.rt_grad_enabled) std.debug.assert(x.requires_grad());

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

        pub const propagate = propagate_ag;

        /// A simple, direct, implementation of propagate
        /// Due to it being direct, it directly accesses device memory and is thus not safe for use with accelerators like CUDA
        /// This implementation is also slower on some platforms for which NDArray ops dispatch to optimized vendor kernels,
        /// so despite having fewer steps it underperforms the above variant that chains many ops together.
        pub fn propagate_direct(self: *Self, h: *Tensor, edge_index: *NDTensor(usize)) !*Tensor {
            if (self.adj_mat == null) {
                const n_node = h.get_dim(0);
                const n_edge = edge_index.get_dim(1);

                var adj_mat = try Tensor.zeros(self.device, &.{ n_node, n_node }, .{
                    .label = "conv_adj_mat",
                    .acquired = true,
                });
                // errdefer adj_mat.deinit();
                const adj_mat_data = adj_mat.get_data();

                const deg = try zg.NDArray(T).scratch(&.{n_node}, self.device);
                deg.fill(1, self.device); // the initial is 1 (self loop)
                for (0..n_edge) |i| {
                    const source = edge_index.get(i * 2);
                    // NOTE: this is not safe until we support `get` and `set` or `inc` for `zg.NDArray`
                    deg.data[source] += 1;
                }

                for (0..n_edge) |i| {
                    const source = edge_index.get(i * 2);
                    const target = edge_index.get(i * 2 + 1);
                    adj_mat_data[source * n_node + target] = std.math.pow(T, deg.data[source], -0.5);
                }
                self.adj_mat = adj_mat;
            }
            if (self.adj_mat) |a| {
                return a.bmm_acc(h, h.get_shape(), .{});
            } else unreachable;
        }

        /// Another simple implementation of propagate
        /// This implementation is also slightly direct but uses the differentiable `gather` and `scatter_add` ops
        pub fn propagate_alt(self: *Self, h: *Tensor, edge_index: *NDTensor(usize)) !*Tensor {
            // NOTE: for now, for simplicity and development, we use unsafe direct data access as we are developing using only the host device
            const n_node = h.get_dim(0);
            const n_features = h.get_dim(1);
            const n_edge = edge_index.get_dim(1);
            const edge_index_data = edge_index.get_data();

            // Compute degrees for normalization
            const deg = try self.device.mem_scratch(T, n_node);
            self.device.mem_fill(T, deg, 1); // the initial is 1 (self loop)
            for (0..n_edge) |i| {
                const source = edge_index_data[i * 2];
                deg[source] += 1;
            }

            // Create normalized features for scatter
            var normalized_h = try h.clone();
            errdefer normalized_h.deinit();
            const h_data = normalized_h.get_data();
            for (0..n_node) |node| {
                const norm_factor = std.math.pow(T, deg[node], -0.5);
                for (0..n_features) |feat| {
                    h_data[node * n_features + feat] *= norm_factor;
                }
            }

            const target_indices = try self.device.mem_scratch(usize, n_edge);
            for (0..n_edge) |i| {
                target_indices[i] = edge_index_data[i * 2 + 1];
            }

            const source_indices = try zg.NDArray(usize).scratch(&.{ n_edge, 1 }, self.device);
            const source_index_data = source_indices.data;
            for (0..n_edge) |i| source_index_data[i] = edge_index_data[i * 2];
            const edge_features = try normalized_h.gather(source_indices, 0);
            const output = try edge_features.scatter_add(target_indices, &.{ n_node, n_features });
            return output;
        }

        /// An implementation of propogate that attempts to rely on Zigrad ops as much as possible
        /// This is the fastest implementation, despite building a larger computation graph.
        /// This is also a development version. It is complex and we are working to reduce the complexity
        /// by introducing more functionality to Zigrad, such as adding more ops.
        /// This should become more concise over time and eventually become the official implementation
        pub fn propagate_ag(self: *Self, h: *Tensor, edge_index: *NDTensor(usize)) !*Tensor {
            // NOTE: currently, some of the deinit lines are commented out. there is a double free, the deinit lines in this
            // method are not responsible for it but while its being tracked down they are commented.
            const n_node = h.get_dim(0);
            const n_features = h.get_dim(1);
            const n_edge = edge_index.get_dim(1);

            std.debug.assert(edge_index.get_dim(0) == 2);

            const edge_index_data = edge_index.get_data();

            // TODO: Compute deg_norm_2d with in place NDArray ops since this doesnt need to be grad tracked.
            //   Will be less verbose and way more performant anyways.
            // Compute degrees
            const deg = try Tensor.ones(self.device, &.{n_node}, .{
                .requires_grad = false,
                .label = "deg",
            }); // Start with self-loops
            defer deg.deinit();

            // Count degrees with scatter_add
            const ones_edges = try Tensor.ones(self.device, &.{n_edge}, .{
                .requires_grad = false,
                .label = "ones_edges",
            });
            defer ones_edges.deinit();

            // TODO: Not device safe!!
            const target_indices = try self.device.mem_alloc(usize, n_edge);
            defer self.device.mem_free(target_indices);
            for (0..n_edge) |i| target_indices[i] = edge_index_data[i * 2 + 1];

            // Scatter to [n_node]
            const deg_contrib = try ones_edges.scatter_add(target_indices, &.{n_node});
            defer deg_contrib.deinit();
            deg_contrib.set_label("deg_contrib");

            const total_deg = try deg_contrib.add(deg);
            defer total_deg.deinit();
            total_deg.set_label("total_deg");

            // Apply degree normalization: deg^(-0.5)
            const deg_norm = try total_deg.pow(-0.5);
            // const deg_norm = try total_deg.rsqrt(); // TODO: rsqrt
            defer deg_norm.deinit();
            deg_norm.set_label("deg_norm");

            // Reshape deg_norm to [n_node, 1] for broadcasting with h [n_node, n_features]
            const deg_norm_2d = try deg_norm.reshape(&.{ n_node, 1 });
            defer if (!zg.rt_grad_enabled) deg_norm_2d.deinit(); // required for normalized_h.grad
            deg_norm_2d.set_label("deg_norm_2d");

            // NOTE: Computation graph starts here as this si the first use of `h`
            const normalized_h = try h.mul(deg_norm_2d); // [n_node, n_features]
            defer if (!normalized_h.requires_grad()) normalized_h.deinit();
            normalized_h.set_label("normalized_h");

            // Create source indices for gathering - must match normalized_h dimensions
            const source_indices_data = try self.device.mem_alloc(usize, n_edge);
            defer self.device.mem_free(source_indices_data);

            for (0..n_edge) |i| {
                source_indices_data[i] = edge_index_data[i * 2];
            }

            // Extract edge features using index_select (preserves gradients)
            const edge_features = try normalized_h.index_select(0, source_indices_data);
            defer if (!edge_features.requires_grad()) edge_features.deinit();
            edge_features.set_label("edge_features");

            // TODO: Not device safe!!
            // Create flattened offsets for scatter_add
            const flat_offsets = try self.device.mem_scratch(usize, n_edge * n_features);
            for (0..n_edge) |edge| {
                const target_node = edge_index_data[edge * 2 + 1];
                for (0..n_features) |feat| {
                    flat_offsets[edge * n_features + feat] = target_node * n_features + feat;
                }
            }

            // Scatter features to target nodes
            const aggregated = try edge_features.scatter_add(flat_offsets, &.{ n_node, n_features });
            defer if (!aggregated.requires_grad()) aggregated.deinit();
            aggregated.set_label("aggregated");

            // Add self-connections (normalized_h represents self-loops)
            try aggregated._add(normalized_h);
            const output = aggregated;

            // Or,
            // const output = try aggregated.add(normalized_h);
            // output.set_label("output");

            return output;
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
                .label = "mask_fwd",
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
