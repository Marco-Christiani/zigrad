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

            try zg.nn.relu_(T, c1);

            const c2 = try self.conv2.forward(c1, edge_index);
            errdefer c2 = c2.deinit();
            c1.soft_deinit();

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
            const h = try x.bmm(self.weights, .{ .trans_b = true });
            errdefer h.deinit();
            std.debug.assert(std.mem.eql(usize, &.{ x.get_dim(0), self.weights.get_dim(0) }, h.get_shape()));

            const y = try self.propagate(h, edge_index);
            errdefer y.deinit();

            defer h.soft_deinit();

            try y._add(self.bias);
            return y;
        }

        pub const propagate = propagate_scatter_gcn_deg_scaled;

        pub fn propagate_scatter_gcn_deg_scaled(self: *Self, h: *Tensor, edge_index: *NDTensor(usize)) !*Tensor {
            const n_node = h.get_dim(0);
            const n_features = h.get_dim(1);
            // std.debug.assert(edge_index.get_dim(1) == 2); // standard layout
            std.debug.assert(edge_index.get_dim(0) == 2); // optimized layout
            // const n_conn = edge_index.get_size() / 2; // in both cases, this holds
            const n_conn = edge_index.get_dim(1); // optimized layout

            // Two possible layouts for edge_index
            // Standard Representation:
            // [
            //   [src_0, tgt_0],
            //   [src_1, tgt_1],
            //   [src_2, tgt_2],
            // ]
            //
            // Data layout - interleaved:
            // [src_0, tgt_0, src_1, tgt_1, src_2, tgt_2]
            //
            // To construct 1D views for src and tgt notice the data interleaved,
            // so our views necessarily non-contiguous. In terms of the data layout
            // this corresponds to strides={2}. For tgt, we need to shift the base
            // pointer by an offset.
            //
            // View layout:
            // src = edge_index[:, 0]
            // -> ptr = edge_index.ptr (i.e. &edge_index[0])
            //    offset = 0
            //    strides = {2}
            //
            // tgt = edge_index[:, 1]
            // -> ptr = edge_index.ptr (i.e. &edge_index[0])
            //    offset = 1
            //    strides = {2}
            //
            // Generalized access semantics:
            // Consider $v \in {src, tgt}$ and
            // let v_i be the value for the i'th source/target and v[k] be the k'th element of v following contiguous
            //   indexing semantics, i.e. in pseudocode &v[k] == v.ptr + k
            //
            // Then,
            //   v_i = (v.ptr + offset)[i * 2]
            //
            // This is, of course, indexing semantics generalized with the addition of an offset and relaxing the
            //   assumption of row contiguity.
            //
            //  ---
            //
            // Optimized Representation:
            // [
            //   [src_0, src_1, src_2],
            //   [tgt_0, tgt_1, tgt_2],
            // ]
            //
            // Data layout:
            // [src_0, src_1, src_2, tgt_0, tgt_1, tgt_2]
            //
            // 1D views for src and tgt are now contiguous slices.
            // I.e.,
            // src = edge_index[0..3]
            // tgt = edge_index[3..6]
            //
            // View layout:
            // src = edge_index[0, :]
            // -> ptr = edge_index.ptr (i.e. &edge_index[0])
            //    offset = 0
            //    strides = {1}
            //
            // tgt = edge_index[1, :]
            // -> ptr = edge_index.ptr
            //    offset = 3
            //    strides = {1}
            // tgt's first element is thus at ptr + offset == &edge_index[3]

            // Standard storage layout
            // NOTE: this is not free'd as its an operand for our scatter_gcn_deg_scaled grad tracked op
            // which doesnt explicitly save (copy) for backward.
            // const src_indices = try self.device.mem_alloc(usize, n_edge);
            // const tgt_indices = try self.device.mem_alloc(usize, n_edge);

            // TODO: make device safe, if not just add a transfer
            // const edge_index_data = edge_index.get_data();
            // for (0..n_edge) |i| {
            //     src_indices[i] = edge_index_data[0 + i * 2]; // [0, i]
            //     tgt_indices[i] = edge_index_data[1 + i * 2]; // [1, i]
            // }

            // Optimized layout
            const src_indices = edge_index.data.data.raw[0..n_conn];
            const tgt_indices = edge_index.data.data.raw[n_conn..];

            const deg = try Tensor.ones(self.device, &.{n_node}, .{ .requires_grad = false });
            defer deg.deinit();

            const ones_edges = try Tensor.ones(self.device, &.{n_conn}, .{ .requires_grad = false });

            const deg_contrib = try ones_edges.scatter_add(tgt_indices, &.{n_node});
            ones_edges.deinit();

            try deg._add(deg_contrib);
            deg_contrib.deinit();

            const deg_norm = try deg.rsqrt(); // shape: [n_node]
            defer deg_norm.soft_deinit();

            if (zg.runtime.grad_enabled) deg_norm.enable_grad();

            return try self.scatter_gcn_deg_scaled(h, deg_norm, src_indices, tgt_indices, n_node, n_features, n_conn);
        }

        /// Differentiable gcn propagate with degree scaling
        pub fn scatter_gcn_deg_scaled(
            _: *const Self,
            h: *Tensor,
            deg: *Tensor,
            src_indices: []const usize,
            tgt_indices: []const usize,
            n_node: usize,
            stride: usize,
            n_edge: usize,
        ) !*Tensor {
            const output = try Tensor.DataType.zeros(&.{ n_node, stride }, h.device);

            // Forward kernel
            h.device.dispatch(opspec.scatter_gcn_deg_scaled(T){
                .dst = output.get_data(),
                .h = h.get_data(),
                .deg = deg.get_data(),
                .src_indices = src_indices,
                .tgt_indices = tgt_indices,
                .stride = stride,
                .n_edge = n_edge,
            });

            const Bwd = struct {
                src_indices: []const usize,
                tgt_indices: []const usize,
                stride: usize,
                n_edge: usize,

                pub fn backward(y: *Tensor, children: *Node.Children, ctx: *@This()) !void {
                    const h_ = children.get_bwd_upcast(Tensor, 0) orelse return;
                    const deg_ = children.get_bwd_upcast(Tensor, 1) orelse return;

                    const grad_output = y.assume_grad_data();
                    const grad_h = try h_.ensure_grad_data(0);
                    const grad_deg = try deg_.ensure_grad_data(0);

                    y.device.dispatch(opspec.scatter_gcn_deg_scaled_bwd(T){
                        .grad_output = grad_output,
                        .h = h_.get_data(),
                        .deg = deg_.get_data(),
                        .src_indices = ctx.src_indices,
                        .tgt_indices = ctx.tgt_indices,
                        .grad_h = grad_h,
                        .grad_deg = grad_deg,
                        .stride = ctx.stride,
                        .n_edge = ctx.n_edge,
                    });
                }
            };

            return try Tensor.create_dependent(Bwd, .{
                .data = output,
                .children = &.{ &h.node, &deg.node },
                .device = h.device,
                .gb = h.node.gb,
                .callback = .{
                    .src_indices = src_indices,
                    .tgt_indices = tgt_indices,
                    .stride = stride,
                    .n_edge = n_edge,
                },
            });
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
                    device.mem_copy(T, x.get_data()[x_start..x_end], buf[buf_start..buf_end]);
                    count += 1;
                }
            }
            const y = try Tensor.DataType.from_slice(
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
