const std = @import("std");
const zg = @import("zigrad");
const opspec = zg.opspec;
const DeviceReference = zg.DeviceReference;
const Graph = zg.Graph;
const Node = Graph.Node;
const NDTensor = zg.NDTensor;

pub fn MnistModel(comptime T: type) type {
    return struct {
        const nn = zg.nn(T);
        const Self = @This();
        const Tensor = NDTensor(T);
        const dims = [_]usize{ 784, 128, 64, 10 };
        const depth = dims.len - 1;

        weights: [depth]*Tensor = undefined,
        biases: [depth]*Tensor = undefined,

        pub fn init(device: DeviceReference) !Self {
            var w_i: usize = 0;
            var b_i: usize = 0;

            var self: Self = .{};

            errdefer { // free everything up to last value
                for (self.weights[0..w_i]) |w|
                    w.deinit();

                for (self.biases[0..b_i]) |b|
                    b.deinit();
            }

            inline for (&self.weights, &self.biases, 0..) |*w, *b, i| {
                const in_features, const out_features = .{ dims[i], dims[i + 1] };

                w.* = try Tensor.random(
                    device,
                    &.{ out_features, in_features },
                    .{ .kaiming = in_features },
                    .{
                        .label = std.fmt.comptimePrint("weights.{d}", .{i}),
                        .requires_grad = true,
                        .acquired = true,
                    },
                );
                w_i += 1;

                b.* = try Tensor.zeros(
                    device,
                    &.{out_features},
                    .{
                        .label = std.fmt.comptimePrint("biases.{d}", .{i}),
                        .requires_grad = true,
                        .acquired = true,
                    },
                );
                b_i += 1;
            }
            return self;
        }

        pub fn deinit(self: *Self) void {
            for (&self.weights, &self.biases) |w, b| {
                w.release();
                w.deinit();
                b.release();
                b.deinit();
            }
            self.* = undefined;
        }

        pub fn forward(self: *Self, x: *Tensor) !*Tensor {
            const batch_dim = x.data.shape.get(0);
            const other_dims = x.data.shape.crop(1, 0);
            const flattened_dim = zg.arrayutils.prod(other_dims);
            const flat = try x.alias();
            flat.data._reshape(&.{ batch_dim, flattened_dim });
            flat.set_label("flattened");
            errdefer flat.deinit();

            const z0 = try nn.linear(flat, self.weights[0], self.biases[0]);
            errdefer z0.deinit();
            try nn.relu_(z0);

            flat.soft_deinit();

            const z1 = try nn.linear(z0, self.weights[1], self.biases[1]);
            errdefer z1.deinit();
            try nn.relu_(z1);

            z0.soft_deinit();

            const z2 = try nn.linear(z1, self.weights[2], self.biases[2]);
            z1.soft_deinit();

            return z2;
        }

        pub fn zero_grad(self: *Self) void {
            for (&self.weights) |*w| w.*.setup_grad(0) catch {};
            for (&self.biases) |*b| b.*.setup_grad(0) catch {};
        }

        pub fn attach_optimizer(self: *Self, optim: zg.Optimizer) !void {
            for (&self.weights, &self.biases) |*w, *b| {
                try optim.attach(w.*);
                try optim.attach(b.*);
            }
        }

        pub fn save(self: *Self, path: []const u8) !void {
            const allocator = std.heap.smp_allocator;
            var params = zg.LayerMap.init(allocator);
            defer params.deinit();
            for (&self.weights, &self.biases) |w, b| {
                try params.put(w.get_label().?, w, .{});
                try params.put(b.get_label().?, b, .{});
            }
            try params.save_to_file(path, allocator);
        }

        pub fn load(path: []const u8, device: DeviceReference) !Self {
            const allocator = std.heap.smp_allocator;

            var params = try zg.LayerMap.load_from_file(path, allocator, device, .{
                .requires_grad = true,
                .acquired = true,
                .owning = false,
            });
            defer params.deinit();

            return params.extract(Self, "", .{});
        }
    };
}
