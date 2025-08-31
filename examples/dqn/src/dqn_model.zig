const std = @import("std");

const zg = @import("zigrad");
const NDTensor = zg.NDTensor;
const DeviceReference = zg.DeviceReference;

pub fn DQNModel(
    /// Type
    comptime T: type,
    /// Number of affine layers
    comptime depth: usize,
) type {
    return struct {
        const Self = @This();
        const nn = zg.nn(T);
        const Tensor = NDTensor(T);

        weights: [depth]*Tensor = undefined,
        biases: [depth]*Tensor = undefined,
        input_size: usize,
        output_size: usize,

        pub fn init(device: DeviceReference, input_size: usize, hidden_size: usize, output_size: usize) !Self {
            var w_i: usize = 0;
            var b_i: usize = 0;

            var self: Self = .{
                .input_size = input_size,
                .output_size = output_size,
            };

            // -2 for input and output, +1 since depth == dims-1
            const layer_dims = [_]usize{input_size} ++ ([_]usize{hidden_size} ** (depth - 2 + 1)) ++ [_]usize{output_size};

            errdefer { // free everything up to last value
                for (self.weights[0..w_i]) |w|
                    w.deinit();

                for (self.biases[0..b_i]) |b|
                    b.deinit();
            }

            inline for (&self.weights, &self.biases, 0..) |*w, *b, i| {
                const in_features, const out_features = .{ layer_dims[i], layer_dims[i + 1] };

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

        /// Duplicate model
        pub fn clone(self: Self) !Self {
            var result = self;

            // Cleanup logic
            var w_i: usize = 0;
            var b_i: usize = 0;
            errdefer { // free everything up to last value
                for (self.weights[0..w_i]) |w|
                    w.deinit();

                for (self.biases[0..b_i]) |b|
                    b.deinit();
            }

            // Clone tensors
            inline for (0..self.weights.len) |i| {
                result.weights[i] = try self.weights[i].clone();
                result.weights[i].detach();
                w_i += 1;
                result.biases[i] = try self.biases[i].clone();
                result.biases[i].detach();
                b_i += 1;
            }
            return result;
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

        /// Forward pass through the DQN network
        /// Input shape: [..., input_size] or [..., flattened_features]
        /// Output shape: [..., output_size]
        pub fn forward(self: *Self, x: *Tensor) !*Tensor {
            // Flatten
            const batch_dim = x.data.shape.get(0);
            const other_dims = x.data.shape.crop(1, 0);
            const flattened_dim = zg.arrayutils.prod(other_dims);

            const flat = try x.alias();
            flat.data._reshape(&.{ batch_dim, flattened_dim });
            flat.set_label("dqn_flattened");
            errdefer flat.deinit();

            std.debug.assert(flattened_dim == self.input_size);

            // Layer 1 Forward
            const z0 = try nn.linear(flat, self.weights[0], self.biases[0]);
            errdefer z0.deinit();
            try nn.relu_(z0);

            flat.soft_deinit(); // Safe to free flattened input

            // Layer 2 Forward
            const z1 = try nn.linear(z0, self.weights[1], self.biases[1]);
            errdefer z1.deinit();
            try nn.relu_(z1);

            z0.soft_deinit(); // Safe to free layer 1 output

            // Layer 3 Forward
            const z2 = try nn.linear(z1, self.weights[2], self.biases[2]);

            z1.soft_deinit(); // Safe to free layer 2 output

            return z2;
        }

        /// Zero out gradients for all parameters
        pub fn zero_grad(self: *Self) void {
            for (&self.weights) |*w| w.*.setup_grad(0) catch {};
            for (&self.biases) |*b| b.*.setup_grad(0) catch {};
        }

        /// Attach optimizer to all parameters
        pub fn attach_optimizer(self: *Self, optim: zg.Optimizer) !void {
            for (&self.weights, &self.biases) |*w, *b| {
                try optim.attach(w.*);
                try optim.attach(b.*);
            }
        }

        /// Save model parameters to file
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

        /// Load model parameters from file
        pub fn load(path: []const u8, device: DeviceReference, input_size: usize, output_size: usize) !Self {
            const allocator = std.heap.smp_allocator;

            var params = try zg.LayerMap.load_from_file(path, allocator, device, .{
                .requires_grad = true,
                .acquired = true,
                .owning = false,
            });
            defer params.deinit();

            var model = params.extract(Self, "", .{});
            model.input_size = input_size;
            model.output_size = output_size;

            return model;
        }

        /// Soft update parameters from another model with interpolation factor tau
        /// self = tau * other + (1 - tau) * self
        pub fn soft_update_from(self: *Self, other: *const Self, tau: T) !void {
            std.debug.assert(self.input_size == other.input_size);
            std.debug.assert(self.output_size == other.output_size);
            std.debug.assert(tau >= 0.0 and tau <= 1.0);

            for (&self.weights, &self.biases, &other.weights, &other.biases) |w_dst, b_dst, w_src, b_src| {
                std.debug.assert(w_dst.device.is_compatible(w_src.device));
                std.debug.assert(b_dst.device.is_compatible(b_src.device));
                const device = w_src.device;

                // w_dst = tau * w_src + (1 - tau) * w_dst
                w_dst.data._scale(1.0 - tau, device);
                var temp_w = try w_src.data.copy(device);
                temp_w._scale(tau, device);
                try w_dst.data._add(temp_w, device);
                temp_w.deinit(device);

                // b_dst = tau * b_src + (1 - tau) * b_dst
                b_dst.data._scale(1.0 - tau, device);
                var temp_b = try b_src.data.copy(device);
                temp_b._scale(tau, device);
                try b_dst.data._add(temp_b, device);
                temp_b.deinit(device);
            }
        }

        /// Get number of trainable parameters
        pub fn parameter_count(self: *const Self) usize {
            var count: usize = 0;
            for (&self.weights, &self.biases) |w, b| {
                count += w.get_size();
                count += b.get_size();
            }
            return count;
        }

        /// Enable/disable gradient computation for all parameters
        pub fn set_requires_grad(self: *Self, requires_grad: bool) void {
            for (&self.weights, &self.biases) |*w, *b| {
                if (requires_grad) {
                    w.*.enable_grad();
                    b.*.enable_grad();
                } else {
                    w.*.disable_grad();
                    b.*.disable_grad();
                }
            }
        }
    };
}
