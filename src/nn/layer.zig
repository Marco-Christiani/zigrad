const std = @import("std");
const zg = @import("../zigrad.zig");
const DeviceReference = zg.DeviceReference;

const NDTensor = zg.NDTensor;
const GraphManager = zg.GraphManager;
const SGD = zg.optim.SGD;

const Shape = zg.Shape;
const NDArray = zg.NDArray;

const prod = zg.arrayutils.prod;
const conv_utils = zg.conv_utils;
const simple_mse_loss = zg.loss.ag_mse_1d;
const winit = zg.winit;

const log = std.log.scoped(.zg_layer);

pub fn Layer(comptime T: type) type {
    return struct {
        const Self = @This();

        vtable: *const VTable,
        ptr: *anyopaque,

        pub const VTable = struct {
            forward: *const fn (ctx: *anyopaque, input: *NDTensor(T)) anyerror!*NDTensor(T),
            get_parameters: *const fn (ctx: *anyopaque) ?[]*NDTensor(T),
            zero_grad: *const fn (ctx: *anyopaque) void,
            deinit: *const fn (ctx: *anyopaque) void,
            release: *const fn (ctx: *anyopaque) void,
        };

        pub fn init(pointer: anytype) Self {
            const Ptr = @TypeOf(pointer);

            const gen = struct {
                fn forward_fn(ctx: *anyopaque, input: *NDTensor(T)) !*NDTensor(T) {
                    const self: Ptr = @ptrCast(@alignCast(ctx));
                    return self.forward(input);
                }

                fn get_parametersFn(ctx: *anyopaque) ?[]*NDTensor(T) {
                    const self: Ptr = @ptrCast(@alignCast(ctx));
                    return self.get_parameters();
                }

                fn zero_gradFn(ctx: *anyopaque) void {
                    const self: Ptr = @ptrCast(@alignCast(ctx));
                    self.zero_grad();
                }

                fn deinit_fn(ctx: *anyopaque) void {
                    const self: Ptr = @ptrCast(@alignCast(ctx));
                    self.deinit();
                }

                fn release_fn(ctx: *anyopaque) void {
                    const self: Ptr = @ptrCast(@alignCast(ctx));
                    self.release();
                }

                const vtable = VTable{
                    .forward = forward_fn,
                    .get_parameters = get_parametersFn,
                    .zero_grad = zero_gradFn,
                    .deinit = deinit_fn,
                    .release = release_fn,
                };
            };

            return .{
                .vtable = &gen.vtable,
                .ptr = pointer,
            };
        }

        pub fn forward(self: Self, input: *NDTensor(T)) !*NDTensor(T) {
            return self.vtable.forward(self.ptr, input);
        }

        /// Note that layer may not have any trainables
        pub fn get_parameters(self: Self) ?[]*NDTensor(T) {
            return self.vtable.get_parameters(self.ptr);
        }

        pub fn zero_grad(self: Self) void {
            self.vtable.zero_grad(self.ptr);
        }

        pub fn deinit(self: Self) void {
            self.vtable.deinit(self.ptr);
        }

        pub fn release(self: Self) void {
            self.vtable.release(self.ptr);
        }

        pub fn enable_grad(self: Self) void {
            self._set_grad(true);
        }

        pub fn disable_grad(self: Self) void {
            self._set_grad(false);
        }

        fn _set_grad(self: Self, enabled: bool) void {
            if (self.get_parameters()) |params| {
                for (params) |param| {
                    param.requires_gradient = enabled; // NOTE: could possibly destroy grad here depending on aggressiveness
                }
            }
        }
    };
}

pub fn LinearLayer(comptime T: type) type {
    return struct {
        const Self = @This();
        weights: *NDTensor(T),
        bias: *NDTensor(T),
        device: DeviceReference,
        parameters: [2]*NDTensor(T),

        pub fn init(device: DeviceReference, in_features: usize, out_features: usize) !*Self {
            const weights_shape = [_]usize{ out_features, in_features };
            const bias_shape = [_]usize{ 1, out_features };
            var weights = try NDTensor(T).empty(&weights_shape, true, device);
            try weights.set_label("linear_weights");
            var bias = try NDTensor(T).empty(&bias_shape, true, device);
            try bias.set_label("linear_bias");
            weights.acquire();
            bias.acquire();

            winit.he_init(T, weights);
            bias.fill(0.0);

            const self = try device.allocator.create(Self);
            self.* = Self{
                .weights = weights,
                .bias = bias,
                .parameters = [_]*NDTensor(T){ weights, bias },
                .device = device,
            };
            return self;
        }
        pub fn forward(self: *Self, input: *NDTensor(T)) !*NDTensor(T) {
            // return try forward_ag(self, input, fwd_device);
            return try forward_manual(self, input);
        }

        // Autograd version with much the performance of the optimized one, but requires an unbroadcast that Zigrad handles
        // in the backward for broadcasted bias addition thats not possible if its not tracked in the op graph so theres a
        // slightly more optimized variant that explicitly handles the broadcast and unbroadcast.
        pub fn forward_ag(self: *const Self, input: *const NDTensor(T)) !*NDTensor(T) {
            // Labels are optional, useful if you want to render a diagram of the computational graph.
            // (W@X^T + b)^T == X@W^T + b
            return (try (try input.bmm(
                self.weights,
                .{ .trans_b = true },
            )).set_label("lin_fwd1").add(self.bias)).set_label("lin_fwd2");
        }

        // Implement the forward and backward passes manually. The only benefit to this is not paying the cost of
        // Zigrad figuring out the broadcast and unbroadcasting logic on the fly. This should highlight how little
        // overhead Zigrad's abstractions are
        pub fn forward_manual(self: *Self, input: *NDTensor(T)) !*NDTensor(T) {
            const batch_size = if (input.data.shape.len() > 1) try input.data.shape.get(0) else 1;
            const out_features = self.weights.data.shape.shape[0];

            var result_nd = try NDArray(T).empty(&[_]usize{ batch_size, out_features }, self.device);
            const bd = self.bias.data.data;
            for (0..batch_size) |i| {
                @memcpy(result_nd.data[i * out_features .. (i + 1) * out_features], bd);
            }

            _ = try input.data._bmm_acc(
                self.weights.data,
                result_nd,
                1.0,
                1.0,
                false,
                true,
                self.device,
            );
            // Hook up the custom backward function.
            return try NDTensor(T).create_dependent(.{
                .data = result_nd,
                .label = "lin_fwdman",
                .children = &.{ input, self.weights, self.bias },
                .requires_gradient = input.requires_gradient or self.weights.requires_gradient or self.bias.requires_gradient,
                .device = self.device,
                ._backward = backward_manual,
                ._backward_ctx = self,
            });
        }

        /// A custom backward impl that, technically, isnt optimized in any way aside from avoiding having
        /// to infer shape broadcasting logic which is shockingly fast in Zigrad for some reason.
        fn backward_manual(tensor: NDTensor(T)) !void {
            const self: *Self = @ptrCast(@alignCast(tensor._backward_ctx orelse return error.NoBackwardContext));
            const grad_output = tensor.grad orelse return error.NoGradient;
            const input = tensor.get_children().?[0];

            const grad_W = self.weights.grad orelse return error.NoGradient;
            const grad_B = self.bias.grad orelse return error.NoGradient;
            const grad_input = input.grad orelse return error.NoGradient;

            // weights grad
            _ = try grad_output._bmm_acc(
                input.data,
                grad_W,
                1.0,
                1.0,
                true, // grad_output^T * input
                false,
                self.device,
            );
            // similar performance to the below optimized variant and simpler. essentially the exact thing that autograd does.
            const bias_grad = try grad_output.sum_along(self.device, .{ .dim = 0 });
            _ = try grad_B._add(bias_grad);
            // This is not really a great way to optimize this, btw. This is meant to demonstrate to a user how they can
            // implement custom functionality by accessing the underlying data directly.
            // const blas = @import("../backend/blas.zig");
            // const batch_size = grad_output.shape.shape[0];
            // const out_features = grad_output.shape.shape[1];
            // const grad_B_data = grad_B.data;
            // const grad_output_data = grad_output.data;
            // // grad_B with the first batch
            // blas.blas_axpy(T, out_features, 1.0, grad_output_data, 1, grad_B_data, 1);
            // // sum over the remaining batches
            // for (1..batch_size) |i| {
            //     blas.blas_axpy(T, out_features, 1.0, grad_output_data[i * out_features ..], 1, grad_B_data, 1);
            // }

            // input gradient
            _ = try grad_output._bmm_acc(
                self.weights.data,
                grad_input,
                1.0,
                1.0,
                false, // grad_output * weights^T
                false,
                self.device,
            );
        }

        pub fn get_parameters(self: *Self) ?[]*NDTensor(T) {
            return &self.parameters;
        }

        pub fn zero_grad(self: *Self) void {
            self.weights.grad.?.fill(0, self.device);
            self.bias.grad.?.fill(0, self.device);
        }

        pub fn deinit(self: *Self) void {
            self.release();
            self.weights.deinit();
            self.bias.deinit();
            self.device.allocator.destroy(self);
        }

        pub fn release(self: *Self) void {
            self.weights.release();
            self.bias.release();
        }

        pub fn as_layer(self: *Self) Layer(T) {
            return Layer(T).init(self);
        }
    };
}

/// Unoptimized in every way, this was for proof of concept. TODO: optimize Conv2DLayer
pub fn Conv2DLayer(comptime T: type) type {
    return struct {
        const Self = @This();

        weights: *NDTensor(T),
        bias: *NDTensor(T),
        device: DeviceReference,
        parameters: [2]*NDTensor(T),
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        input: ?NDTensor(T), // backward ctx, TBD if we should go back to this for static graph support later

        pub fn init(
            device: DeviceReference,
            in_channels: usize,
            out_channels: usize,
            kernel_size: usize,
            stride: usize,
            padding: usize,
            dilation: usize,
        ) !*Self {
            const weights_shape = [_]usize{ out_channels, in_channels, kernel_size, kernel_size };
            const bias_shape = [_]usize{out_channels};
            var weights = try NDTensor(T).empty(&weights_shape, true, device);
            try weights.set_label("conv2d_weights");
            var bias = try NDTensor(T).empty(&bias_shape, true, device);
            try bias.set_label("conv2d_bias");
            weights.acquire();
            bias.acquire();
            winit.he_init(T, weights);
            bias.fill(0.0);
            const self = try device.allocator.create(Self);

            self.* = .{
                .weights = weights,
                .bias = bias,
                .device = device,
                .parameters = [_]*NDTensor(T){ weights, bias },
                .in_channels = in_channels,
                .out_channels = out_channels,
                .kernel_size = kernel_size,
                .stride = stride,
                .padding = padding,
                .dilation = dilation,
                .input = null,
            };
            return self;
        }

        pub fn forward(self: *Self, input: *NDTensor(T)) !*NDTensor(T) {
            const batch_size = try input.data.shape.get(0);
            const in_height = try input.data.shape.get(2);
            const in_width = try input.data.shape.get(3);
            const out_height = (in_height + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1;
            const out_width = (in_width + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1;

            const col: *NDArray(T) = try conv_utils.im2col(T, input.data.*, self.kernel_size, self.stride, self.padding, self.dilation);
            defer col.deinit();

            try self.weights.data._reshape(&.{ self.out_channels, self.in_channels * self.kernel_size * self.kernel_size });

            // broadcasted mm for batching
            var output = try self.weights.data.bmm(col, false, false);

            // reshape to 4D
            try output._reshape(&.{ batch_size, self.out_channels, out_height, out_width });

            // bias
            try self.bias.data._reshape(&.{ 1, self.out_channels, 1, 1 });
            _ = try output._add(self.bias.data);

            defer {
                // restore weights shape after forward (TODO: this can be optimized out later?)
                self.weights.data._reshape(&.{
                    self.out_channels,
                    self.in_channels,
                    self.kernel_size,
                    self.kernel_size,
                }) catch std.debug.panic("Weights reshape failed.", .{});

                self.bias.data._reshape(&[_]usize{
                    self.out_channels,
                }) catch std.debug.panic("Bias reshape failed.", .{});
            }
            return try NDTensor(T).create_dependent(.{
                .data = output,
                .children = &[_]*const NDTensor(T){input},
                .label = "conv2d_out",
                .requires_gradient = true,
                .device = self.device,
                ._backward = backward,
                ._backward_ctx = self,
            });
        }

        pub fn backward(tensor: NDTensor(T)) anyerror!void {
            const self: *Self = @ptrCast(@alignCast(tensor._backward_ctx orelse return error.NoBackwardContext));
            std.debug.assert(tensor.children.?.len == 1);
            const input = tensor.children.?[0];
            const batch_size = try input.data.shape.get(0);
            const out_channels = self.out_channels;
            const out_height = try tensor.data.shape.get(2);
            const out_width = try tensor.data.shape.get(3);

            // copy for reshaping
            var grad_output_copy = try tensor.grad.?.copy(tensor.device);
            defer grad_output_copy.deinit(tensor.device);
            try grad_output_copy._reshape(&.{ batch_size * out_channels, out_height * out_width });

            // w.r.t. weights
            const col = try conv_utils.im2col(T, input.data.*, self.kernel_size, self.stride, self.padding, self.dilation);
            defer col.deinit();

            try grad_output_copy._reshape(&[_]usize{ out_channels, batch_size * out_height * out_width });
            try col._reshape(&[_]usize{ batch_size * out_height * out_width, self.in_channels * self.kernel_size * self.kernel_size });

            const grad_weights = try grad_output_copy.bmm(col, false, false);
            defer grad_weights.deinit();

            // grad_weights should be (out_channels, in_channels * kernel_size * kernel_size)
            try grad_weights._reshape(&[_]usize{
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            });
            _ = try self.weights.grad.?._add(grad_weights);

            // w.r.t. bias
            const bias_grad = try tensor.grad.?.sum(tensor.device);
            defer bias_grad.deinit(tensor.device);
            _ = try self.bias.grad.?._add(bias_grad);

            // w.r.t. input
            var weights_copy = try self.weights.data.copy(self.device);
            defer weights_copy.deinit(self.device);
            try weights_copy._reshape(&[_]usize{ out_channels, self.in_channels * self.kernel_size * self.kernel_size });

            const grad_col = try weights_copy.bmm(grad_output_copy, true, false, self.device);
            defer grad_col.deinit(self.device);

            const grad_input = try conv_utils.col2im(
                T,
                grad_col.*,
                input.data.shape.shape,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
            );
            defer grad_input.deinit(self.device);

            if (input.grad) |input_grad| {
                _ = try input_grad._add(grad_input);
            } else {
                @panic("grad didnt exist in backward");
            }
        }

        fn get_parameters(self: *Self) ?[]*NDTensor(T) {
            return &self.parameters;
        }

        pub fn zero_grad(self: *Self) void {
            self.weights.grad.?.fill(0, self.device);
            self.bias.grad.?.fill(0, self.device);
        }

        pub fn deinit(self: *Self) void {
            self.release();
            self.weights.deinit();
            self.bias.deinit();
            // self.input should not be freed be us unless we choose to copy for backward later
            // if (self.input) |input| input.deinit(allocator);
            self.device.allocator.destroy(self);
        }

        pub fn release(self: *Self) void {
            self.weights.release();
            self.bias.release();
        }

        pub fn as_layer(self: *Self) Layer(T) {
            return Layer(T).init(self);
        }
    };
}

// TODO: yes, there is no reason for this to be a layer, need to add support to the Layer interface for fromStatelessFunc kinda thing
// or, since we have eager dynamic graph design, write your forward in pytorch style
pub fn ReLULayer(comptime T: type) type {
    return struct {
        const Self = @This();
        device: DeviceReference,

        pub fn init(device: DeviceReference) !*Self {
            const self = try device.allocator.create(Self);
            self.* = Self{
                .device = device,
            };
            return self;
        }

        pub fn forward(self: *Self, input: *NDTensor(T)) !*NDTensor(T) {
            const output = try NDTensor(T).from_cache(input.get_shape(), input.requires_grad(), input.device);

            for (input.data, output.data) |x, *y| {
                y.* = if (x > 0) x else 0;
            }

            return try NDTensor(T).create_dependent(.{
                .data = output,
                .children = &.{input},
                .label = "relu_out",
                .requires_gradient = input.requires_grad(),
                .device = self.device,
                ._backward = backward,
            });
        }

        fn backward(tensor: NDTensor(T)) !void {
            const children = tensor.get_children() orelse return error.NoChildren;
            std.debug.assert(children.len == 1);
            const grad_t = tensor.grad orelse return error.NoGradient;
            const input = children[0];
            for (input.grad.?.data, input.data.data, grad_t.data) |*grad_in, value_in, grad_out| {
                grad_in.* += if (value_in > 0) grad_out else 0;
            }
        }

        pub fn get_parameters(_: *Self) ?[]*NDTensor(T) {
            return null;
        }

        pub fn zero_grad(_: *Self) void {}

        pub fn deinit(self: *Self) void {
            self.device.allocator.destroy(self);
        }

        pub fn release(_: *Self) void {}

        pub fn as_layer(self: *Self) Layer(T) {
            return Layer(T).init(self);
        }
    };
}

pub fn ReLULayerCompare(comptime T: type) type {
    return struct {
        const Self = @This();

        pub fn forward(_: Self, input: *NDTensor(T)) !*NDTensor(T) {
            const output = input.device.cache.get(NDTensor(T), input.get_shape()) orelse try input.clone();

            for (input.get_data(), output.get_data()) |*x, *y| {
                y.* = if (x.* > 0) x.* else 0;
            }

            // these probably shouldn't be on the heap.
            output.label = null;

            if (input.requires_grad()) {
                if (output.grad) |g| {
                    g.fill(0, input.device);
                } else {
                    output.grad = try NDArray(T).zeros(output.get_shape(), output.device);
                }
                output.requires_gradient = true;
            }

            if (output.children) |c| {
                c[0] = input;
            } else {
                output.children = try input.device.allocator.dupe(*NDTensor(T), &.{input});
            }

            output.acquired = false;
            output._backward = backward;
            output._backward_ctx = null;

            return output;
        }

        fn backward(tensor: NDTensor(T)) !void {
            std.debug.assert(tensor.children.?.len == 1);
            const children = tensor.children orelse return error.NoChildren;
            const grad_t = tensor.grad orelse return error.NoGradient;
            const input = children[0];
            for (input.grad.?.data, input.data.data, grad_t.data) |*grad_in, value_in, grad_out| {
                grad_in.* += if (value_in > 0) grad_out else 0;
            }
        }
    };
}

pub fn FlattenLayer(comptime T: type) type {
    return struct {
        const Self = @This();
        device: DeviceReference,

        pub fn init(device: DeviceReference) !*Self {
            const self = try device.allocator.create(Self);
            self.* = .{ .device = device };
            return self;
        }

        pub fn forward(self: *Self, input: *NDTensor(T)) !*NDTensor(T) {
            const batch_dim = input.data.shape.shape[0];
            const other_dims = input.data.shape.shape[1..];
            const flattened_dim = prod(other_dims);

            // view of input tensor with new shape
            const result = try NDTensor(T).create_dependent(.{
                .data = input.data, // Reuse the same data
                // .op = .FLATTEN,
                .children = &.{input},
                .label = "flattened",
                .requires_gradient = input.requires_gradient,
                .device = self.device,
                ._backward = backward,
            });

            const new_shape = &.{ batch_dim, flattened_dim };
            try result.data._reshape(new_shape);
            if (result.grad) |g| try g._reshape(new_shape);

            return result;
        }

        fn backward(tensor: NDTensor(T)) !void {
            var input = tensor.get_children().?[0];
            try input.grad.?._reshape(input.data.shape.shape);
            @memcpy(input.grad.?.data, tensor.grad.?.data);
        }

        pub fn get_parameters(_: *Self) ?[]*NDTensor(T) {
            return null;
        }

        pub fn zero_grad(_: *Self) void {}

        pub fn deinit(self: *Self) void {
            self.device.allocator.destroy(self);
        }

        pub fn release(_: *Self) void {}

        pub fn as_layer(self: *Self) Layer(T) {
            return Layer(T).init(self);
        }
    };
}

// TODO: MaxPool2D, this was a very fast thrown together impl to test something. Hard on cache, its just bad.
pub fn MaxPool2DLayer(comptime T: type) type {
    return struct {
        const Self = @This();

        kernel_size: usize,
        stride: usize,
        padding: usize,
        device: DeviceReference,

        pub fn init(device: DeviceReference, kernel_size: usize, stride: usize, padding: usize) !*Self {
            const self = try device.allocator.create(Self);
            self.* = .{
                .kernel_size = kernel_size,
                .stride = stride,
                .padding = padding,
                .device = device,
            };
            return self;
        }

        pub fn forward(self: *Self, input: *NDTensor(T)) !*NDTensor(T) {
            const batch_size = try input.data.shape.get(0);
            const channels = try input.data.shape.get(1);
            const in_height = try input.data.shape.get(2);
            const in_width = try input.data.shape.get(3);

            const out_height = (in_height + 2 * self.padding - self.kernel_size) / self.stride + 1;
            const out_width = (in_width + 2 * self.padding - self.kernel_size) / self.stride + 1;

            var output = try NDArray(T).empty(&.{ batch_size, channels, out_height, out_width });
            var indices = try NDArray(usize).empty(&.{ batch_size, channels, out_height, out_width });

            for (0..batch_size) |b| {
                for (0..channels) |c| {
                    for (0..out_height) |h| {
                        for (0..out_width) |w| {
                            var max_val: T = -std.math.inf(T);
                            var max_index: usize = 0;

                            for (0..self.kernel_size) |kh| {
                                for (0..self.kernel_size) |kw| {
                                    const h_in = h * self.stride + kh - self.padding;
                                    const w_in = w * self.stride + kw - self.padding;

                                    if (h_in < 0 or h_in >= in_height or w_in < 0 or w_in >= in_width) {
                                        continue;
                                    }

                                    const index = b * channels * in_height * in_width +
                                        c * in_height * in_width +
                                        h_in * in_width +
                                        w_in;
                                    const val = input.data.data[index];

                                    if (val > max_val) {
                                        max_val = val;
                                        max_index = index;
                                    }
                                }
                            }

                            const out_index = b * channels * out_height * out_width +
                                c * out_height * out_width +
                                h * out_width +
                                w;
                            output.data[out_index] = max_val;
                            indices.data[out_index] = max_index;
                        }
                    }
                }
            }

            return try NDTensor(T).create_dependent(.{
                .data = output,
                .children = &.{input},
                .label = "maxpool2d_out",
                .requires_gradient = true,
                .allocator = self.device,
                ._backward = backward,
                ._backward_ctx = indices,
            });
        }

        pub fn backward(tensor: NDTensor(T)) anyerror!void {
            const indices = @as(*NDArray(usize), @ptrCast(@alignCast(tensor._backward_ctx orelse return error.NoBackwardContext)));
            std.debug.assert(tensor.children.?.len == 1);
            var input = tensor.children.?[0];

            for (0..tensor.grad.?.data.len) |i| {
                const index = indices.data[i];
                input.grad.?.data[index] += tensor.grad.?.data[i];
            }

            indices.deinit(tensor.device);
        }

        pub fn get_parameters(_: *Self) ?[]*NDTensor(T) {
            return null;
        }

        pub fn zero_grad(_: *Self) void {}

        pub fn deinit(self: *Self) void {
            self.device.allocator.destroy(self);
        }

        pub fn release(_: *Self) void {}

        pub fn as_layer(self: *Self) Layer(T) {
            return Layer(T).init(self);
        }
    };
}

pub fn read_csv(comptime T: type, file_path: []const u8, alloc: std.mem.Allocator) ![][]T {
    var file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    var reader = file.reader();
    const bytes = try reader.readAllAlloc(alloc, 1 << 32);
    var line_iter = std.mem.tokenizeScalar(u8, bytes, '\n');

    var arr = std.ArrayList([]T).init(alloc);
    var row_arr = std.ArrayList(T).init(alloc);

    while (line_iter.next()) |line| {
        var row_iter = std.mem.tokenizeScalar(u8, line, ',');
        while (row_iter.next()) |value| {
            try row_arr.append(try std.fmt.parseFloat(T, value));
        }
        try arr.append(try row_arr.toOwnedSlice());
    }

    return try arr.toOwnedSlice();
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const T = f32;

    std.debug.print("Reading data\n", .{});
    const data = try read_csv(T, "/tmp/data.csv", alloc);
    std.debug.print("data.len={}\n", .{data.len});
    try train_grad_accum(T, data, alloc);
}

/// An old demo, still works of course but there are way easier and faster ways to do this now.
fn train_grad_accum(comptime T: type, data: [][]T, alloc: std.mem.Allocator) !void {
    const Tensor = NDTensor(T);
    var cpu = zg.device.HostDevice.init(alloc);
    defer cpu.deinit();
    const device = cpu.reference();

    const input_size = 2;
    const output_size = 1;

    const layer = try LinearLayer(T).init(device, input_size, output_size);
    defer layer.deinit();
    const params = layer.get_parameters();
    defer if (params) |p| device.mem_free(p);
    for (params.?) |p| {
        p.fill(1);
    }

    var gm = GraphManager(NDTensor(T)).init(device.allocator, .{});
    defer gm.deinit();
    var optimizer = SGD(T){ .lr = 0.01, .grad_clip_enabled = true };
    const lr_epoch_decay = 1.0; // disable

    const grad_acc_steps = 32;
    var loss: *Tensor = try Tensor.init(&[_]T{0.0}, null, true, device);
    loss.acquire();
    defer loss.teardown();
    defer loss.release();
    // used to avoid reallocating
    const input = try Tensor.empty(&.{ 1, input_size }, true, device);
    try input.set_label("input");
    input.fill(0);
    const target = try Tensor.empty(&.{output_size}, true, device);
    try target.set_label("input");
    target.fill(0);
    input.acquire();
    target.acquire();
    defer input.release();
    defer target.release();
    zg.rt_grad_enabled = true;

    for (0..15) |epoch| {
        var start_i: usize = 0;
        while (start_i < data.len) : (start_i += grad_acc_steps) {
            const end_i = @min(start_i + grad_acc_steps, data.len);
            loss.fill(0);
            loss.grad.?.fill(0, device);
            for (data[start_i..end_i]) |row| {
                // clever way, but NEED to zero these grads
                input.data.data = row[0..input_size];
                target.data.data = row[row.len - output_size .. row.len];
                // this is the more intuitive, less performant method but this keeps all prev ones on the graph which may be req in some cases
                // const input = try Tensor.init(row[0..input_size], &[_]usize{ input_size, output_size }, true, alloc);
                // const target = try Tensor.init(row[input_size..row.len], null, true, alloc);
                const output = try layer.forward(input);
                try output.set_label("output");

                const curr_loss = try simple_mse_loss(f32, output, target, device);
                _ = try loss.data._add(curr_loss.data);
                curr_loss.grad.?.fill(1.0 / @as(T, grad_acc_steps), device);
                try gm.backward(curr_loss);
            }

            optimizer.step(params.?);
            layer.zero_grad();
        }
        optimizer.lr *= lr_epoch_decay;
        log.info("Epoch: {d:<4.4}", .{epoch + 1});
        log.info("AccLoss: {d:<10.4}\t", .{loss.data.data[0]});
        log.info("Weights: {d:>6.4}\t", .{layer.weights.data.data});
        log.info("Bias: {d:.4}\n", .{layer.bias.data.data});
    }
    try std.testing.expectApproxEqAbs(1.5, layer.weights.get(&.{ 0, 0 }), 0.1);
    try std.testing.expectApproxEqAbs(3, layer.weights.get(&.{ 0, 1 }), 0.1);
    try std.testing.expectApproxEqAbs(0, layer.bias.get(&.{ 0, 0 }), 0.1);
}

test "train_grad_accum" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const T = f32;

    const data = read_csv(T, "/tmp/data.csv", alloc) catch |e| {
        std.log.warn("{s} error opening test file. Skipping `train_grad_accum` test.", .{@errorName(e)});
        return;
    };
    try train_grad_accum(T, data, alloc);
}

/// For this implementation the dataset must be evenly divisible by the batch size as a simplification
fn train_batched(comptime T: type, data: [][]T, device: DeviceReference) !void {
    // using an arena is more efficient, less freeing here.
    const X = try device.allocator.alloc([]T, data.len);
    const y = try device.mem_alloc(T, data.len);
    const n_features = 2;
    for (0..data.len) |i| {
        X[i] = try device.mem_alloc(T, n_features);
        for (0..n_features) |j| {
            X[i][j] = data[i][j];
        }

        y[i] = data[i][n_features];
    }
    const Tensor = NDTensor(T);
    const batch_size = 32;
    const input_size = 2;
    const output_size = 1;

    const layer = try LinearLayer(T).init(device, input_size, 1);
    defer layer.deinit();
    const params = layer.get_parameters();
    defer device.allocator.free(params.?);

    // this is a trick to avoid reallocating
    const input = try Tensor.empty(&.{ batch_size, input_size }, true, device);
    try input.set_label("input");
    input.fill(0);

    const target = try Tensor.empty(&.{ batch_size, output_size }, true, device);
    try target.set_label("input");
    target.fill(0);

    input.acquire();
    target.acquire();

    var gm = GraphManager(NDTensor(T)).init(device.allocator, .{ .eager_teardown = true });
    defer gm.deinit();
    var optimizer = SGD(T){ .lr = 0.01, .grad_clip_enabled = true };

    for (0..15) |_| {
        var epoch_loss: T = 0;
        var batch_start_i: usize = 0;
        while (batch_start_i < y.len) : (batch_start_i += batch_size) {
            input.grad.?.fill(0.0, device);
            target.grad.?.fill(0.0, device);
            const batch_end_i = @min(batch_start_i + batch_size, y.len);
            const batch_x = X[batch_start_i..batch_end_i];
            const batch_y = y[batch_start_i..batch_end_i];

            for (0..batch_x.len) |bi| {
                for (0..input_size) |bj| {
                    try input.set(&[_]usize{ bi, bj }, batch_x[bi][bj]);
                }
            }
            for (0..batch_y.len) |i| try target.set(&[_]usize{ i, 0 }, batch_y[i]);

            const output = try layer.forward(input);
            const loss = try simple_mse_loss(f32, output, target, device);
            epoch_loss += loss.data.data[0];

            loss.grad.?.fill(1.0, device);
            layer.zero_grad();
            try gm.backward(loss);
            optimizer.step(params.?);
        }
    }

    try std.testing.expectApproxEqAbs(1.5, layer.weights.get(&.{ 0, 0 }), 0.1);
    try std.testing.expectApproxEqAbs(3, layer.weights.get(&.{ 0, 1 }), 0.1);
    try std.testing.expectApproxEqAbs(0, layer.bias.get(&.{ 0, 0 }), 0.1);
}

test "train_batched" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    var cpu = zg.device.HostDevice.init(arena.allocator());
    defer cpu.deinit();

    const T = f32;

    const data = read_csv(T, "/tmp/data.csv", cpu.allocator) catch |e| {
        std.log.warn("{s} error opening test file. Skipping `train_batched` test.", .{@errorName(e)});
        return;
    };
    try train_batched(T, data, cpu.reference());
}

test "LinearLayer forward and backward" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var cpu = zg.device.HostDevice.init(arena.allocator());
    defer cpu.deinit();

    const device = cpu.reference();

    const T = f32;
    zg.rt_grad_enabled = true;
    // input 2, output 2
    var layer = try LinearLayer(T).init(cpu.reference(), 2, 1);
    layer.weights.fill(2);

    const input_shape = &[_]usize{ 1, 2 };
    const input = try NDTensor(T).init(&[_]T{ 3, 3 }, input_shape, true, cpu.reference());
    const output = try layer.forward(input);
    output.grad.?.fill(1.0, device);

    var gm = GraphManager(NDTensor(T)).init(cpu.allocator, .{});
    defer gm.deinit();
    try gm.backward(output);

    try std.testing.expectEqual(12, output.get(&.{ 0, 0 }));
}
