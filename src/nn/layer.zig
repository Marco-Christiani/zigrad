const std = @import("std");
const zg = @import("../root.zig");

const NDTensor = zg.NDTensor;
const GraphManager = zg.GraphManager;
const SGD = zg.optim.SGD;

const Shape = zg.Shape;
const NDArray = zg.NDArray;

const prod = zg.arrayutils.prod;
const conv_utils = zg.conv_utils;
const simple_mse_loss = zg.loss.simple_mse_loss;
const winit = zg.winit;

const log = std.log.scoped(.zg_layer);

pub fn Layer(comptime T: type) type {
    return struct {
        const Self = @This();

        vtable: *const VTable,
        ptr: *anyopaque,

        pub const VTable = struct {
            forward: *const fn (ctx: *anyopaque, input: *const NDTensor(T), allocator: std.mem.Allocator) anyerror!*NDTensor(T),
            getParameters: *const fn (ctx: *anyopaque) ?[]*NDTensor(T),
            zeroGrad: *const fn (ctx: *anyopaque) void,
            deinit: *const fn (ctx: *anyopaque) void,
            release: *const fn (ctx: *anyopaque) void,
        };

        pub fn init(pointer: anytype) Self {
            const Ptr = @TypeOf(pointer);

            const gen = struct {
                fn forwardFn(ctx: *anyopaque, input: *const NDTensor(T), allocator: std.mem.Allocator) !*NDTensor(T) {
                    const self: Ptr = @ptrCast(@alignCast(ctx));
                    return self.forward(input, allocator);
                }

                fn getParametersFn(ctx: *anyopaque) ?[]*NDTensor(T) {
                    const self: Ptr = @ptrCast(@alignCast(ctx));
                    return self.getParameters();
                }

                fn zeroGradFn(ctx: *anyopaque) void {
                    const self: Ptr = @ptrCast(@alignCast(ctx));
                    self.zeroGrad();
                }

                fn deinitFn(ctx: *anyopaque) void {
                    const self: Ptr = @ptrCast(@alignCast(ctx));
                    self.deinit();
                }

                fn releaseFn(ctx: *anyopaque) void {
                    const self: Ptr = @ptrCast(@alignCast(ctx));
                    self.release();
                }

                const vtable = VTable{
                    .forward = forwardFn,
                    .getParameters = getParametersFn,
                    .zeroGrad = zeroGradFn,
                    .deinit = deinitFn,
                    .release = releaseFn,
                };
            };

            return .{
                .vtable = &gen.vtable,
                .ptr = pointer,
            };
        }

        pub fn forward(self: Self, input: *const NDTensor(T), allocator: std.mem.Allocator) !*NDTensor(T) {
            return self.vtable.forward(self.ptr, input, allocator);
        }

        /// Note that layer may not have any trainables
        pub fn getParameters(self: Self) ?[]*NDTensor(T) {
            return self.vtable.getParameters(self.ptr);
        }

        pub fn zeroGrad(self: Self) void {
            self.vtable.zeroGrad(self.ptr);
        }

        pub fn deinit(self: Self) void {
            self.vtable.deinit(self.ptr);
        }

        pub fn release(self: Self) void {
            self.vtable.release(self.ptr);
        }
    };
}

pub fn LinearLayer(comptime T: type) type {
    return struct {
        const Self = @This();
        const Tensor = NDTensor(T);
        weights: *Tensor,
        bias: *Tensor,
        allocator: std.mem.Allocator,
        parameters: [2]*NDTensor(T),

        pub fn init(allocator: std.mem.Allocator, input_size: usize, output_size: usize) !*Self {
            // TODO: maybe stack bias (benchmark)
            const weights_shape = [_]usize{ output_size, input_size };
            const bias_shape = [_]usize{ output_size, 1 };
            var weights = (try Tensor.empty(&weights_shape, true, allocator)).setLabel("linear_weights");
            var bias = (try Tensor.empty(&bias_shape, true, allocator)).setLabel("linear_bias");
            weights.acquire();
            bias.acquire();
            winit.heInit(T, weights);
            bias.fill(0.0);
            const self = try allocator.create(Self);
            self.* = Self{
                .weights = weights,
                .bias = bias,
                .parameters = [_]*NDTensor(T){ weights, bias },
                .allocator = allocator,
            };
            return self;
        }

        pub fn forward(self: *const Self, input: *const Tensor, fwd_allocator: std.mem.Allocator) !*Tensor {
            var result = (try self.weights.matmul(input, fwd_allocator, .{ .trans_b = true })).setLabel("fwd1w");
            self.weights.logShape(null);
            self.bias.logShape(null);
            input.logShape(null);
            result.logShape(null);
            result = (try self.bias.add(result, fwd_allocator)).setLabel("fwd1b");
            const rt = try result.transpose();
            return rt;
        }

        pub fn backward(self: Self, allocator: std.mem.Allocator) !void {
            _ = self;
            _ = allocator;
        }

        pub fn getParameters(self: *Self) ?[]*NDTensor(T) {
            return @constCast(&self.parameters);
        }

        pub fn zeroGrad(self: *Self) void {
            self.weights.grad.?.fill(0);
            self.bias.grad.?.fill(0);
        }

        pub fn deinit(self: *Self) void {
            self.release();
            self.weights.deinit();
            self.bias.deinit();
            self.allocator.destroy(self);
            self.* = undefined;
        }

        fn release(self: *Self) void {
            self.weights.release();
            self.bias.release();
        }

        pub fn asLayer(self: *Self) Layer(T) {
            return Layer(T).init(self);
        }
    };
}

pub fn Conv2DLayer(comptime T: type) type {
    return struct {
        const Self = @This();

        weights: *NDTensor(T),
        bias: *NDTensor(T),
        allocator: std.mem.Allocator,
        parameters: [2]*NDTensor(T),
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        input: ?NDTensor(T), // backward ctx, TBD if we should go back to this for static graph support later

        pub fn init(
            allocator: std.mem.Allocator,
            in_channels: usize,
            out_channels: usize,
            kernel_size: usize,
            stride: usize,
            padding: usize,
            dilation: usize,
        ) !*Self {
            const weights_shape = [_]usize{ out_channels, in_channels, kernel_size, kernel_size };
            const bias_shape = [_]usize{out_channels};
            var weights = (try NDTensor(T).empty(
                &weights_shape,
                true,
                allocator,
            )).setLabel("conv2d_weights");
            var bias = (try NDTensor(T).empty(&bias_shape, true, allocator)).setLabel("conv2d_bias");
            weights.acquire();
            bias.acquire();
            winit.heInit(T, weights);
            bias.fill(0.0);
            const self = try allocator.create(Self);

            self.* = .{
                .weights = weights,
                .bias = bias,
                .allocator = allocator,
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

        pub fn forward(self: *Self, input: *const NDTensor(T), fwd_allocator: std.mem.Allocator) !*NDTensor(T) {
            const batch_size = try input.data.shape.get(0);
            const in_height = try input.data.shape.get(2);
            const in_width = try input.data.shape.get(3);
            const out_height = (in_height + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1;
            const out_width = (in_width + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1;

            const col: *NDArray(T) = try conv_utils.im2col(T, input.data.*, self.kernel_size, self.stride, self.padding, self.dilation, fwd_allocator);
            defer col.deinit(fwd_allocator);

            try self.weights.data._reshape(&[_]usize{ self.out_channels, self.in_channels * self.kernel_size * self.kernel_size });

            // broadcasted mm for batching
            var output = try self.weights.data.matmul(col, false, false, fwd_allocator);

            // reshape to 4D
            try output._reshape(&[_]usize{ batch_size, self.out_channels, out_height, out_width });

            // bias
            try self.bias.data._reshape(&[_]usize{ 1, self.out_channels, 1, 1 });
            _ = try output._add(self.bias.data);

            defer {
                // restore weights shape after forward (TODO: this can be optimized out later?)
                self.weights.data._reshape(&[_]usize{
                    self.out_channels,
                    self.in_channels,
                    self.kernel_size,
                    self.kernel_size,
                }) catch std.debug.panic("Weights reshape failed.", .{});

                self.bias.data._reshape(&[_]usize{
                    self.out_channels,
                }) catch std.debug.panic("Bias reshape failed.", .{});
            }
            return try NDTensor(T).createDependent(.{
                .data = output,
                .children = &[_]*const NDTensor(T){input},
                .label = "conv2d_out",
                .requires_grad = true,
                .allocator = fwd_allocator,
                ._backward = backward,
                ._backward_ctx = self,
            });
        }

        pub fn backward(tensor: NDTensor(T), allocator: std.mem.Allocator) anyerror!void {
            const self: *Self = @ptrCast(@alignCast(tensor._backward_ctx orelse return error.NoBackwardContext));
            std.debug.assert(tensor.children.?.len == 1);
            const input = tensor.children.?[0];
            const batch_size = try input.data.shape.get(0);
            const out_channels = self.out_channels;
            const out_height = try tensor.data.shape.get(2);
            const out_width = try tensor.data.shape.get(3);

            // copy for reshaping
            var grad_output_copy = try tensor.grad.?.copy(allocator);
            defer grad_output_copy.deinit(allocator);
            try grad_output_copy._reshape(&[_]usize{ batch_size * out_channels, out_height * out_width });

            // w.r.t. weights
            const col = try conv_utils.im2col(T, input.data.*, self.kernel_size, self.stride, self.padding, self.dilation, allocator);
            defer col.deinit(allocator);

            try grad_output_copy._reshape(&[_]usize{ out_channels, batch_size * out_height * out_width });
            try col._reshape(&[_]usize{ batch_size * out_height * out_width, self.in_channels * self.kernel_size * self.kernel_size });

            const grad_weights = try grad_output_copy.matmul(col, false, false, allocator);
            defer grad_weights.deinit(allocator);

            // grad_weights should be (out_channels, in_channels * kernel_size * kernel_size)
            try grad_weights._reshape(&[_]usize{
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            });
            _ = try self.weights.grad.?._add(grad_weights);

            // w.r.t. bias
            const bias_grad = try tensor.grad.?.sum(allocator);
            defer bias_grad.deinit(allocator);
            _ = try self.bias.grad.?._add(bias_grad);

            // w.r.t. input
            var weights_copy = try self.weights.data.copy(allocator);
            defer weights_copy.deinit(allocator);
            try weights_copy._reshape(&[_]usize{ out_channels, self.in_channels * self.kernel_size * self.kernel_size });

            const grad_col = try weights_copy.matmul(grad_output_copy, true, false, allocator);
            defer grad_col.deinit(allocator);

            const grad_input = try conv_utils.col2im(
                T,
                grad_col.*,
                input.data.shape.shape,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                allocator,
            );
            defer grad_input.deinit(allocator);

            if (input.grad) |input_grad| {
                _ = try input_grad._add(grad_input);
            } else {
                @panic("grad didnt exist in backward");
            }
        }

        fn getParameters(self: *Self) ?[]*NDTensor(T) {
            return @constCast(&self.parameters);
        }

        pub fn zeroGrad(self: *Self) void {
            self.weights.grad.?.fill(0);
            self.bias.grad.?.fill(0);
        }

        pub fn deinit(self: *Self) void {
            self.release();
            self.weights.deinit();
            self.bias.deinit();
            // self.input should not be freed be us unless we choose to copy for backward later
            // if (self.input) |input| input.deinit(allocator);
            self.allocator.destroy(self);
        }

        pub fn release(self: *Self) void {
            self.weights.release();
            self.bias.release();
        }

        pub fn asLayer(self: *Self) Layer(T) {
            return Layer(T).init(self);
        }
    };
}

// TODO: yes, there is no reason for this to be a layer, need to add support to the Layer interface for fromStatelessFunc kinda thing
pub fn ReLULayer(comptime T: type) type {
    return struct {
        const Self = @This();
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) !*Self {
            const self = try allocator.create(Self);
            self.* = Self{
                .allocator = allocator,
            };
            return self;
        }

        pub fn forward(_: *Self, input: *const NDTensor(T), allocator: std.mem.Allocator) !*NDTensor(T) {
            const output = try input.data.copy(allocator);
            for (output.data) |*e| {
                e.* = if (e.* > 0) e.* else 0;
            }

            return try NDTensor(T).createDependent(.{
                .data = output,
                .children = &[_]*const NDTensor(T){input},
                .label = "relu_out",
                .requires_grad = true,
                .allocator = allocator,
                ._backward = backward,
            });
        }

        fn backward(tensor: NDTensor(T), _: std.mem.Allocator) !void {
            std.debug.assert(tensor.children.?.len == 1);
            const input = tensor.children.?[0];
            if (tensor.grad) |g| {
                const grad_input = input.grad.?;
                for (grad_input.data, 0..) |*grad_value, i| {
                    grad_value.* = if (input.data.data[i] > 0) g.data[i] else 0;
                    // if (input.data.data[i] > 0 and g.data[i] != 0) std.debug.print("panda\t", .{});
                }
            }
        }

        pub fn getParameters(self: *Self) ?[]*NDTensor(T) {
            _ = self;
            return null;
        }

        pub fn zeroGrad(self: *Self) void {
            _ = self;
        }

        pub fn deinit(self: *Self) void {
            self.allocator.destroy(self);
        }

        pub fn release(self: *Self) void {
            _ = self;
        }

        pub fn asLayer(self: *Self) Layer(T) {
            return Layer(T).init(self);
        }
    };
}

pub fn FlattenLayer(comptime T: type) type {
    return struct {
        const Self = @This();

        pub fn forward(_: *Self, input: *const NDTensor(T), _: std.mem.Allocator) !*NDTensor(T) {
            const batch_dim = input.data.shape.shape[0];
            const other_dims = input.data.shape.shape[1..];
            return try input.reshape(&.{ batch_dim, prod(other_dims) });
        }

        pub fn getParameters(_: *Self) ?[]*NDTensor(T) {
            return null;
        }

        pub fn zeroGrad(_: *Self) void {}

        pub fn deinit(_: *Self) void {}

        pub fn release(_: *Self) void {}

        pub fn asLayer(self: *Self) Layer(T) {
            return Layer(T).init(self);
        }
    };
}

pub fn MaxPool2DLayer(comptime T: type) type {
    return struct {
        const Self = @This();

        kernel_size: usize,
        stride: usize,
        padding: usize,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, kernel_size: usize, stride: usize, padding: usize) !*Self {
            const self = try allocator.create(Self);
            self.* = .{
                .kernel_size = kernel_size,
                .stride = stride,
                .padding = padding,
                .allocator = allocator,
            };
            return self;
        }

        pub fn forward(self: *Self, input: *const NDTensor(T), fwd_allocator: std.mem.Allocator) !*NDTensor(T) {
            const batch_size = try input.data.shape.get(0);
            const channels = try input.data.shape.get(1);
            const in_height = try input.data.shape.get(2);
            const in_width = try input.data.shape.get(3);

            const out_height = (in_height + 2 * self.padding - self.kernel_size) / self.stride + 1;
            const out_width = (in_width + 2 * self.padding - self.kernel_size) / self.stride + 1;

            var output = try NDArray(T).empty(&[_]usize{ batch_size, channels, out_height, out_width }, fwd_allocator);
            var indices = try NDArray(usize).empty(&[_]usize{ batch_size, channels, out_height, out_width }, fwd_allocator);

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

            return try NDTensor(T).createDependent(.{
                .data = output,
                .children = &[_]*const NDTensor(T){input},
                .label = "maxpool2d_out",
                .requires_grad = true,
                .allocator = fwd_allocator,
                ._backward = backward,
                ._backward_ctx = indices,
            });
        }

        pub fn backward(tensor: NDTensor(T), allocator: std.mem.Allocator) anyerror!void {
            const indices = @as(*NDArray(usize), @ptrCast(@alignCast(tensor._backward_ctx orelse return error.NoBackwardContext)));
            std.debug.assert(tensor.children.?.len == 1);
            var input = tensor.children.?[0];

            for (0..tensor.grad.?.data.len) |i| {
                const index = indices.data[i];
                input.grad.?.data[index] += tensor.grad.?.data[i];
            }

            indices.deinit(allocator);
        }

        pub fn getParameters(self: *Self) ?[]*NDTensor(T) {
            _ = self;
            return null; // MaxPool2D has no learnable parameters
        }

        pub fn zeroGrad(self: *Self) void {
            _ = self;
            // No gradients to zero out
        }

        pub fn deinit(self: *Self) void {
            self.allocator.destroy(self);
        }

        pub fn release(self: *Self) void {
            _ = self;
            // No tensors to release
        }

        pub fn asLayer(self: *Self) Layer(T) {
            return Layer(T).init(self);
        }
    };
}

pub fn readCsv(comptime T: type, file_path: []const u8, alloc: std.mem.Allocator) ![][]T {
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
    const data = try readCsv(T, "/tmp/data.csv", alloc);
    std.debug.print("data.len={}\n", .{data.len});
    try trainGradAccum(T, data, alloc);
}

fn trainGradAccum(comptime T: type, data: [][]T, alloc: std.mem.Allocator) !void {
    const Tensor = NDTensor(T);
    const input_size = 2;
    const output_size = 1;

    const layer = try LinearLayer(T).init(alloc, input_size, output_size);
    defer layer.deinit();
    const params = layer.getParameters();
    defer if (params) |p| alloc.free(p);
    for (params.?) |p| {
        p.fill(1);
    }

    var gm = GraphManager(NDTensor(T)).init(alloc, .{});
    gm.eager_teardown = false;
    gm.grad_clip_enabled = true;
    gm.grad_clip_delta = 1e-6;
    gm.grad_clip_max_norm = 10.0;
    defer gm.deinit();
    var optimizer = SGD(T){ .lr = 0.01 };
    const lr_epoch_decay = 1.0; // disable

    const grad_acc_steps = 32;
    var loss: *Tensor = try Tensor.init(&[_]T{0.0}, null, true, alloc);
    loss.acquire();
    defer loss.teardown();
    defer loss.release();
    // this is a trick to avoid reallocating
    const input = (try Tensor.init(
        @constCast(&[_]T{0} ** input_size),
        &[_]usize{ 1, input_size },
        true,
        alloc,
    )).setLabel("input");
    const target = (try Tensor.init(
        @constCast(&[_]T{0} ** output_size),
        &[_]usize{output_size},
        true,
        alloc,
    )).setLabel("target");
    input.acquire();
    target.acquire();
    defer input.release();
    defer target.release();

    for (0..15) |epoch| {
        var start_i: usize = 0;
        while (start_i < data.len) : (start_i += grad_acc_steps) {
            const end_i = @min(start_i + grad_acc_steps, data.len);
            loss.fill(0);
            loss.grad.?.fill(0);
            for (data[start_i..end_i]) |row| {
                // clever way, but NEED to zero these grads
                input.data.data = row[0..input_size];
                target.data.data = row[row.len - output_size .. row.len];
                // this is the more intuitive, less performant method but this keeps all prev ones on the graph which may be req in some cases
                // const input = try Tensor.init(row[0..input_size], &[_]usize{ input_size, output_size }, true, alloc);
                // const target = try Tensor.init(row[input_size..row.len], null, true, alloc);
                const output = (try layer.forward(input, alloc)).setLabel("output");

                const curr_loss = try simple_mse_loss(f32, output, target, alloc);
                _ = try loss.data._add(curr_loss.data);
                curr_loss.grad.?.fill(1.0 / @as(T, grad_acc_steps));
                try gm.backward(curr_loss, alloc);
            }

            optimizer.step(params.?);
            layer.zeroGrad();
        }
        optimizer.lr *= lr_epoch_decay;
        std.debug.print("Epoch: {d:<4.4}", .{epoch + 1});
        std.debug.print("AccLoss: {d:<10.4}\t", .{loss.data.data[0]});
        std.debug.print("Weights: {d:>6.4}\t", .{layer.weights.data.data});
        std.debug.print("Bias: {d:.4}\n", .{layer.bias.data.data});
    }
}

test "trainGradAccum" {
    std.debug.print("{s} trainGradAccum {s}\n", .{ "-" ** 5, "-" ** 5 });
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const T = f32;

    const data = try readCsv(T, "/tmp/data.csv", alloc);
    try trainGradAccum(T, data, alloc);
}

/// For this implementation the dataset must be evenly divisible by the batch size as a simplification
fn trainBatched(comptime T: type, data: [][]T, alloc: std.mem.Allocator) !void {
    // using an arena is more efficient, less freeing here.
    const X = try alloc.alloc([]T, data.len);
    const y = try alloc.alloc(T, data.len);
    const n_features = 2;
    for (0..data.len) |i| {
        X[i] = try alloc.alloc(T, n_features);
        for (0..n_features) |j| {
            X[i][j] = data[i][j];
        }

        y[i] = data[i][n_features];
    }
    const Tensor = NDTensor(T);
    const batch_size = 32;
    const input_size = 2;
    const output_size = 1;

    const layer = try LinearLayer(T).init(alloc, input_size, 1);
    defer layer.deinit();
    const params = layer.getParameters();
    defer alloc.free(params.?);

    // this is a trick to avoid reallocating
    const input = try Tensor.init(@constCast(&[_]T{0} ** (batch_size * input_size)), &[_]usize{ batch_size, input_size }, true, alloc);
    const target = try Tensor.init(@constCast(&[_]T{0} ** (batch_size * output_size)), &[_]usize{ batch_size, output_size }, true, alloc);
    input.acquire();
    target.acquire();

    var gm = GraphManager(NDTensor(T)).init(alloc, .{});
    gm.eager_teardown = true;
    gm.grad_clip_enabled = true;
    gm.grad_clip_max_norm = 10.0;
    gm.grad_clip_delta = 1e-6;
    defer gm.deinit();
    var optimizer = SGD(T){ .lr = 0.01 };

    for (0..15) |epoch| {
        var epoch_loss: T = 0;
        var batch_start_i: usize = 0;
        while (batch_start_i < y.len) : (batch_start_i += batch_size) {
            input.grad.?.fill(0.0);
            target.grad.?.fill(0.0);
            const batch_end_i = @min(batch_start_i + batch_size, y.len);
            const batch_x = X[batch_start_i..batch_end_i];
            const batch_y = y[batch_start_i..batch_end_i];

            for (0..batch_x.len) |bi| {
                for (0..input_size) |bj| {
                    try input.set(&[_]usize{ bi, bj }, batch_x[bi][bj]);
                }
            }
            for (0..batch_y.len) |i| try target.set(&[_]usize{ i, 0 }, batch_y[i]);

            const output = try layer.forward(input, alloc);
            const loss = try simple_mse_loss(f32, output, target, alloc);
            epoch_loss += loss.data.data[0];

            loss.grad.?.fill(1.0);
            layer.zeroGrad();
            try gm.backward(loss, alloc);
            optimizer.step(params.?);
        }

        std.debug.print("Epoch {d:<4}", .{epoch + 1});
        std.debug.print("Loss: {d:<10.4}\t", .{epoch_loss});
        std.debug.print("Weights: {d:>6.4}\t", .{layer.weights.data.data[0..input_size]});
        std.debug.print("Bias: {d:<4.4}\n", .{layer.bias.data.data});
    }

    std.debug.print("Weights: ", .{});
    layer.weights.print();
    std.debug.print("Biases: ", .{});
    layer.bias.print();
    std.debug.print("\n", .{});
    try std.testing.expectApproxEqAbs(1.5, layer.weights.get(&.{ 0, 0 }), 0.1);
    try std.testing.expectApproxEqAbs(3, layer.weights.get(&.{ 0, 1 }), 0.1);
    try std.testing.expectApproxEqAbs(0, layer.bias.get(&.{ 0, 0 }), 0.1);
}

test "trainBatched" {
    std.debug.print("{s} trainBatched2 {s}\n", .{ "-" ** 5, "-" ** 5 });
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const T = f32;

    const data = try readCsv(T, "/tmp/data.csv", alloc);
    try trainBatched(T, data, alloc);
}

test "LinearLayer forward and backward" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f32;

    // input 2, output 2
    var layer = try LinearLayer(T).init(alloc, 2, 2);
    layer.weights.fill(2);

    const input_shape = &[_]usize{ 1, 1, 2 };
    const input = try NDTensor(T).init(@constCast(&[_]T{ 3, 3 }), input_shape, true, alloc);

    const output = try layer.forward(input, alloc);
    @memset(output.grad.?.data, 1.0);

    var gm = GraphManager(NDTensor(T)).init(alloc, .{});
    defer gm.deinit();
    try gm.backward(output, alloc);

    std.debug.print("Weights: ", .{});
    layer.weights.print();
    std.debug.print("bias: ", .{});
    layer.bias.print();
    try std.testing.expectEqual(12, output.get(&.{ 0, 0, 0 }));
}
