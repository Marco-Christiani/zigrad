const std = @import("std");
const conv_utils = @import("conv_utils.zig");
const ops = @import("ops.zig");
const Loss = @import("tensor.zig").Loss;
const NDTensor = @import("tensor.zig").NDTensor;
const zarray = @import("zarray.zig");
const NDArray = zarray.NDArray;
const Shape = zarray.Shape;
const SGD = @import("tensor.zig").SGD;
const settings = @import("zigrad").settings;

const log = std.log.scoped(.zigrad_layer);

pub fn Layer(comptime T: type) type {
    return struct {
        const Self = @This();

        vtable: *const VTable,
        ptr: *anyopaque,

        pub const VTable = struct {
            forward: *const fn (ctx: *anyopaque, input: *NDTensor(T), allocator: std.mem.Allocator) anyerror!*NDTensor(T),
            getParameters: *const fn (ctx: *anyopaque) ?[]*const NDTensor(T),
            zeroGrad: *const fn (ctx: *anyopaque) void,
            deinit: *const fn (ctx: *anyopaque) void,
            release: *const fn (ctx: *anyopaque) void,
        };

        pub fn init(pointer: anytype) Self {
            const Ptr = @TypeOf(pointer);

            const gen = struct {
                fn forwardFn(ctx: *anyopaque, input: *NDTensor(T), allocator: std.mem.Allocator) !*NDTensor(T) {
                    const self: Ptr = @ptrCast(@alignCast(ctx));
                    return self.forward(input, allocator);
                }

                fn getParametersFn(ctx: *anyopaque) ?[]*const NDTensor(T) {
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

        pub fn forward(self: Self, input: *NDTensor(T), allocator: std.mem.Allocator) !*NDTensor(T) {
            return self.vtable.forward(self.ptr, input, allocator);
        }

        /// Optional as a layer may not have any trainables
        pub fn getParameters(self: Self) ?[]*const NDTensor(T) {
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

        pub fn init(allocator: std.mem.Allocator, input_size: usize, output_size: usize) !*Self {
            // TODO: random init and, maybe, stack bias (benchmark)
            const self = try allocator.create(Self);
            const weights_shape = &[_]usize{ output_size, input_size };
            const bias_shape = &[_]usize{output_size};
            const weights = (try Tensor.init(try allocator.alloc(T, input_size * output_size), weights_shape, true, allocator)).setLabel("layer_weights");
            const bias = (try Tensor.init(try allocator.alloc(T, output_size), bias_shape, true, allocator)).setLabel("layer_bias");
            weights.acquire();
            bias.acquire();
            weights.fill(1.0);
            bias.fill(1.0);

            self.* = Self{
                .weights = weights,
                .bias = bias,
                .allocator = allocator,
            };

            return self;
        }

        pub fn forward(self: *const Self, input: *Tensor, allocator: std.mem.Allocator) !*Tensor {
            const batch_size = input.data.shape.shape[0];
            const flattened_size = zarray.prod(input.data.shape.shape[1..]);
            const flattened_input = try input.reshape(&[_]usize{ batch_size, flattened_size });
            var result = try flattened_input.matmul(self.weights, allocator, .{ .trans_b = true });
            _ = result.setLabel("fwd1w");

            var biased_result = try result.add(self.bias, allocator);
            return biased_result.setLabel("fwd1b");
        }

        pub fn backward(self: *Self, grad_output: *NDTensor(T), allocator: std.mem.Allocator) !void {
            _ = self;
            _ = grad_output;
            _ = allocator;
        }

        pub fn getParameters(self: *Self) ?[]*const NDTensor(T) {
            // TODO:
            _ = self;
            return null;
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
        input: ?*NDTensor(T), // backward ctx

        pub fn init(allocator: std.mem.Allocator, in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize, dilation: usize) !*Self {
            const self = try allocator.create(Self);
            const weights_shape = &[_]usize{ out_channels, in_channels, kernel_size, kernel_size };
            const bias_shape = &[_]usize{out_channels};
            const weights = (try NDTensor(T).init(
                try allocator.alloc(T, out_channels * in_channels * kernel_size * kernel_size),
                weights_shape,
                true,
                allocator,
            )).setLabel("conv_weights");
            const bias = (try NDTensor(T).init(try allocator.alloc(T, out_channels), bias_shape, true, allocator)).setLabel("conv_bias");

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

            self.weights.acquire();
            self.bias.acquire();
            // TODO: weight init
            self.weights.fill(0.01);
            self.bias.fill(0);

            return self;
        }

        pub fn forward(self: *Self, input: *NDTensor(T), allocator: std.mem.Allocator) !*NDTensor(T) {
            const batch_size = try input.data.shape.get(0);
            const in_height = try input.data.shape.get(2);
            const in_width = try input.data.shape.get(3);

            std.debug.print("Conv2D input shape: {any}\n", .{input.data.shape.shape});
            std.debug.print("Conv2D weights shape: {any}\n", .{self.weights.data.shape.shape});

            const out_height = (in_height + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1;
            const out_width = (in_width + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1;

            std.debug.print("Calculated output dimensions: {d}x{d}\n", .{ out_height, out_width });

            const col = try conv_utils.im2col(T, input.data, self.kernel_size, self.stride, self.padding, self.dilation, allocator);
            defer col.deinit(allocator);

            std.debug.print("im2col output shape: {any}\n", .{col.shape.shape});

            // Reshape weights to 2D for matmul
            var weights_2d = try self.weights.data.copy(allocator);
            defer weights_2d.deinit(allocator);
            try weights_2d.reshape(&[_]usize{ self.out_channels, self.in_channels * self.kernel_size * self.kernel_size });

            std.debug.print("Reshaped weights shape: {any}\n", .{weights_2d.shape.shape});

            // TODO: have broadcasting now tho
            var output_data = try allocator.alloc(T, batch_size * self.out_channels * out_height * out_width);
            errdefer allocator.free(output_data);

            for (0..batch_size) |b| {
                const col_slice = col.data[b * col.shape.shape[1] * col.shape.shape[2] .. (b + 1) * col.shape.shape[1] * col.shape.shape[2]];
                const col_2d = try NDArray(T).init(col_slice, &[_]usize{ col.shape.shape[1], col.shape.shape[2] }, allocator);
                defer col_2d.deinit(allocator);

                const output_2d = try weights_2d.matmul(col_2d, false, false, allocator);
                defer output_2d.deinit(allocator);

                const start_idx = b * self.out_channels * out_height * out_width;
                const end_idx = (b + 1) * self.out_channels * out_height * out_width;
                @memcpy(output_data[start_idx..end_idx], output_2d.data);
            }

            var output = try NDTensor(T).init(output_data, &[_]usize{ batch_size, self.out_channels, out_height, out_width }, true, allocator);

            std.debug.print("Final output shape: {any}\n", .{output.data.shape.shape});

            // Add bias
            for (0..batch_size) |b| {
                for (0..self.out_channels) |oc| {
                    for (0..out_height) |oh| {
                        for (0..out_width) |ow| {
                            const index = ((b * self.out_channels + oc) * out_height + oh) * out_width + ow;
                            output.data.data[index] += self.bias.data.data[oc];
                        }
                    }
                }
            }

            return output;
        }
        // pub fn backward(tensor: *NDTensor(T), allocator: std.mem.Allocator) anyerror!void {
        //     log.info("conv2d backward() start", .{});
        //     const self: *Self = @ptrCast(@alignCast(tensor._backward_ctx.?));
        //     if (self.input == null) {
        //         return error.NoBackwardContext;
        //     }
        //     const batch_size = try self.input.?.data.shape.get(0);
        //     const out_channels = self.out_channels;
        //     const out_height = try tensor.data.shape.get(2);
        //     const out_width = try tensor.data.shape.get(3);
        //     log.info("upstream data: {d}", .{tensor.data.data});
        //     log.info("upstream grad: {d}", .{tensor.grad.?.data});
        //     // copy for reshaping
        //     var grad_output_copy: *NDArray(T) = try tensor.grad.?.copy(allocator);
        //     defer grad_output_copy.deinit(allocator);
        //     try grad_output_copy.reshape(&[_]usize{ batch_size * out_channels, out_height * out_width });
        //     log.info("Reshaped grad_output shape: {any}", .{grad_output_copy.shape.shape});
        //     log.info("Reshaped grad_output data: {d}", .{grad_output_copy.data});
        //     // Compute gradient w.r.t. weights
        //     const col = try conv_utils.im2col(T, self.input.?.data, self.kernel_size, self.stride, self.padding, self.dilation, allocator);
        //     defer col.deinit(allocator);
        //     const grad_weights: *NDArray(T) = try grad_output_copy.matmul(col, false, true, allocator);
        //     _ = try self.weights.grad.?._add(grad_weights);
        //     log.info("grad_weights shape: {any}", .{grad_weights.shape.shape});
        //     log.info("grad_weights data: {d}", .{grad_weights.data});
        //     // Compute gradient w.r.t. bias
        //     const bias_grad: *NDArray(T) = try tensor.grad.?.sum(allocator);
        //     _ = try self.bias.grad.?._add(bias_grad);
        //     log.info("Updated bias grad: {d}", .{self.bias.grad.?.data});
        //     // Compute gradient w.r.t. input
        //     var weights_copy: *NDTensor(T) = try self.weights.copy(allocator);
        //     defer weights_copy.deinit();
        //     _ = try weights_copy.reshape(&[_]usize{ out_channels, self.in_channels * self.kernel_size * self.kernel_size });
        //     log.info("Reshaped weights shape: {any}", .{weights_copy.data.shape.shape});
        //     log.info("Reshaped weights data: {d}", .{weights_copy.data.data});
        //     const grad_col = try weights_copy.data.matmul(grad_output_copy, true, false, allocator);
        //     defer grad_col.deinit(allocator);
        //     log.info("grad_col shape: {any}", .{grad_col.shape.shape});
        //     log.info("grad_col data: {d}", .{grad_col.data});
        //     const grad_input = try conv_utils.col2im(
        //         T,
        //         grad_col,
        //         self.input.?.data.shape.shape,
        //         self.kernel_size,
        //         self.stride,
        //         self.padding,
        //         self.dilation,
        //         allocator,
        //     );
        //     defer grad_input.deinit(allocator);
        //     log.info("grad_input shape: {any}", .{grad_input.shape.shape});
        //     log.info("grad_input data: {d}", .{grad_input.data});
        //     if (self.input.?.grad) |input_grad| {
        //         _ = try input_grad._add(grad_input);
        //     } else {
        //         self.input.?.grad = try grad_input.copy(allocator);
        //     }
        //     log.info("Final input grad: {d}", .{self.input.?.grad.?.data});
        //     log.info("conv2d backward() end", .{});
        // }

        pub fn backward(tensor: *NDTensor(T), allocator: std.mem.Allocator) anyerror!void {
            log.debug("conv2d backward() start", .{});
            const self: *Self = @ptrCast(@alignCast(tensor._backward_ctx.?));
            if (self.input == null) {
                return error.NoBackwardContext;
            }

            const batch_size = try self.input.?.data.shape.get(0);
            const out_channels = self.out_channels;
            const out_height = try tensor.data.shape.get(2);
            const out_width = try tensor.data.shape.get(3);

            log.debug("upstream data: {d}", .{tensor.data.data});
            log.debug("upstream grad: {d}", .{tensor.grad.?.data});

            // Copy for reshaping
            var grad_output_copy: *NDArray(T) = try tensor.grad.?.copy(allocator);
            defer grad_output_copy.deinit(allocator);
            try grad_output_copy.reshape(&[_]usize{ batch_size * out_channels, out_height * out_width });

            log.debug("Reshaped grad_output shape: {any}", .{grad_output_copy.shape.shape});
            log.debug("Reshaped grad_output data: {d}", .{grad_output_copy.data});

            // Compute gradient w.r.t. weights
            const col = try conv_utils.im2col(T, self.input.?.data, self.kernel_size, self.stride, self.padding, self.dilation, allocator);
            defer col.deinit(allocator);

            const grad_weights: *NDArray(T) = try grad_output_copy.matmul(col, false, true, allocator);
            defer grad_weights.deinit(allocator);
            _ = try self.weights.grad.?._add(grad_weights);

            log.debug("grad_weights shape: {any}", .{grad_weights.shape.shape});
            log.debug("grad_weights data: {d}", .{grad_weights.data});

            // Compute gradient w.r.t. bias
            const bias_grad: *NDArray(T) = try tensor.grad.?.sum(allocator);
            defer bias_grad.deinit(allocator);
            _ = try self.bias.grad.?._add(bias_grad);

            log.debug("Updated bias grad: {d}", .{self.bias.grad.?.data});

            // Compute gradient w.r.t. input
            var weights_copy: *NDArray(T) = try self.weights.data.copy(allocator);
            defer weights_copy.deinit(allocator);
            try weights_copy.reshape(&[_]usize{ out_channels, self.in_channels * self.kernel_size * self.kernel_size });

            log.debug("Reshaped weights shape: {any}", .{weights_copy.shape.shape});
            log.debug("Reshaped weights data: {d}", .{weights_copy.data});

            const grad_col = try weights_copy.matmul(grad_output_copy, true, false, allocator);
            defer grad_col.deinit(allocator);

            log.debug("grad_col shape: {any}", .{grad_col.shape.shape});
            log.debug("grad_col data: {d}", .{grad_col.data});

            const grad_input = try conv_utils.col2im(
                T,
                grad_col,
                self.input.?.data.shape.shape,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                allocator,
            );
            defer grad_input.deinit(allocator);

            log.debug("grad_input shape: {any}", .{grad_input.shape.shape});
            log.debug("grad_input data: {d}", .{grad_input.data});

            if (self.input.?.grad) |input_grad| {
                _ = try input_grad._add(grad_input);
            } else {
                self.input.?.grad = grad_input;
            }

            log.debug("Final input grad: {d}", .{self.input.?.grad.?.data});
            log.debug("conv2d backward() end", .{});
        }

        fn getParameters(self: *Self) ?[]*const NDTensor(T) {
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

pub fn ReLULayer(comptime T: type) type {
    return struct {
        const Self = @This();
        allocator: std.mem.Allocator,
        input: ?*NDTensor(T), // backward ctx

        pub fn init(allocator: std.mem.Allocator) !*Self {
            const self = try allocator.create(Self);
            self.* = .{
                .allocator = allocator,
                .input = null,
            };
            return self;
        }

        pub fn forward(self: *Self, input: *NDTensor(T), allocator: std.mem.Allocator) !*NDTensor(T) {
            self.input = input; // just a ref, wont have to free
            const output = try NDTensor(T).init(input.data.data, input.data.shape.shape, true, allocator);
            output.label = "relu";

            for (output.data.data, 0..) |*value, i| {
                value.* = if (input.data.data[i] > 0) input.data.data[i] else 0;
            }
            try output.setChildren(@constCast(&[_]*const NDTensor(T){input}));
            output.setBackward(backward, self);
            return output;
        }

        fn backward(tensor: *NDTensor(T), allocator: std.mem.Allocator) !void {
            log.info("relu backward()", .{});
            const self: *Self = @ptrCast(@alignCast(tensor._backward_ctx.?));
            if (self.input) |input| {
                if (tensor.grad) |g| {
                    const grad_input = input.grad orelse blk: {
                        const new_grad = NDArray(T).init(g.data, g.shape.shape, allocator) catch unreachable;
                        input.grad = new_grad;
                        break :blk new_grad;
                    };
                    for (grad_input.data, 0..) |*grad_value, i| {
                        grad_value.* = if (input.data.data[i] > 0) g.data[i] else 0;
                    }
                }
            }
        }

        pub fn getParameters(self: *Self) ?[]*const NDTensor(T) {
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

pub fn heInitialization(comptime T: type, tensor: *NDTensor(f32)) void {
    const fan_in: T = @floatCast(tensor.data.shape.shape[1]);
    const std_dev = @sqrt(2.0 / fan_in);

    var rng = std.rand.DefaultPrng.init(settings.seed);
    const random = rng.random();

    for (tensor.data.data) |*value| {
        value.* = random.floatNorm(f32) * std_dev;
    }
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
    const params = [_]*NDTensor(f32){ layer.weights, layer.bias };

    var gm = Loss(NDTensor(T)).init(alloc);
    gm.eager_teardown = true;
    gm.grad_clip_enabled = true;
    gm.grad_clip_delta = 1e-6;
    gm.grad_clip_max_norm = 10.0;
    defer gm.deinit();
    var optimizer = SGD(T){ .lr = 0.01 };
    const lr_epoch_decay = 0.9;

    const grad_acc_steps = 32;
    var loss: *Tensor = try Tensor.init(&[_]T{0.0}, null, true, alloc);
    loss.acquire();
    defer loss.teardown(alloc);
    defer loss.release();
    // this is a trick to avoid reallocating
    const input = try Tensor.init(@constCast(&[_]T{0} ** input_size), &[_]usize{ input_size, 1 }, true, alloc);
    input.label = "input";
    const target = try Tensor.init(@constCast(&[_]T{0} ** output_size), &[_]usize{output_size}, true, alloc);
    target.label = "target";
    input.acquire();
    target.acquire();
    defer input.release();
    defer target.release();

    for (0..3) |epoch| {
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
                const output = (try layer.forward(input)).setLabel("output");

                const curr_loss = try ops.simple_mse_loss(f32, output, target, alloc);
                _ = try loss.data._add(curr_loss.data);
                curr_loss.grad.?.fill(1.0 / @as(T, grad_acc_steps));
                try gm.backward(curr_loss, alloc);
            }

            optimizer.step(&params);
            layer.zeroGrad();
        }
        optimizer.lr *= lr_epoch_decay;
        std.debug.print("Epoch: {d:<4.4}", .{epoch + 1});
        std.debug.print("AccLoss: {d:<8.4}\t", .{loss.data.data});
        std.debug.print("Weights: {d:>8.4}\t", .{layer.weights.data.data});
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
    const batch_size = 100;
    const input_size = 2;
    const output_size = 1;

    const layer = try LinearLayer(T).init(alloc, input_size, output_size);
    defer layer.deinit();
    const params = [_]*NDTensor(f32){ layer.weights, layer.bias };
    // try layer.weights.set(&[_]usize{ 0, 0 }, 3);
    //
    // try layer.weights.set(&[_]usize{ 0, 1 }, -0.5);

    // this is a trick to avoid reallocating
    const input = try Tensor.init(@constCast(&[_]T{0} ** (batch_size * input_size)), &[_]usize{ input_size, batch_size }, true, alloc);
    const target = try Tensor.init(@constCast(&[_]T{0} ** (batch_size * output_size)), &[_]usize{ output_size, batch_size }, true, alloc);
    input.acquire();
    target.acquire();

    var gm = Loss(NDTensor(T)).init(alloc);
    gm.eager_teardown = true;
    gm.grad_clip_enabled = true;
    gm.grad_clip_max_norm = 10.0;
    gm.grad_clip_delta = 1e-6;
    defer gm.deinit();
    var optimizer = SGD(T){ .lr = 0.01 };

    for (0..3) |epoch| {
        var batch_start_i: usize = 0;
        while (batch_start_i < y.len) : (batch_start_i += batch_size) {
            input.grad.?.fill(0.0);
            target.grad.?.fill(0.0);
            const batch_end_i = @min(batch_start_i + batch_size, y.len);
            const batch_x = X[batch_start_i..batch_end_i];
            const batch_y = y[batch_start_i..batch_end_i];

            for (0..batch_x.len) |bi| {
                for (0..input_size) |bj| {
                    try input.set(&[_]usize{ bj, bi }, batch_x[bi][bj]);
                }
            }
            for (0..batch_y.len) |i| try target.set(&[_]usize{ 0, i }, batch_y[i]);
            // target.data.data = batch_y;

            const output = try layer.forward(input);
            const loss = try ops.simple_mse_loss(f32, output, target, alloc);

            std.debug.print("Epoch {d}\t", .{epoch + 1});
            std.debug.print("Loss: {d:>10.4}\t", .{loss.data.data});
            std.debug.print("Weights: {d:>10.4}\t", .{layer.weights.data.data});
            std.debug.print("Bias: {d:>10.4}\n", .{layer.bias.data.data});
            std.debug.print("input: {d:.4}\n", .{input.data.data});
            std.debug.print("target: {d:.4}\n", .{target.data.data});
            std.debug.print("output: {d:.4}\n\n", .{output.data.data});

            loss.grad.?.fill(1.0);
            layer.zeroGrad();
            try gm.backward(loss, alloc);
            optimizer.step(&params);
        }
    }

    std.debug.print("Weights: ", .{});
    layer.weights.print();
    std.debug.print("Biases: ", .{});
    layer.bias.print();
    std.debug.print("\n", .{});
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

    const input_shape = &[_]usize{ 2, 1 };
    const input = try NDTensor(T).init(@constCast(&[_]T{ 3, 3 }), input_shape, true, alloc);

    const output = try layer.forward(input);
    @memset(output.grad.?.data, 1.0);

    var gm = Loss(NDTensor(T)).init(alloc);
    defer gm.deinit();
    try gm.backward(output, alloc);

    std.debug.print("Weights: ", .{});
    layer.weights.print();
    std.debug.print("bias: ", .{});
    layer.bias.print();
}
