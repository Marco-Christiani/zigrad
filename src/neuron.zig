const std = @import("std");
const grad = @import("grad.zig");
const Value = grad.Value;
const add = grad.add;
const mul = grad.mul;

pub const Neuron = struct {
    const Self = @This();
    weights: []*Value,
    bias: *Value,
    alloc: *const std.mem.Allocator,

    // Initialize with random weights and zero bias
    pub fn init(allocator: *const std.mem.Allocator, input_dim: usize, seed: u64) !*Neuron {
        var random = std.rand.DefaultPrng.init(seed);
        const weights: []*Value = try allocator.alloc(*Value, input_dim);

        for (0.., weights) |i, _| {
            const label = try std.fmt.allocPrint(allocator.*, "neuron-w-{}", .{i});
            defer allocator.free(label);
            weights[i] = try Value.init(allocator, random.random().float(f64), label);
        }

        const bias = try Value.init(allocator, 0.0, "neuron-bias");
        const self = try allocator.create(Neuron);
        self.* = Neuron{
            .weights = weights,
            .bias = bias,
            .alloc = allocator,
        };
        return self;
    }

    pub fn forward(self: *Self, allocator: *const std.mem.Allocator, inputs: []*Value) ![]*Value {
        // operations modify the computation graph which need to outlive this function, so use pointers
        var outputs = try allocator.alloc(*Value, inputs.len);
        for (inputs, 0..) |input, i| {
            outputs[i] = add(self.alloc, mul(self.alloc, input, self.weights[i]), self.bias);
        }
        return outputs;
    }

    pub fn deinit(self: *Self) void {
        for (self.weights) |w| {
            w.deinit();
        }
        self.alloc.free(self.weights);
        self.bias.deinit();
        self.alloc.destroy(self);
    }

    pub fn detach(self: *Self) void {
        for (self.weights) |w| {
            w.detach();
        }
        self.bias.detach();
    }
    pub fn attach(self: *Self) void {
        for (self.weights) |w| {
            w.detached = false;
        }
        self.bias.detached = false;
    }

    pub fn print(self: *Self) void {
        for (0.., self.weights) |i, w| {
            std.debug.print("weight {}\n", .{i});
            w.print();
        }
        std.debug.print("bias\n", .{});
        self.bias.print();
    }

    pub fn update(self: *Self, lr: f32) void {
        for (self.weights) |w| {
            w.value -= lr * w.grad;
        }
        self.bias.value -= lr * self.bias.grad;
    }
};

test "neuron 1->1" {
    const allocator = std.testing.allocator;
    // const allocator = std.heap.page_allocator;
    const input_size = 1;
    const n_epochs = 5;
    const learning_rate = 0.01;

    var neuron = try Neuron.init(&allocator, input_size, 42);

    const inputs = [_]f64{ 1, 2, 3, 4, 5 };
    const targets = [_]f64{ 3, 5, 7, 9, 11 };
    // const inputs = [_]f64{1};

    for (0..n_epochs) |epoch| {
        for (0.., inputs) |i, _| {
            neuron.attach();
            const input_value = try Value.init(&allocator, inputs[i], "input-");
            var inputs_i = [_]*Value{input_value};

            const target_value = try Value.init(&allocator, targets[i], "target-");
            var target_i = [_]*Value{target_value};

            // inputs_i[0].print();
            // target_i[0].print();

            const output = try neuron.forward(&allocator, &inputs_i);

            // std.debug.print("\nGraph 1: {s}\n", .{"{"});
            // output[0].print_arrows();
            // std.debug.print("{s}\n", .{"}"});

            const loss = try grad.loss_mse(&allocator, output, &target_i);
            defer allocator.free(output);
            defer loss.deinit();
            try loss.backward();
            neuron.update(learning_rate);
            try loss.zero_grad();

            neuron.detach();

            // std.debug.print("Graph 2: {s}\n", .{"{"});
            // loss.print_arrows();
            // std.debug.print("{s}\n", .{"}"});

            std.debug.print("i: {}, loss: {}\n", .{ i, loss.value });
        }

        std.debug.print("Epoch: {}\n", .{epoch});
    }
    neuron.print();
    neuron.deinit();
}
