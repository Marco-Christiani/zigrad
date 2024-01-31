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
            weights[i] = try Value.init(allocator, random.random().float(f64), null);
        }

        const bias = try Value.init(allocator, 0.0, null);
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
            outputs[i] = try add(self.alloc, try mul(self.alloc, input, self.weights[i]), self.bias);
        }
        return outputs;
    }

    // TODO: deinit(): see comments below why this wont work as expected
    pub fn deinit(self: *Self) void {
        // should tear down the entire graph by cascading from the tail
        self.weights[self.weights.len - 1].deinit();
        self.alloc.free(self.weights);
        self.alloc.destroy(self.bias);
    }
};

test "neuron 1->1" {
    const allocator = std.testing.allocator;
    const input_size = 1;
    const n_epochs = 10;
    const learning_rate = 0.01;
    _ = learning_rate;

    var neuron = try Neuron.init(&allocator, input_size, 42);

    // const inputs = [_]f64{ 1, 2, 3, 4, 5 };
    const targets = [_]f64{ 1, 4, 9, 16, 25 };
    // _ = targets;
    const inputs = [_]f64{1};

    for (0..n_epochs) |epoch| {
        for (0.., inputs) |i, _| {
            const input_value = try Value.init(&allocator, inputs[i], null);
            var inputs_i = [_]*Value{input_value};

            const target_value = try Value.init(&allocator, targets[i], null);
            var target_i = [_]*Value{target_value};

            // NOTE: outputs have to be free'd but need to live until backprop
            // similar to linearModel() we can just deinit() loss to tear down the graph?
            const output = try neuron.forward(&allocator, &inputs_i);

            // TODO: learning
            const loss = try grad.loss_mse(&allocator, output, &target_i);

            // NOTE: This will destroy the weights in the Neuron
            // Assume what needs to happen is we break the graph after backprop to destroy all intermediate
            // values, but preserve the Neuron's values (or, could possible use the `label` field as a flag)
            defer loss.deinit();
            // neuron.backward(loss);
            // neuron.zeroGrad();
            std.debug.print("i: {}, output: {}\n", .{ i, output[0].value });
        }

        std.debug.print("Epoch: {}\n", .{epoch});
    }
    neuron.deinit();
}
