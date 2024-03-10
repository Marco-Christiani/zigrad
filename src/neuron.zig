const std = @import("std");
const grad = @import("grad.zig");
const Value = grad.Value;
const add = grad.add;
const mul = grad.mul;

pub const ACTIVATION = enum {
    NONE,
    TANH,
};

pub const Neuron = struct {
    const Self = @This();
    weights: []*Value,
    bias: *Value,
    alloc: *const std.mem.Allocator,
    activation: ACTIVATION,

    // Initialize with random weights and zero bias
    pub fn init(allocator: *const std.mem.Allocator, input_dim: usize, seed: u64, activation: ACTIVATION) !*Neuron {
        var random = std.rand.DefaultPrng.init(seed);
        const weights: []*Value = try allocator.alloc(*Value, input_dim);

        for (0.., weights) |i, _| {
            // const label = try std.fmt.allocPrint(allocator.*, "neuron-w-{}-", .{i});
            // defer allocator.free(label);
            weights[i] = try Value.init(allocator, random.random().float(f64), "neuron-");
        }

        const bias = try Value.init(allocator, 0.0, "neuron-bias-");
        const self = try allocator.create(Neuron);
        self.* = Neuron{
            .weights = weights,
            .bias = bias,
            .alloc = allocator,
            .activation = activation,
        };
        return self;
    }

    pub fn forward(self: *Self, allocator: *const std.mem.Allocator, inputs: []*Value) !*Value {
        // operations modify the computation graph which need to outlive this function, so use pointers
        var output = try Value.init(allocator, 0.0, "output-");
        for (inputs, 0..) |input, i| {
            output = add(self.alloc, output, mul(self.alloc, input, self.weights[i]));
        }
        output = add(self.alloc, output, self.bias);
        output = switch (self.activation) {
            .TANH => grad.tanh(self.alloc, output),
            .NONE => output,
        };
        return output;
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

    pub fn update(self: *Self, lr: f64) void {
        for (self.weights) |w| {
            w.value -= lr * w.grad;
        }
        self.bias.value -= lr * self.bias.grad;
    }

    pub fn backward(self: *Self, loss: *Value, lr: f64) void {
        loss.backward() catch {
            @panic("loss.backward() failed.");
        };
        self.update(lr);
        loss.zero_grad();
        self.detach();
        loss.deinit();
        self.attach();
    }
};

pub fn toValues(allocator: *const std.mem.Allocator, arr: []const f64) []*Value {
    var result = allocator.alloc(*Value, arr.len) catch {
        @panic("Failed to allocate for values.");
    };
    for (0..arr.len) |i| {
        result[i] = Value.init(allocator, arr[i], null) catch {
            @panic("Failed to allocate value");
        };
    }
    return result;
}

test "neuron 1->1 backward()" {
    const allocator = std.testing.allocator;
    const input_size = 5;
    const learning_rate = 0.01;

    var neuron = try Neuron.init(&allocator, input_size, 42, ACTIVATION.NONE);

    // 1D array of Values, 1 input of dim 5
    var inputs: []*Value = toValues(&allocator, &[input_size]f64{ 1, 2, 3, 4, 5 });
    for (inputs) |i| {
        _ = i.setLabel("input-");
    }
    // 1D array of Values, 1 target of dim 1
    var targets = [_]*Value{try Value.init(&allocator, 3.0, "target-")};

    const output = try neuron.forward(&allocator, inputs);
    _ = output.setLabel("output-");

    output.print_arrows();

    var preds = [_]*Value{output};
    const loss = try grad.loss_mse(&allocator, &preds, &targets);
    std.debug.print("loss: {}\n", .{loss.value});
    neuron.backward(loss, learning_rate);
    neuron.deinit();
    // since the input values are a part of the computation graph, they will be free'd by now
    // thus, only have to free the inputs array here
    allocator.free(inputs);
}

// test "neuron batch" {
//     const allocator = std.testing.allocator;
//     // const allocator = std.heap.page_allocator;
//     const inputs = [_]f64{ 1, 2, 3, 4, 5 };
//     const targets = [_]f64{ 3, 5.2, 7.2, 9.2, 6 };
//     const input_size = inputs.len;
//     const n_epochs = 5;
//     const learning_rate = 0.01;

//     var neuron = try Neuron.init(&allocator, input_size, 42, ACTIVATION.NONE);

//     for (0..n_epochs) |epoch| {
//         var input_values = toValues(&allocator, @constCast(&inputs));
//         var target_values = toValues(&allocator, @constCast(&targets));
//         const output = try neuron.forward(&allocator, input_values);
//         const loss = try grad.loss_mse(&allocator, output, target_values);
//         try loss.backward();
//         neuron.update(learning_rate);
//         loss.zero_grad();
//         neuron.detach();
//         std.debug.print("epoch: {}, loss: {}\n", .{ epoch, loss.value });

//         loss.deinit();
//         neuron.attach();
//         defer allocator.free(output);
//         defer allocator.free(input_values);
//         defer allocator.free(target_values);
//     }
//     neuron.print();
//     neuron.deinit();
// }

// test "neuron backward()" {
//     const allocator = std.testing.allocator;
//     const inputs = [_]f64{ 1, 2, 3, 4, 5 };
//     const targets = [_]f64{ 3, 5.2, 7.2, 9.2, 6 };
//     const input_size = inputs.len;
//     const n_epochs = 5;
//     const learning_rate = 0.01;

//     var neuron = try Neuron.init(&allocator, input_size, 42, ACTIVATION.NONE);

//     for (0..n_epochs) |epoch| {
//         var input_values = toValues(&allocator, @constCast(&inputs));
//         var target_values = toValues(&allocator, @constCast(&targets));
//         const output = try neuron.forward(&allocator, input_values);
//         const loss_value = try grad.loss_mse(&allocator, output, target_values);
//         const loss = loss_value.value;
//         neuron.backward(loss_value, learning_rate);

//         std.debug.print("epoch: {}, loss: {}\n", .{ epoch, loss });
//         defer allocator.free(output);
//         defer allocator.free(input_values);
//         defer allocator.free(target_values);
//     }
//     neuron.print();
//     neuron.deinit();
// }
