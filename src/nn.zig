const std = @import("std");
const n = @import("neuron.zig");
const Value = @import("grad.zig").Value;
const loss_mse = @import("grad.zig").loss_mse;
const ACTIVATION = @import("neuron.zig").ACTIVATION;
const Neuron = @import("neuron.zig").Neuron;

pub const Error = error{DimMismatch};

pub const Layer = struct {
    const Self = @This();
    dimIn: usize,
    dimOut: usize,
    activation: ACTIVATION,
    neurons: []*Neuron,
    alloc: *const std.mem.Allocator,

    pub fn init(allocator: *const std.mem.Allocator, dimIn: usize, dimOut: usize, seed: u64, activation: ACTIVATION) !*Self {
        var neurons = try allocator.alloc(*Neuron, dimOut);
        for (0..neurons.len) |i| {
            neurons[i] = try Neuron.init(allocator, dimIn, seed, activation);
        }
        const self = try allocator.create(Self);
        self.* = Self{
            .dimIn = dimIn,
            .dimOut = dimOut,
            .activation = activation,
            .neurons = neurons,
            .alloc = allocator,
        };
        return self;
    }

    pub fn forward(self: *Self, allocator: *const std.mem.Allocator, inputs: []*Value) Error![]*Value {
        if (inputs.len != self.dimIn) {
            std.log.err("Dimensions must match. Got inputs.len={} Layer.dimIn={}\n", .{ inputs.len, self.dimIn });
            return error.DimMismatch;
        }
        var outputs = allocator.alloc(*Value, self.neurons.len) catch unreachable;
        for (0..self.neurons.len) |i| {
            outputs[i] = self.neurons[i].forward(allocator, inputs) catch {
                @panic("Neuron fwd failed.");
            };
        }
        return outputs;
    }

    pub fn update(self: *Self, lr: f64) void {
        for (self.neurons) |ni| {
            ni.update(lr);
        }
    }

    pub fn deinit(self: *Self) void {
        for (self.neurons) |ni| {
            ni.deinit();
        }
        self.alloc.free(self.neurons);
        self.alloc.destroy(self);
    }

    pub fn attach(self: *Self) void {
        for (self.neurons) |ni| {
            ni.attach();
        }
    }

    pub fn detach(self: *Self) void {
        for (self.neurons) |ni| {
            ni.detach();
        }
    }
};

test "nn/layer/init-deinit" {
    const alloc = &std.testing.allocator;
    var layer = try Layer.init(alloc, 2, 2, 44, ACTIVATION.TANH);
    layer.deinit();
    std.debug.print("\n\nlayer init/deinit done\n\n", .{});
}

test "nn/layer/forward" {
    const alloc = std.testing.allocator;
    var layer = try Layer.init(&alloc, 2, 2, 44, ACTIVATION.TANH);
    var inputs: []*Value = n.toValues(&alloc, &[_]f64{ 1, 2 });
    var outputs = layer.forward(&alloc, inputs) catch unreachable;

    var targets: []*Value = n.toValues(&alloc, &[_]f64{ 1.0, 2.0 });
    std.debug.print("outputs.len={} targets.len={}\n", .{ outputs.len, targets.len });
    const loss = try loss_mse(&alloc, outputs, targets);
    std.debug.print("loss: {}\n", .{loss.value});

    layer.detach();
    loss.deinit();
    layer.attach(); // reattaching is not strictly necessary since zigrad knows you want to destroy this subgraph directly
    layer.deinit();

    alloc.free(outputs);
    alloc.free(inputs);
    alloc.free(targets);
}

test "nn/layer/backward" {
    const alloc = std.testing.allocator;
    var layer = try Layer.init(&alloc, 2, 2, 44, ACTIVATION.TANH);
    var inputs: []*Value = n.toValues(&alloc, &[_]f64{ 1, 2 });
    var outputs = layer.forward(&alloc, inputs) catch unreachable;

    var targets: []*Value = n.toValues(&alloc, &[_]f64{ 1.0, 2.0 });
    std.debug.print("outputs.len={} targets.len={}\n", .{ outputs.len, targets.len });
    const loss = try loss_mse(&alloc, outputs, targets);
    std.debug.print("loss: {}\n", .{loss.value});
    try loss.backward();
    layer.update(1e-3);
    layer.detach();
    loss.deinit();
    layer.attach(); // reattaching is not strictly necessary since zigrad knows you want to destroy this subgraph directly
    layer.deinit();

    alloc.free(outputs);
    alloc.free(inputs);
    alloc.free(targets);
}
