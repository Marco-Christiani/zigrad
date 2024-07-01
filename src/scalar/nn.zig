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
    const inputs: []*Value = n.toValues(&alloc, &[_]f64{ 1, 2 });
    const outputs = layer.forward(&alloc, inputs) catch unreachable;

    const targets: []*Value = n.toValues(&alloc, &[_]f64{ 1.0, 2.0 });
    std.debug.print("outputs.len={} targets.len={}\n", .{ outputs.len, targets.len });
    const loss = try loss_mse(&alloc, outputs, targets);
    std.debug.print("loss: {}\n", .{loss.value});

    layer.detach();
    loss.deinit();
    layer.attach(); // reattaching is not strictly necessary since zigrad knows you want to destroy this subgraph directly
    layer.deinit();

    alloc.free(outputs); // outputs already deinit'd
    alloc.free(inputs); // inputs already deinit'd
    alloc.free(targets); // targets already deinit'd
}

test "nn/layer/backward" {
    const alloc = std.testing.allocator;
    var layer = try Layer.init(&alloc, 2, 2, 44, ACTIVATION.TANH);
    const inputs: []*Value = n.toValues(&alloc, &[_]f64{ 1, 2 });
    const outputs = layer.forward(&alloc, inputs) catch unreachable;

    const targets: []*Value = n.toValues(&alloc, &[_]f64{ 1.0, 2.0 });
    std.debug.print("outputs.len={} targets.len={}\n", .{ outputs.len, targets.len });
    const loss = try loss_mse(&alloc, outputs, targets);
    std.debug.print("loss: {}\n", .{loss.value});
    try loss.backward();
    layer.update(1e-3);
    layer.detach();
    loss.deinit();
    layer.attach(); // reattaching is not strictly necessary since zigrad knows you want to destroy this subgraph directly
    layer.deinit();

    alloc.free(outputs); // outputs already deinit'd
    alloc.free(inputs); // inputs already deinit'd
    alloc.free(targets); // targets already deinit'd
}

pub fn trainLayer(alloc: *const std.mem.Allocator) !void {
    const data = @embedFile("data.csv");
    std.debug.print("{s}\n", .{data[0..16]});

    const lr: f64 = 1e-2;
    const batchsize = 70;
    const n_epochs = 100;
    const seed: u64 = 321;

    var layer = try Layer.init(alloc, 1, 1, seed, ACTIVATION.NONE);

    var nSteps: usize = 0;
    for (0..n_epochs) |e| {
        var batchy = try alloc.alloc(*Value, batchsize);
        defer alloc.free(batchy);
        var batchyh = try alloc.alloc(*Value, batchsize);
        defer alloc.free(batchyh);

        var data_iter = std.mem.tokenizeScalar(u8, data, '\n');
        var i: usize = 0;

        var currLoss: f64 = undefined;
        while (data_iter.next()) |line| : (i += 1) {
            nSteps += 1;
            var row_iter = std.mem.tokenizeScalar(u8, line, ',');
            const x = try std.fmt.parseFloat(f64, row_iter.next().?);
            const y = try std.fmt.parseFloat(f64, row_iter.next().?);
            const vx = n.toValues(alloc, &[_]f64{x});
            const vy = n.toValues(alloc, &[_]f64{y});
            defer alloc.free(vx);
            defer alloc.free(vy);

            const pred = try layer.forward(alloc, vx);
            defer alloc.free(pred);
            batchy[i] = vy[0];
            batchyh[i] = pred[0];
            if (i >= batchsize - 1) {
                const loss = try loss_mse(alloc, batchyh, batchy);
                currLoss = loss.value;
                loss.zero_grad();
                try loss.backward();
                layer.update(lr);
                layer.detach();
                loss.deinit();
                layer.attach();
                i = 0;
                break;
            }
        }
        std.debug.print("({}) loss={d:.3}\n", .{ e, currLoss });
    }
    std.debug.print("nSteps={}\n", .{nSteps});
    layer.deinit();
}

test "nn/trainLayer" {
    try trainLayer(&std.testing.allocator);
}
