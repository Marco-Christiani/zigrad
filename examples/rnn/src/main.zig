const std = @import("std");

const zg = @import("zigrad");
const opspec = zg.opspec;
const DeviceReference = zg.DeviceReference;
const NDTensor = zg.NDTensor;

pub fn main() !void {
    var cpu = zg.device.HostDevice.init(std.heap.smp_allocator);
    defer cpu.deinit();

    const N: usize = 10;
    const examples: usize = 4;

    var battery = try Samples(examples).init(N, cpu.reference());
    defer battery.deinit();

    for (0..4) |i| {
        battery.buffer[i].data.data[i] = @floatFromInt(i);
    }

    var graph: zg.GraphManager(Rnn.Tensor) = .{
        .allocator = std.heap.smp_allocator,
        .eager_teardown = true,
    };
    defer graph.deinit();

    var optim: zg.optim.SGD(f32) = .{
        .grad_clip_max_norm = 1.0,
        .grad_clip_delta = 1e-6,
        .grad_clip_enabled = false,
        .lr = 0.01,
    };

    var rnn = try Rnn.init(
        std.heap.smp_allocator,
        cpu.reference(),
        .{
            .requires_grad = true,
            .hid_size = 128,
            .inp_size = N,
        },
    );
    defer rnn.deinit();

    const decay_factor: f32 = 0.9;
    var moving_avg: f32 = 0;
    var best_loss: f32 = std.math.inf(f32);
    const patience: usize = 3;
    var patience_counter: usize = 0;
    const min_delta: f32 = 1e-2;

    const max_epochs: usize = 200;
    for (0..max_epochs) |epoch| {
        const total_loss = try Rnn.Tensor.zeros(&.{1}, true, cpu.reference());
        for (0..examples - 1) |i| {
            const y = try rnn.forward(battery.buffer[i]);
            const loss = try zg.loss.mse_loss(f32, y, battery.buffer[i + 1], cpu.reference());
            loss.add_(total_loss) catch {};
        }

        const current_loss = total_loss.get(0) / @as(f32, @floatFromInt(examples - 1));
        std.debug.print("Epoch {d}: loss = {d:.5}\n", .{ epoch, current_loss });

        if (epoch == 0) {
            moving_avg = current_loss;
        } else {
            moving_avg = decay_factor * moving_avg + (1 - decay_factor) * current_loss;
        }

        if (moving_avg < best_loss - min_delta) {
            best_loss = moving_avg;
            patience_counter = 0;
        } else {
            patience_counter += 1;
            if (patience_counter >= patience) {
                std.debug.print("Early stopping at epoch {d}. No improvement for {d} epochs.\n", .{ epoch, patience });
                break;
            }
        }

        try graph.backward(total_loss);
        optim.step(&rnn.weights.all);
        rnn.reset();
    }
}

const Rnn = struct {
    const Tensor = zg.NDTensor(f32);

    const Output = struct {
        h: *Tensor,
        y: *Tensor,
    };

    const Weights = extern union {
        all: [2]*Tensor,
        get: extern struct {
            W1: *Tensor,
            W2: *Tensor,
        },
    };

    allocator: std.mem.Allocator,
    outputs: std.ArrayListUnmanaged(Output) = .empty,
    weights: Weights,
    pub fn init(allocator: std.mem.Allocator, device: zg.DeviceReference, config: struct {
        requires_grad: bool,
        inp_size: usize,
        hid_size: usize,
    }) !Rnn {
        var weights: Weights = .{
            .get = .{ // doesn't matter - ignoring initialization for right now
                .W1 = try Tensor.random(&.{ config.hid_size, config.inp_size }, config.requires_grad, .normal, device),
                .W2 = try Tensor.random(&.{ config.inp_size, config.hid_size }, config.requires_grad, .normal, device),
            },
        };
        for (&weights.all, '1'..) |wgt, i| {
            try wgt.set_label(&.{ 'W', @intCast(i) });
            wgt.acquire();
        }
        return .{
            .weights = weights,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Rnn) void {
        self.reset();
        self.outputs.deinit(self.allocator);
        for (&self.weights.all) |wgt| {
            wgt.release();
            wgt.deinit();
        }
    }

    pub fn forward(self: *Rnn, x: *Tensor) !*Tensor {
        const prev_h = blk: {
            const prev = self.outputs.getLastOrNull() orelse break :blk null;
            break :blk prev.h;
        };
        const out = try self.compute(x, prev_h);
        errdefer {
            out.h.deinit();
            out.y.deinit();
        }
        try self.outputs.append(self.allocator, out);
        return out.y;
    }

    // y1 = W2.ReLU(W1.x + y0)
    fn compute(self: *Rnn, x: *Tensor, h0: ?*Tensor) !Output {
        const h1 = try self.weights.get.W1.matvec(x, .{});
        errdefer h1.deinit();

        if (h0) |_h0| try _h0.add_(h1);

        try MaskedReLU(f32).forward_(h1);

        const y = try self.weights.get.W2.matvec(h1, .{});

        return .{ .h = h1, .y = y };
    }

    fn reset(self: *Rnn) void {
        if (!self.weights.get.W1.requires_grad()) {
            for (self.outputs.items) |out| {
                out.h.deinit();
                out.y.deinit();
            }
        }
        self.outputs.clearRetainingCapacity();
    }
};

pub fn MaskedReLU(comptime T: type) type {
    return struct {
        const Self = @This();
        const Tensor = NDTensor(T);
        mask: []u8, // for reverse

        pub fn forward_(x: *Tensor) !void {
            if (x.requires_grad()) {
                return x.device.dispatch(opspec.relu_fwd(T){
                    .x = x.get_data(),
                    .y = x.get_data(),
                });
            }

            const mask = try x.device.mem_alloc_byte_mask(x.get_size());
            errdefer x.device.mem_free(mask);

            x.device.dispatch(opspec.relu_mask_fwd(T){
                .x = x.get_data(),
                .mask = mask,
            });
            try Tensor.prepend_dependent(Self, x, .{
                .callback = .{ .mask = mask },
                .children = &.{},
                .device = x.device,
            });
        }

        pub fn callback(y: *Tensor, _: *Tensor.Children, ctx: *Self) !void {
            y.device.dispatch(opspec.relu_mask_bwd(T){
                .x_g = try y.ensure_grad_data(0),
                .mask = ctx.mask,
            });
            y.device.mem_free(ctx.mask);
        }
    };
}

// We're just predicting the k+1'th example from the kth example.
pub fn Samples(comptime n: usize) type {
    return struct {
        const Self = @This();
        buffer: [n]*Rnn.Tensor = undefined,
        pub fn init(len: usize, device: zg.DeviceReference) !Self {
            var self: Self = .{};
            for (0..n, '0'..) |i, c| {
                self.buffer[i] = try Rnn.Tensor.zeros(&.{len}, false, device);
                try self.buffer[i].set_label(&.{ 'x', @intCast(c) });
                self.buffer[i].acquire();
            }
            return self;
        }
        pub fn deinit(self: *Self) void {
            for (&self.buffer) |entry| {
                entry.release();
                entry.deinit();
            }
        }
    };
}
