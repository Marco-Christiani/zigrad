const std = @import("std");

const zg = @import("zigrad");
const opspec = zg.opspec;
const DeviceReference = zg.DeviceReference;
const NDTensor = zg.NDTensor;

pub fn main() !void {
    var cpu = zg.device.HostDevice.init(std.heap.smp_allocator);
    defer cpu.deinit();

    //////////////////////////////////////////////////
    // I'm setting up a toy example where each sequence
    // given to the rnn must predict the same sequence
    // shifted over 1 to the right.
    const N: usize = 10;
    const examples: usize = 4;

    var samples = try Samples(examples).init(N, cpu.reference());
    defer samples.deinit();

    for (0..4) |i|
        samples.buffer[i].data.data[i] = @floatFromInt(i);

    var rnn = try Rnn.init(
        cpu.reference(),
        .{
            .hid_size = 128,
            .inp_size = N,
        },
    );
    defer rnn.deinit();

    try train_seq_to_seq(&rnn, .{
        .inputs = &.{samples.buffer[0..3]},
        .targets = &.{samples.buffer[1..4]},
        .total_epochs = 200,
        .print_loss_every_n = 1,
        .max_bwd_steps = null,
    });
}

const Rnn = struct {
    const Tensor = zg.NDTensor(f32);

    const Output = struct {
        h: *Tensor,
        y: *Tensor,
    };

    device: zg.DeviceReference,
    weights: extern union {
        all: [2]*Tensor,
        get: extern struct {
            W1: *Tensor,
            W2: *Tensor,
        },
    },

    pub fn init(device: zg.DeviceReference, config: struct {
        inp_size: usize,
        hid_size: usize,
    }) !Rnn {
        return .{
            .device = device,
            .weights = .{
                .get = .{
                    .W1 = try Tensor.random(&.{ config.hid_size, config.inp_size }, false, .normal, device),
                    .W2 = try Tensor.random(&.{ config.inp_size, config.hid_size }, false, .normal, device),
                },
            },
        };
    }

    pub fn deinit(self: *Rnn) void {
        for (&self.weights.all) |wgt| {
            wgt.release();
            wgt.deinit();
        }
    }

    // y1 = W2.ReLU(W1.x + y0)
    pub fn forward(self: *Rnn, x: *Tensor, h0: ?*Tensor) !Output {
        const h1 = try self.compute_hidden(x, h0);
        errdefer h1.deinit();

        const y = try self.weights.get.W2.matvec(h1, .{});

        return .{ .h = h1, .y = y };
    }

    // small optimization to make training faster - we can
    // compute just the hidden state and not the full output
    fn compute_hidden(self: *Rnn, x: *Tensor, h0: ?*Tensor) !*Tensor {
        const h1 = try self.weights.get.W1.matvec(x, .{});
        errdefer h1.deinit();

        if (h0) |_h0| try _h0.add_(h1);

        try nn.relu_(f32, h1);

        return h1;
    }
};

// TODO: Move this to the nn module - wrapping in struct for
// for keeping the syntax similar after the move..
const nn = struct {
    pub fn relu_(T: type, x: *zg.NDTensor(T)) !void {
        const Tensor = NDTensor(T);

        const BwdClosure = struct {
            mask: []u8,
            pub fn callback(y: *Tensor, _: *Tensor.Children, ctx: *@This()) !void {
                y.device.dispatch(opspec.relu_mask_bwd(T){
                    .x_g = try y.ensure_grad_data(0),
                    .mask = ctx.mask,
                });
                y.device.mem_free(ctx.mask);
            }
        };

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

        try Tensor.prepend_dependent(BwdClosure, x, .{
            .callback = .{ .mask = mask },
            .children = &.{},
            .device = x.device,
        });
    }
};

pub fn train_seq_to_seq(
    rnn: *Rnn,
    config: struct {
        inputs: []const []*Rnn.Tensor,
        targets: []const []*Rnn.Tensor,
        total_epochs: usize,
        print_loss_every_n: usize,
        max_bwd_steps: ?usize,
    },
) !void {
    if (!zg.rt_grad_enabled)
        @panic("Zigrad runtime gradients must be enabled to train models.");

    if (config.total_epochs == 0)
        @panic("Total epochs must be greater than zero to train.");

    if (config.inputs.len != config.targets.len)
        @panic("The number of inputs must match the number of targets.");

    if (config.max_bwd_steps != null and config.max_bwd_steps.? == 0)
        @panic("Max backward steps must be either null or greater than 0.");

    // start by aquiring the weights so backward won't free them (labeling is optional)
    for (&rnn.weights.all, '1'..) |wgt, i| {
        try wgt.set_label(&.{ 'W', @intCast(i) });
        wgt.acquire();
    }

    // TODO: make this part of configuration? Probably.
    const decay_factor: f32 = 0.9;
    var moving_avg: f32 = 0;
    var best_loss: f32 = std.math.inf(f32);
    const patience: usize = 3;
    var patience_counter: usize = 0;
    const min_delta: f32 = 1e-2;

    // It's important to have `eager_teardown == true` here
    // because I'm choosing to not manually track the outputs.
    var gm: zg.GraphManager(Rnn.Tensor) = .{
        .allocator = std.heap.smp_allocator,
        .eager_teardown = true,
    };
    defer gm.deinit();

    // or your favorite optimizer...
    var optim: zg.optim.SGD(f32) = .{
        .grad_clip_max_norm = 1.0,
        .grad_clip_delta = 1e-6,
        .grad_clip_enabled = false,
        .lr = 0.01,
    };

    for (0..config.total_epochs) |epoch| {
        for (config.inputs, config.targets, 0..) |inp_seq, trg_seq, example_num| {

            // Backward start step tells us where to
            // beging recording the backward graph. This
            // only applices if "requires_grad" is true.
            //
            // Example:
            //
            // seq.len = 5, max_bwd_steps = 3
            //       v
            // 0, 1, 2, 3, 4
            const bwd_start_step = if (config.max_bwd_steps) |m|
                if (inp_seq.len <= m) 0 else inp_seq.len - m
            else
                0; // keep the entire sequence

            var prev_hid: ?*Rnn.Tensor = null;

            // we're going to score the total loss for the entire sequence
            const total_loss = try Rnn.Tensor.zeros(&.{1}, true, rnn.device);

            for (inp_seq, trg_seq, 0..) |x_i, y_i, time_step| {
                // std.debug.print("bwd_start_step: {}, time_step: {}\n", .{ bwd_start_step, time_step });

                if (time_step == bwd_start_step) {
                    // set to true to start tracking the operations from this step onwards
                    for (&rnn.weights.all) |wgt| wgt._requires_grad = true;
                }

                if (time_step >= bwd_start_step) {
                    // we need the full forward to calculate loss
                    const out = try rnn.forward(x_i, prev_hid);
                    const loss = try zg.loss.mse_loss(f32, out.y, y_i, rnn.device);
                    loss.add_(total_loss) catch {};
                    prev_hid = out.h;
                } else {
                    // in this case, we only need the hidden state
                    // because we don't actually care about what
                    // the output was (it won't effect the gradient)
                    const h1 = try rnn.compute_hidden(x_i, prev_hid);
                    if (prev_hid) |ph| ph.deinit();
                    prev_hid = h1;
                }
            }

            /////////////////////////////////////
            // Decide if we need to stop training

            // TODO: This totally assumes a cpu device - not safe for GPU
            const current_loss = total_loss.get(0) / @as(f32, @floatFromInt(inp_seq.len));

            if (config.print_loss_every_n > 0 and example_num % config.print_loss_every_n == 0)
                std.debug.print("Epoch {d}: loss = {d:.5}\n", .{ epoch, current_loss });

            if (epoch == 0 and example_num == 0) {
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
                    total_loss.teardown(&gm);
                    std.debug.print("Early stopping at epoch {d}. No improvement for {d} epochs.\n", .{ epoch, patience });
                    return;
                }
            }

            // collects gradients and free intermediate state
            try gm.backward(total_loss);

            // update the weights
            optim.step(&rnn.weights.all);

            // set to false to allow truncation to occur
            for (&rnn.weights.all) |wgt|
                wgt._requires_grad = false;
        }
    }
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
