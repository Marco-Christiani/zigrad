const std = @import("std");

const zg = @import("zigrad");
const opspec = zg.opspec;
const DeviceReference = zg.DeviceReference;
const NDTensor = zg.NDTensor;

pub fn main() !void {
    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();

    var graph = zg.GraphManager.init(std.heap.smp_allocator, .{
        .eager_teardown = true,
    });
    defer graph.deinit();

    //////////////////////////////////////////////////
    // I'm setting up a toy example where each sequence
    // given to the rnn must predict the same sequence
    // shifted over 1 to the right.
    const N: usize = 10;
    const examples: usize = 4;

    var samples = try Samples(examples).init(N, &graph, cpu.reference());
    defer samples.deinit();

    for (0..4) |i|
        samples.buffer[i].data.data[i] = @floatFromInt(i);

    var gru = try Gru.init(
        &graph,
        cpu.reference(),
        .{
            .requires_grad = true,
            .hid_size = N,
            .inp_size = N,
        },
    );
    defer gru.deinit();

    var decoder = try Decoder.init(
        &graph,
        cpu.reference(),
        .{
            .requires_grad = true,
            .inp_size = N,
            .hid_size = 128,
        },
    );
    defer decoder.deinit();

    try train_seq_to_seq(
        &graph,
        cpu.reference(),
        &gru,
        &decoder,
        .{
            .inputs = &.{samples.buffer[0..3]},
            .targets = &.{samples.buffer[1..4]},
            .total_epochs = 200,
            .print_loss_every_n = 1,
            .max_bwd_steps = null,
        },
    );
}

const Decoder = struct {
    const Tensor = zg.NDTensor(f32);

    weights: extern union {
        all: [2]*Tensor,
        get: extern struct {
            Wh: *Tensor,
            Wy: *Tensor,
        },
    },

    pub fn init(graph: *zg.GraphManager, device: zg.DeviceReference, config: struct {
        requires_grad: bool,
        inp_size: usize,
        hid_size: usize,
    }) !Decoder {
        const wgts_config: zg.TensorConfig = .{
            .requires_grad = config.requires_grad,
            .device = device,
            .node_allocator = graph.heap(),
            .acquired = true,
        };
        return .{
            .weights = .{
                .get = .{
                    .Wh = try Tensor.random(&.{ config.hid_size, config.inp_size }, .normal, wgts_config),
                    .Wy = try Tensor.random(&.{ config.inp_size, config.hid_size }, .normal, wgts_config),
                },
            },
        };
    }

    pub fn deinit(self: *Decoder) void {
        for (&self.weights.all) |wgt| {
            wgt.release();
            wgt.deinit();
        }
    }

    pub fn forward(self: *Decoder, x: *Tensor) !*Tensor {
        const h = try self.weights.get.Wh.matvec(x, .{});
        errdefer h.deinit();

        try zg.nn.tanh_(f32, h);

        const y = try self.weights.get.Wy.matvec(h, .{});

        if (!h.requires_grad())
            h.deinit();

        return y;
    }
};

// Let's explain what problem this object is trying
// to solve. Let's observe that on the forward pass,
// we do not need intermediate values - we only need
// the output. The problem though comes from double
// freeing... observe:
//
//////////////////////
// Case 1: defer-deinit
//
// x = try a.op(b)
// defer x.deinit(); // Fails for training - destroys backwards child.
//
//////////////////////////
// Case 3: errdefer-deinit
//
// y = try W1.op(x);
// errdefer x.deinit(); // fine for reverse, no error...
//
// z = try W2.op(y);
//
// if (!x.requires_grad())
//   x.deinit(); // free x because it's no-longer used
//
// try op_(z); // Whoops - if this fails, x gets freed twice.
//
/////////////////////////////////////
// Case 3: errdefer-deinit-eviscerate
//
// y = try W1.op(x);
// errdefer x.deinit(); // fine for reverse, no error...
//
// z = try W2.op(y);
//
// if (!x.requires_grad())
//   x.eviscerate(); // free x's *data* because it's no-longer used
//
// // Whoops - didn't free x's pointer.
//
//////////////////////////////////////
// Thus, this simple struct helps us decide when
// it's right to deinit a tensor. Only deinit
// if there was an error OR gradient not required.
const DeinitManager = struct {
    had_error: bool = false,
    pub fn handle(mgr: @This(), x: anytype) void {
        if (mgr.had_error or !x.requires_grad()) x.deinit();
    }
};

const Gru = struct {
    const Tensor = zg.NDTensor(f32);

    weights: extern union {
        all: [9]*Tensor,
        get: extern struct {
            // update gate
            Wz: *Tensor,
            Uz: *Tensor,
            bz: *Tensor,
            // reset gate
            Wr: *Tensor,
            Ur: *Tensor,
            br: *Tensor,
            // hidden state
            Wh: *Tensor,
            Uh: *Tensor,
            bh: *Tensor,
        },
    },

    pub fn init(graph: *zg.GraphManager, device: zg.DeviceReference, config: struct {
        requires_grad: bool,
        inp_size: usize,
        hid_size: usize,
    }) !Gru {
        const wgts_config: zg.TensorConfig = .{
            .requires_grad = config.requires_grad,
            .device = device,
            .node_allocator = graph.heap(),
            .acquired = true,
        };
        return .{
            .weights = .{
                .get = .{
                    .Wz = try Tensor.random(&.{ config.hid_size, config.inp_size }, .normal, wgts_config),
                    .Uz = try Tensor.random(&.{ config.hid_size, config.inp_size }, .normal, wgts_config),
                    .bz = try Tensor.zeros(&.{config.inp_size}, wgts_config),
                    .Wr = try Tensor.random(&.{ config.hid_size, config.inp_size }, .normal, wgts_config),
                    .Ur = try Tensor.random(&.{ config.hid_size, config.inp_size }, .normal, wgts_config),
                    .br = try Tensor.zeros(&.{config.inp_size}, wgts_config),
                    .Wh = try Tensor.random(&.{ config.hid_size, config.inp_size }, .normal, wgts_config),
                    .Uh = try Tensor.random(&.{ config.hid_size, config.inp_size }, .normal, wgts_config),
                    .bh = try Tensor.zeros(&.{config.inp_size}, wgts_config),
                },
            },
        };
    }

    pub fn deinit(self: *Gru) void {
        for (&self.weights.all) |wgt| {
            wgt.release();
            wgt.deinit();
        }
    }

    pub fn forward(self: *Gru, x0: *Tensor, h0: *Tensor) !*Tensor {
        ////////////////////////////////////////////
        // update gate /////////////////////////////

        const z1 = try self.weights.get.Wz.matvec(x0, .{});
        errdefer z1.deinit();

        try self.weights.get.Uz.matvec_(h0, z1, .{ .beta = 1.0 });
        try z1._add(self.weights.get.bz);
        try zg.nn.sigm_(f32, z1);

        ////////////////////////////////////////////
        // reset gate //////////////////////////////

        const r1 = try self.weights.get.Wz.matvec(x0, .{});
        errdefer r1.deinit();

        try self.weights.get.Ur.matvec_(h0, r1, .{ .beta = 1.0 });
        try r1._add(self.weights.get.br);
        try zg.nn.sigm_(f32, r1);

        ////////////////////////////////////////////
        // activation candidate ////////////////////

        const ac_mul = try r1.mul(h0);
        errdefer ac_mul.deinit();

        if (!r1.requires_grad())
            r1.clear();

        const ac = try self.weights.get.Wh.matvec(x0, .{});
        errdefer ac.deinit();

        try self.weights.get.Uh.matvec_(ac_mul, ac, .{ .beta = 1.0 });

        if (!ac_mul.requires_grad())
            ac_mul.clear();

        try ac._add(self.weights.get.bh);
        try zg.nn.tanh_(f32, ac);

        ////////////////////////////////////////////
        // hidden state ////////////////////////////

        const sub_one = try z1.sub_scalar(1);
        errdefer sub_one.deinit();

        const h_mul_a = try z1.mul(ac);
        errdefer h_mul_a.deinit();

        if (!z1.requires_grad())
            z1.clear();

        const h_mul_b = try sub_one.mul(h0);
        errdefer h_mul_b.clear();

        if (!sub_one.requires_grad())
            sub_one.clear();

        const h1 = try h_mul_a.sub(h_mul_b);

        if (!h1.requires_grad()) {
            h_mul_a.deinit();
            h_mul_b.deinit();
        }
        return h1;
    }
};

pub fn train_seq_to_seq(
    graph: *zg.GraphManager,
    device: zg.DeviceReference,
    gru: *Gru,
    decoder: *Decoder,
    config: struct {
        inputs: []const []*Gru.Tensor,
        targets: []const []*Gru.Tensor,
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

    // TODO: make this part of configuration? Probably.
    const decay_factor: f32 = 0.9;
    var moving_avg: f32 = 0;
    var best_loss: f32 = std.math.inf(f32);
    const patience: usize = 3;
    var patience_counter: usize = 0;
    const min_delta: f32 = 1e-2;

    // It's important to have `eager_teardown == true` here
    // because I'm choosing to not manually track the outputs.
    graph.eager_teardown = true;

    // or your favorite optimizer...
    var optim: zg.optim.SGD(f32) = .{
        .grad_clip_max_norm = 10.0,
        .grad_clip_delta = 1e-6,
        .grad_clip_enabled = true,
        .lr = 0.01,
    };

    // assumes zeros...
    // cloning takes the properties of the tensor being
    // being cloned so this tensor is "acquired"
    const initial_hid = try gru.weights.get.bh.clone();
    defer {
        initial_hid.release();
        initial_hid.deinit();
    }

    initial_hid.disable_grad();

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

            var prev_hid: *Gru.Tensor = initial_hid;

            // we're going to score the total loss for the entire sequence
            const total_loss = try Gru.Tensor.zeros(&.{1}, .{
                .requires_grad = true,
                .node_allocator = graph.heap(),
                .device = device,
            });
            defer total_loss.deinit();

            for (inp_seq, trg_seq, 0..) |x_i, y_i, time_step| {
                // std.debug.print("bwd_start_step: {}, time_step: {}\n", .{ bwd_start_step, time_step });
                if (time_step >= bwd_start_step) {
                    // we need the full forward to calculate loss
                    const hid = try gru.forward(x_i, prev_hid);
                    const pred = try decoder.forward(hid);
                    const loss = try zg.loss.mse_loss(f32, pred, y_i);
                    loss.add_(total_loss) catch {};
                    prev_hid = hid;
                } else {
                    // in this case, we only need the hidden state
                    // because we don't actually care about what
                    // the output was (it won't effect the gradient)
                    const hid = try gru.forward(x_i, prev_hid);
                    if (example_num > 0) prev_hid.deinit();
                    prev_hid = hid;
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
                    graph.teardown(total_loss);
                    std.debug.print("Early stopping at epoch {d}. No improvement for {d} epochs.\n", .{ epoch, patience });
                    return;
                }
            }

            // collects gradients and free intermediate state
            try graph.backward(total_loss);

            // update the weights
            //optim.step(&gru.weights.all);
            optim.step(&decoder.weights.all);

            // clear the computation graph
            graph.reset();
        }
    }
}

// We're just predicting the k+1'th example from the kth example.
pub fn Samples(comptime n: usize) type {
    return struct {
        const Self = @This();
        buffer: [n]*Gru.Tensor = undefined,
        pub fn init(len: usize, graph: *zg.GraphManager, device: zg.DeviceReference) !Self {
            var self: Self = .{};
            for (0..n) |i| {
                self.buffer[i] = try Gru.Tensor.zeros(&.{len}, .{
                    .requires_grad = false,
                    .node_allocator = graph.heap(),
                    .device = device,
                    .acquired = true,
                });
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
