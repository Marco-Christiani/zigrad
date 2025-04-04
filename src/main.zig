const std = @import("std");
pub const zg = @import("zigrad");
//const layer = zg.layer;
pub const std_options = zg.std_options;

///////////////////////////////

pub fn main() !void {
    const Tensor = zg.NDTensor(f32);

    var optim = zg.optim.SGD(f32){
        .lr = 0.1,
    };

    var cpu = zg.device.HostDevice.init(std.heap.smp_allocator);
    defer cpu.deinit();

    var gm = zg.GraphManager(Tensor).init(std.heap.smp_allocator, .{
        .eager_teardown = false,
    });
    defer gm.deinit();

    ///////////////////////////////////////////////////////////////////////

    const x = try Tensor.ones(&.{10}, true, cpu.reference());
    defer x.deinit();

    const y = try Tensor.zeros(&.{10}, false, cpu.reference());
    defer y.deinit();

    try optim.attach(x);

    y.set(0, 1);

    for (0..100) |_| {
        const loss = try zg.loss.softmax_cross_entropy_loss(f32, x, y);
        std.debug.print("LOSS: {}\n\n", .{loss.get(0)});
        try gm.backward(loss);
    }

    std.debug.print("X GRAD: {any}\n\n", .{x.assume_grad_data()});

    /////////////////
    //var ts: [5]*Tensor = undefined;

    //for (0..ts.len) |i| {
    //    ts[i] = try Tensor.sequence(@floatFromInt(i + 1), 1.0, &.{10}, true, cpu.reference());
    //    ts[i].acquire();
    //}
    //defer for (&ts) |t| {
    //    t.release();
    //    t.deinit();
    //};

    //var out = x;
    //std.debug.print("DATA: {any}\n", .{out.get_data()});
    //for (&ts) |t| {
    //    out = try t.mul(out);
    //    std.debug.print("DATA: {any}\n", .{out.get_data()});
    //}
    //try out.setup_grad(1);

    //std.debug.print("\n\n", .{});

    //for (&ts) |t| {
    //    std.debug.print("GRAD: {any}\n", .{t.assume_grad_data()});
    //}

    //try mnist.main();
}

pub fn LinearLayer(comptime T: type) type {
    return struct {
        const Self = @This();
        const Tensor = zg.NDTensor(T);
        weights: *Tensor,
        bias: *Tensor,
        pub fn init(device: zg.DeviceReference, in_features: usize, out_features: usize) !Self {
            var weights = try Tensor.ones(&.{ out_features, in_features }, true, device);
            errdefer weights.deinit();

            var bias = try Tensor.zeros(&.{ 1, out_features }, true, device);
            errdefer bias.deinit();

            try weights.set_label("linear_weights");
            try bias.set_label("linear_bias");

            weights.acquire();
            bias.acquire();

            // winit.he_init(T, weights);
            //bias.fill(0.0);

            return .{ .weights = weights, .bias = bias };
        }

        pub fn deinit(self: *Self) void {
            self.weights.release();
            self.weights.deinit();

            self.bias.release();
            self.bias.deinit();
        }

        pub fn forward(self: *const Self, x: *Tensor) !*Tensor {
            const batch_size = if (x.data.shape.len > 1) x.get_dim(0) else 1;
            const n_features = self.weights.data.shape.get(0);
            const b_y = try x.bmm_acc(self.weights, &.{ batch_size, n_features }, .{ .trans_b = true });
            return self.bias.add(b_y);
        }
    };
}
