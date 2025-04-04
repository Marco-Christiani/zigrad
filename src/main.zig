const std = @import("std");
pub const zg = @import("zigrad");
//const layer = zg.layer;
pub const std_options = zg.std_options;

///////////////////////////////

pub fn main() !void {
    const Tensor = zg.NDTensor(f32);

    var cpu = zg.device.HostDevice.init(std.heap.smp_allocator);
    defer cpu.deinit();

    var gm = zg.GraphManager(Tensor).init(std.heap.smp_allocator, .{
        .eager_teardown = true,
    });
    defer gm.deinit();

    var lyr = try LinearLayer(f32).init(cpu.reference(), 5, 5);
    defer lyr.deinit();

    ///////////////////////////////////////////////////////////////////////

    const x = try Tensor.sequence(1.0, 1.0, &.{5}, true, cpu.reference());
    defer x.deinit();

    const out = try lyr.forward(x);
    try out.setup_grad(1.0);

    try gm.backward(out);

    std.debug.print("W DATA: {any}\n\n", .{lyr.weights.get_data()});
    std.debug.print("W GRAD: {any}\n\n", .{lyr.weights.assume_grad()});

    std.debug.print("B DATA: {any}\n\n", .{lyr.bias.get_data()});
    std.debug.print("B GRAD: {any}\n\n", .{lyr.bias.assume_grad()});
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
