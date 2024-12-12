const tracy = @import("tracy");
const std = @import("std");
const zg = @import("../zigrad.zig");

const GraphManager = zg.GraphManager;
const NDTensor = zg.NDTensor;
const Model = zg.Model;
const SGD = zg.optim.SGD;
const cross_entropy_loss = zg.loss.softmax_cross_entropy_loss;
const mse_loss = zg.loss.mse_loss;
const softmax = zg.loss.softmax;

const log = std.log.scoped(.zg_trainer);

pub const LossFns = enum { mse, ce };

pub fn Trainer(comptime T: type, comptime loss_fn: LossFns) type {
    const lossf = switch (loss_fn) {
        .ce => cross_entropy_loss,
        .mse => mse_loss,
    };

    return struct {
        const Self = @This();
        model: Model(T),
        params: []*NDTensor(T),
        optimizer: SGD(T),
        graph_manager: GraphManager(NDTensor(T)),

        pub fn init(
            model: Model(T),
            optimizer: SGD(T),
            graph_config: GraphManager(T).GraphOpts,
        ) Self {
            return .{
                .model = model,
                .params = model.getParameters(),
                .optimizer = optimizer,
                .graph_manager = GraphManager(NDTensor(T)).init(model.allocator, graph_config),
            };
        }

        pub fn deinit(self: *Self) void {
            log.info("deinit gm", .{});
            self.graph_manager.deinit();
            log.info("free params ref", .{});
            self.model.allocator.free(self.params);
            self.* = undefined;
        }

        pub fn trainStep(
            self: *Self,
            input: *NDTensor(T),
            target: *NDTensor(T),
            fwd_allocator: std.mem.Allocator,
            bwd_allocator: std.mem.Allocator,
        ) !*NDTensor(T) {
            const zone = tracy.initZone(@src(), .{ .name = "trainer/train_step" });
            defer zone.deinit();
            const fwd_zone = tracy.initZone(@src(), .{ .name = "trainer/fwd" });
            const output = try self.model.forward(input, fwd_allocator);
            fwd_zone.deinit();
            const loss_zone = tracy.initZone(@src(), .{ .name = "trainer/loss" });
            const loss = try lossf(T, output, target, fwd_allocator);
            loss_zone.deinit();
            self.model.zeroGrad();
            loss.grad.?.fill(1.0);
            const bw_zone = tracy.initZone(@src(), .{ .name = "trainer/bw" });
            try self.graph_manager.backward(loss, bwd_allocator);
            bw_zone.deinit();
            const step_zone = tracy.initZone(@src(), .{ .name = "trainer/step" });
            self.optimizer.step(self.params);
            step_zone.deinit();
            return loss;
        }

        pub fn train(self: Self, inputs: []NDTensor(T), targets: []NDTensor(T), epochs: usize) !void {
            _ = epochs; // autofix
            _ = targets; // autofix
            _ = inputs; // autofix
            _ = self; // autofix

            // for (0..epochs) |epoch| {
            //     var total_loss: T = 0;
            //     for (inputs, targets) |input, target| {
            //         const loss = try self.trainStep(input, target);
            //         total_loss += loss;
            //     }
            //     const avg_loss = total_loss / @as(T, @floatFromInt(inputs.len));
            //     log.info("Epoch {d}: Avg Loss = {d:.4}", .{ epoch + 1, avg_loss });
            // }
        }

        fn logData(msg: []const u8, data: []T) void {
            var i: usize = 0;
            const max_n = @min(data.len, 20);
            while (i < max_n) : (i += 10) {
                log.debug("{s}({d}){d}\n", .{ msg, data.len, data });
            }
        }
    };
}
