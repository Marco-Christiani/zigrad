const std = @import("std");
const zg = @import("../root.zig");

const Loss = zg.tensor.Loss;
const NDTensor = zg.tensor.NDTensor;
const Model = zg.Model;
const SGD = zg.tensor.SGD;
const cross_entropy_loss = zg.loss.cross_entropy_loss;
const mse_loss = zg.loss.mse_loss;
const softmax = zg.loss.softmax;
const utils = zg.utils;

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
        params: []*const NDTensor(T),
        optimizer: SGD(T),
        graph_manager: Loss(NDTensor(T)),

        pub fn init(
            model: Model(T),
            learning_rate: T,
            loss_config: Loss(T).LossConfig,
        ) Self {
            return .{
                .model = model,
                .params = model.getParameters(),
                .optimizer = .{ .lr = learning_rate },
                .graph_manager = Loss(NDTensor(T)).init(model.allocator, loss_config),
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
            target: *const NDTensor(T),
            fwd_allocator: std.mem.Allocator,
            bwd_allocator: std.mem.Allocator,
        ) !*NDTensor(T) {
            log.debug("calling fwd...", .{});
            var output = try self.model.forward(input, fwd_allocator);
            output.logShape(null);
            target.logShape(null);

            logData("output(1): ", output.data.data);
            log.debug("softmaxing", .{});
            output = try softmax(T, output, 0, bwd_allocator);
            logData("output(2): ", output.data.data);
            logData("target: ", target.data.data);
            log.debug("calculating loss", .{});
            const loss = try lossf(T, output, target, bwd_allocator);

            self.model.zeroGrad();
            loss.grad.?.fill(1.0);
            log.debug("running backward", .{});
            try self.graph_manager.backward(loss, bwd_allocator);

            // log.info("rendering", .{});
            // try utils.renderD2(loss, utils.PrintOptions.plain, self.allocator, "/tmp/trainergraph.svg");
            // log.info("done", .{});
            // try utils.sesame("/tmp/trainergraph.svg", self.allocator);

            log.debug("updating", .{});
            self.optimizer.step(self.params);
            for (self.model.getParameters()) |param| {
                log.debug("param {?s} grad norm is {d} max: {d} min: {d}", .{
                    param.label,
                    param.grad.?.l2_norm(),
                    std.mem.max(f32, param.grad.?.data),
                    std.mem.min(f32, param.grad.?.data),
                });
            }
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
