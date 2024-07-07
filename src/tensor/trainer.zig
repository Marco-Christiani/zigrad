const std = @import("std");
const Loss = @import("tensor.zig").Loss;
const Model = @import("model.zig").Model;
const NDTensor = @import("tensor.zig").NDTensor;
const SGD = @import("tensor.zig").SGD;
const ops = @import("ops.zig");
const utils = @import("utils.zig");

const log = std.log.scoped(.zg_trainer);

pub const LossFns = enum { mse, ce };

pub fn Trainer(comptime T: type, comptime loss_fn: LossFns) type {
    const lossf = switch (loss_fn) {
        .ce => ops.cross_entropy_loss,
        .mse => ops.mse_loss,
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
            const output = try self.model.forward(input, fwd_allocator);
            // log.info("softmaxing", .{});
            // TODO: softmax
            // output = try ops.simple_softmax(T, output, backward_allocator);
            // log.debug("calculating loss", .{});
            const loss = try lossf(T, output, target, bwd_allocator);
            // _ = lossf;
            // const loss = try ops.simple_mse_loss(T, output, target, self.allocator);

            self.model.zeroGrad();
            loss.grad.?.fill(1);
            log.debug("running backward", .{});
            try self.graph_manager.backward(loss, bwd_allocator);

            // log.info("rendering", .{});
            // try utils.renderD2(loss, utils.PrintOptions.plain, self.allocator, "/tmp/trainergraph.svg");
            // log.info("done", .{});
            // try utils.sesame("/tmp/trainergraph.svg", self.allocator);

            log.debug("updating", .{});
            self.optimizer.step(self.params);
            // for (self.model.getParameters()) |param| {
            //     param.print();
            //     std.debug.print("\n\n", .{});
            // }
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
    };
}
