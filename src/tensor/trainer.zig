const std = @import("std");
const Loss = @import("tensor.zig").Loss;
const Model = @import("model.zig").Model;
const NDTensor = @import("tensor.zig").NDTensor;
const SGD = @import("tensor.zig").SGD;
const ops = @import("ops.zig");
const utils = @import("utils.zig");

const log = std.log.scoped(.zigrad_trainer);

// NOTE: temporarily hardcoded
// const loss_fn = ops.simple_mse_loss;
pub const LossFns = enum { mse, ce };

pub fn Trainer(comptime T: type, comptime loss_fn: LossFns) type {
    const lossf = switch (loss_fn) {
        .ce => ops.cross_entropy_loss,
        .mse => ops.mse_loss,
    };
    return struct {
        const Self = @This();
        model: *Model(T),
        params: []*NDTensor(T),
        optimizer: SGD(T),
        allocator: std.mem.Allocator,
        graph_manager: Loss(NDTensor(T)),

        pub fn init(
            model: *Model(T),
            learning_rate: T,
            allocator: std.mem.Allocator,
        ) Self {
            return .{
                .model = model,
                .params = model.getParameters(),
                .optimizer = .{ .lr = learning_rate },
                .allocator = allocator,
                .graph_manager = Loss(NDTensor(T)).init(allocator, .{}),
            };
        }

        pub fn deinit(self: *Self) void {
            log.info("deinit gm", .{});
            self.graph_manager.deinit();
            log.info("free params", .{});
            self.allocator.free(self.params);
        }

        pub fn trainStep(self: *Self, input: *NDTensor(T), target: *NDTensor(T)) !*NDTensor(T) {
            const output = try self.model.forward(input);
            std.log.info("calculating loss", .{});
            const loss = try lossf(T, output, target, self.allocator);

            // std.log.info("rendering", .{});
            // try utils.renderD2(loss, utils.PrintOptions.plain, self.allocator, "/tmp/trainergraph.png");
            // std.log.info("done", .{});
            // try utils.sesame("/tmp/trainergraph.png", self.allocator);

            self.model.zeroGrad();
            loss.grad.?.fill(1);
            std.log.info("running backward", .{});
            try self.graph_manager.backward(loss, self.allocator);

            std.log.info("updating", .{});
            self.optimizer.step(self.params);
            return loss;
        }

        pub fn train(self: *Self, inputs: []*NDTensor(T), targets: []*NDTensor(T), epochs: usize) !void {
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
