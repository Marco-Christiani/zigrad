const std = @import("std");
const Loss = @import("tensor.zig").Loss;
const Model = @import("model.zig").Model;
const NDTensor = @import("tensor.zig").NDTensor;
const SGD = @import("tensor.zig").SGD;
const ops = @import("ops.zig");

// pub fn Trainer(comptime T: type, loss_fn: *const fn (type, *NDTensor(T), *NDTensor(T), std.mem.Allocator) anyerror!*NDTensor(T)) type {
const loss_fn = ops.mse_loss;
pub fn Trainer(comptime T: type) type {
    return struct {
        const Self = @This();
        model: *Model(T),
        optimizer: SGD(T),
        allocator: std.mem.Allocator,
        graph_manager: *Loss(NDTensor(T)),

        pub fn init(model: *Model(T), learning_rate: T, allocator: std.mem.Allocator) Self {
            return .{
                .model = model,
                .optimizer = .{ .lr = learning_rate },
                .allocator = allocator,
                .graph_manager = Loss(NDTensor(T)).init(allocator),
            };
        }

        pub fn deinit(self: *Self) void {
            self.graph_manager.deinit();
        }

        pub fn trainStep(self: *Self, input: *NDTensor(T), target: *NDTensor(T)) !T {
            const output = try self.model.forward(input);
            const loss = try loss_fn(T, output, target, self.allocator);

            loss.grad.?.fill(1);
            try self.graph_manager.backward(loss, self.allocator);

            const params = self.model.getParameters();
            self.optimizer.step(params);
            self.model.zeroGrad();

            return loss.data.data[0];
        }

        pub fn train(self: *Self, inputs: []*NDTensor(T), targets: []*NDTensor(T), epochs: usize) !void {
            for (0..epochs) |epoch| {
                var total_loss: T = 0;
                for (inputs, targets) |input, target| {
                    const loss = try self.trainStep(input, target);
                    total_loss += loss;
                }
                const avg_loss = total_loss / @as(T, @floatFromInt(inputs.len));
                std.debug.print("Epoch {d}: Avg Loss = {d:.4}\n", .{ epoch + 1, avg_loss });
            }
        }
    };
}
