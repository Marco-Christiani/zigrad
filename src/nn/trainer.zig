const std = @import("std");
const zg = @import("../zigrad.zig");

const DeviceReference = zg.DeviceReference;
const GraphManager = zg.GraphManager;
const NDTensor = zg.NDTensor;
const Model = zg.Model;
const SGD = zg.optim.SGD;
const cross_entropy_loss = zg.loss.softmax_cross_entropy_loss;
const mse_loss = zg.loss.mse_loss;
const softmax = zg.loss.softmax;

const log = std.log.scoped(.zg_trainer);

pub const LossFns = enum { mse, ce };

//pub fn Trainer(comptime T: type, comptime loss_fn: LossFns) type {
//    const lossf = switch (loss_fn) {
//        .ce => cross_entropy_loss,
//        .mse => mse_loss,
//    };
//
//    return struct {
//        const Self = @This();
//        model: Model(T),
//        params: []*NDTensor(T),
//        graph_manager: GraphManager(NDTensor(T)),
//
//        pub fn init(model: Model(T), graph_config: GraphManager(T).GraphOpts) Self {
//            return .{
//                .model = model,
//                .graph_manager = GraphManager(NDTensor(T)).init(model.device.allocator, graph_config),
//            };
//        }
//
//        pub fn deinit(self: *Self) void {
//            log.info("deinit gm", .{});
//            self.graph_manager.deinit();
//            log.info("free params ref", .{});
//            self.model.device.allocator.free(self.params);
//            self.* = undefined;
//        }
//
//        pub fn train_step(
//            self: *Self,
//            input: *NDTensor(T),
//            target: *NDTensor(T),
//        ) !*NDTensor(T) {
//            const output = try self.model.forward(input);
//            const loss = try lossf(T, output, target);
//            loss.acquire();
//            try loss.setup_grad(1);
//            try self.graph_manager.backward(loss);
//            loss.release();
//            return loss;
//        }
//
//        fn log_data(msg: []const u8, data: []T) void {
//            var i: usize = 0;
//            const max_n = @min(data.len, 20);
//            while (i < max_n) : (i += 10) {
//                log.debug("{s}({d}){d}\n", .{ msg, data.len, data });
//            }
//        }
//    };
//}
