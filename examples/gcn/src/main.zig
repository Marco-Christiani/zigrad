const std = @import("std");
const zg = @import("zigrad");

const Dataset = @import("dataset.zig").Dataset;
const GCN = @import("model.zig").GCN;
const MaskLayer = @import("model.zig").MaskLayer;

const std_options = .{ .log_level = .info };
const log = std.log.scoped(.gnn);
const T = f32;
const Optimizer = zg.optim.SGD(T);

pub fn run_cora(data_dir: []const u8) !void {
    var debug_allocator = std.heap.DebugAllocator(.{}).init;
    const allocator = debug_allocator.allocator();

    zg.init_global_graph(allocator, .{
        .eager_teardown = true,
    });
    defer zg.deinit_global_graph();

    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();

    const device = cpu.reference();

    var buf1: [1024]u8 = undefined;
    var buf2: [1024]u8 = undefined;
    const edge_path = try std.fmt.bufPrint(&buf1, "{s}/cora/csv/cites.csv", .{data_dir});
    const node_path = try std.fmt.bufPrint(&buf2, "{s}/cora/csv/papers.csv", .{data_dir});
    const dataset = try Dataset(T).load_cora(allocator, device, node_path, edge_path);
    defer dataset.deinit();

    var optim: Optimizer = .{
        .lr = 0.01,
        .grad_clip_max_norm = 10.0,
        .grad_clip_delta = 1e-6,
        .grad_clip_enabled = false,
    };

    var model = try GCN(T, Optimizer).init(
        device,
        dataset.num_features,
        dataset.num_classes,
    );
    defer model.deinit();

    const label = dataset.y;

    var total_train_time: f64 = 0;
    var total_test_time: f64 = 0;

    var timer = try std.time.Timer.start();

    const num_epoochs = 50;
    for (0..num_epoochs) |epoch| {
        var loss_val: T = 0;
        var acc = [_]f32{ 0, 0, 0 };
        var train_time_ms: f64 = 0;
        var test_time_ms: f64 = 0;
        {
            zg.rt_grad_enabled = true;
            timer.reset();

            const output = try model.forward(dataset.x, dataset.edge_index);
            defer output.deinit();

            const mask_layer = MaskLayer(T){ .mask = dataset.train_mask };
            const output_ = try mask_layer.forward(output);
            defer output_.deinit();
            const label_ = try mask_layer.forward(label);
            defer label_.deinit();
            const loss = try zg.loss.softmax_cross_entropy_loss(T, output_, label_);
            loss_val = loss.get(0);

            try loss.backward();
            try model.update(&optim);
            model.zero_grad();

            train_time_ms = @as(f64, @floatFromInt(timer.lap())) / @as(f64, @floatFromInt(std.time.ns_per_ms));
            total_train_time += train_time_ms;
        }

        {
            zg.rt_grad_enabled = false;
            timer.reset();

            const output = try model.forward(dataset.x, dataset.edge_index);
            defer output.deinit();
            for ([_]*zg.NDTensor(bool){ dataset.train_mask, dataset.eval_mask, dataset.test_mask }, 0..) |mask, i| {
                var correct: f32 = 0;
                const mask_layer = MaskLayer(T){ .mask = mask };
                const output_ = try mask_layer.forward(output);
                defer output_.deinit();
                const label_ = try mask_layer.forward(label);
                defer label_.deinit();

                const total = output_.get_dim(0);

                for (0..total) |j| {
                    const start = j * dataset.num_classes;
                    const end = start + dataset.num_classes;
                    const yh = std.mem.indexOfMax(T, output_.data.data[start..end]);
                    const y = std.mem.indexOfMax(T, label_.data.data[start..end]);
                    correct += if (yh == y) 1 else 0;
                }
                acc[i] = correct / @as(f32, @floatFromInt(total));
            }
            test_time_ms = @as(f64, @floatFromInt(timer.lap())) / @as(f64, @floatFromInt(std.time.ns_per_ms));
            total_test_time += test_time_ms;
        }
        std.debug.print(
            "Epoch: {d:>2}, Loss: {d:<5.4}, Train_acc: {d:<2.2}, Val_acc: {d:<2.2}, Test_acc: {d:<2.2}, Train_time {d:<3.2} ms, Test_time {d:<3.2} ms\n",
            .{ epoch + 1, loss_val, acc[0], acc[1], acc[2], train_time_ms, test_time_ms },
        );
    }
    std.debug.print("Avg epoch train time: {d:.2} ms, Avg epoch test time: {d:.2} ms\n", .{ total_train_time / num_epoochs, total_test_time / num_epoochs });
    std.debug.print("Total train time: {d:.2} ms, Total test time: {d:.2} ms\n", .{ total_train_time, total_test_time });
}

pub fn main() !void {
    const data_dir = std.posix.getenv("ZG_DATA_DIR") orelse "data";
    try run_cora(data_dir);
}
