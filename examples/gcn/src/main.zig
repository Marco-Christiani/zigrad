const std = @import("std");
const zg = @import("zigrad");

const Dataset = @import("dataset.zig").Dataset;
const GCN = @import("model.zig").GCN;
const MaskLayer = @import("model.zig").MaskLayer;

const std_options = .{ .log_level = .info };
const log = std.log.scoped(.gnn);
const T = f32;
const Optimizer = zg.optim.Adam;

pub fn run_cora(data_dir: []const u8) !void {
    var debug_allocator = std.heap.DebugAllocator(.{}).init;
    const allocator = debug_allocator.allocator();
    defer {
        const leak = debug_allocator.deinit();
        if (leak == std.heap.Check.leak) {
            std.debug.print("memory leak !!\n", .{});
        }
    }

    zg.global_graph_init(allocator, .{
        .eager_teardown = true,
    });
    defer zg.global_graph_deinit();

    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();

    const device = cpu.reference();

    var buf1: [1024]u8 = undefined;
    var buf2: [1024]u8 = undefined;
    const edge_path = try std.fmt.bufPrint(&buf1, "{s}/cora/csv/cities.csv", .{data_dir});
    const node_path = try std.fmt.bufPrint(&buf2, "{s}/cora/csv/papers.csv", .{data_dir});
    const dataset = try Dataset(T).load_cora(allocator, device, node_path, edge_path);
    defer dataset.deinit();

    var adam = Optimizer.init(allocator, .{
        .lr = 0.01,
        .beta1 = 0.9,
        .beta2 = 0.999,
        .epsilon = 1e-8,
    });
    defer adam.deinit();

    const optim = adam.optimizer();

    var model = try GCN(T).init(
        device,
        dataset.num_features,
        dataset.num_classes,
        .{ .optim = optim },
    );
    defer model.deinit();

    const label = dataset.y;

    var total_train_time: f64 = 0;
    var total_test_time: f64 = 0;

    var timer = try std.time.Timer.start();

    var max_train_time = std.math.floatMin(f64);
    var max_test_time = std.math.floatMin(f64);
    const num_epochs = 50;
    for (0..num_epochs) |epoch| {
        var loss_val: T = 0;
        var acc = [_]f32{ 0, 0, 0 };
        var train_time_ms: f64 = 0;
        var test_time_ms: f64 = 0;
        {
            zg.runtime.grad_enabled = true;
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
            try optim.step();
            model.zero_grad();

            train_time_ms = @as(f64, @floatFromInt(timer.lap())) / @as(f64, @floatFromInt(std.time.ns_per_ms));
            total_train_time += train_time_ms;
            max_train_time = @max(max_train_time, train_time_ms);
        }

        {
            zg.runtime.grad_enabled = false;
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
                    // TODO: update for device compatablity
                    const yh = std.mem.indexOfMax(T, output_.get_data()[start..end]);
                    const y = std.mem.indexOfMax(T, label_.get_data()[start..end]);
                    correct += if (yh == y) 1 else 0;
                }
                acc[i] = correct / @as(f32, @floatFromInt(total));
            }
            test_time_ms = @as(f64, @floatFromInt(timer.lap())) / @as(f64, @floatFromInt(std.time.ns_per_ms));
            total_test_time += test_time_ms;
            max_test_time = @max(max_test_time, test_time_ms);
        }
        std.debug.print(
            "Epoch: {d:>2}, Loss: {d:<5.4}, Train_acc: {d:<2.2}, Val_acc: {d:<2.2}, Test_acc: {d:<2.2}, Train_time {d:<3.2} ms, Test_time {d:<3.2} ms\n",
            .{ epoch + 1, loss_val, acc[0], acc[1], acc[2], train_time_ms, test_time_ms },
        );
    }
    std.debug.print("Avg epoch train time: {d:.2} ms, Avg epoch test time: {d:.2} ms\n", .{ total_train_time / num_epochs, total_test_time / num_epochs });
    std.debug.print("Total train time: {d:.2} ms, Total test time: {d:.2} ms\n", .{ total_train_time, total_test_time });

    const total_train_time_ms_trimmed = total_train_time - max_train_time;
    const total_test_time_ms_trimmed = total_test_time - max_test_time;
    std.debug.print("(trimmed) Avg epoch train time: {d:.2} ms, Avg epoch test time: {d:.2} ms\n", .{
        total_train_time_ms_trimmed / (num_epochs - 1),
        total_test_time_ms_trimmed / (num_epochs - 1),
    });
    std.debug.print("(trimmed) Total train time: {d:.2} ms, Total test time: {d:.2} ms\n", .{
        total_train_time_ms_trimmed,
        total_test_time_ms_trimmed,
    });

    const json_obj = try std.json.stringifyAlloc(allocator, .{
        .avg_epoch_train_fbs_ms = total_train_time / num_epochs,
        .avg_epoch_test_fbs_ms = total_test_time / num_epochs,
        .total_train_fbs_ms = total_train_time,
        .total_test_fbs_ms = total_test_time,
        .avg_epoch_train_fbs_trimmed_ms = total_train_time_ms_trimmed / (num_epochs - 1),
        .avg_epoch_test_fbs_trimmed_ms = total_test_time_ms_trimmed / (num_epochs - 1),
        .total_train_fbs_trimmed_ms = total_train_time_ms_trimmed,
        .total_test_fbs_trimmed_ms = total_test_time_ms_trimmed,
    }, .{ .whitespace = .indent_2 });
    defer allocator.free(json_obj);
    const stdout = std.io.getStdOut().writer();
    try stdout.print("{s}\n", .{json_obj});
}

pub fn main() !void {
    const data_dir = std.posix.getenv("ZG_DATA_DIR") orelse "data";
    try run_cora(data_dir);
}
