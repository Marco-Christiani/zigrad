const std = @import("std");
const zg = @import("zigrad");
const MnistModel = @import("model.zig").MnistModel;
const MnistDataset = @import("dataset.zig").MnistDataset;

const std_options = .{ .log_level = .info };
const log = std.log.scoped(.mnist);
const T = f32;

pub fn runMnist(train_path: []const u8, test_path: []const u8) !void {
    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);
    const stdout = bw.writer();

    // Model and dataset memory can be managed separately
    var acquired_arena = std.heap.ArenaAllocator.init(std.heap.raw_c_allocator);
    defer acquired_arena.deinit();

    // Forward pass allocator
    var fw_arena = std.heap.ArenaAllocator.init(std.heap.raw_c_allocator);
    defer fw_arena.deinit();

    // Backward pass allocator
    var bw_arena = std.heap.ArenaAllocator.init(std.heap.raw_c_allocator);
    defer bw_arena.deinit();

    var model = try MnistModel(T).init(acquired_arena.allocator()); // 109_386
    try stdout.print("n params = {}\n", .{model.countParams()});

    try stdout.print("Loading train data...\n", .{});
    const batch_size = 64;
    const train_dataset = try MnistDataset(T).load(acquired_arena.allocator(), train_path, batch_size);

    var trainer = zg.Trainer(T, .ce).init(
        model.model,
        zg.optim.SGD(T){
            .lr = 0.1,
            .grad_clip_max_norm = 10.0,
            .grad_clip_delta = 1e-6,
            .grad_clip_enabled = false,
        },
        .{},
    );

    defer {
        trainer.deinit();
        model.deinit();
    }

    var step_timer = try std.time.Timer.start();
    var timer = try std.time.Timer.start();
    try stdout.print("Training...\n", .{});
    const num_epochs = 3;
    for (0..num_epochs) |epoch| {
        var total_loss: f64 = 0;
        for (train_dataset.images, train_dataset.labels, 0..) |image, label, i| {
            step_timer.reset();
            const loss = try trainer.trainStep(
                image.setLabel("image_batch"),
                label.setLabel("label_batch"),
                fw_arena.allocator(),
                bw_arena.allocator(),
            );
            const t1 = @as(f64, @floatFromInt(step_timer.read()));
            const ms_per_sample = t1 / @as(f64, @floatFromInt(std.time.ns_per_ms * batch_size));
            total_loss += loss.get(&[_]usize{0});
            log.info("train_loss: {d:<5.5} [{d}/{d}] [ms/sample: {d}]\n", .{
                loss.data.data[0],
                i,
                train_dataset.images.len,
                ms_per_sample,
            });
            // Optional: Render an svg of the traced graph
            // if (epoch == 0 and i == 0) {
            //     try zg.utils.renderD2(loss, zg.utils.PrintOptions.plain, fw_arena.allocator(), "./docs/comp_graph_mnist.svg");
            // }

            if (!bw_arena.reset(.retain_capacity)) log.err("Issue in bw arena reset", .{});
            if (!fw_arena.reset(.retain_capacity)) log.err("Issue in fw arena reset", .{});
        }
        const avg_loss = total_loss / @as(f32, @floatFromInt(train_dataset.images.len));
        try stdout.print("Epoch {d}: Avg Loss = {d:.4}\n", .{ epoch + 1, avg_loss });
    }
    if (!bw_arena.reset(.free_all)) log.err("Issue in fw arena reset", .{});
    const train_time_ms = @as(f64, @floatFromInt(timer.lap())) / @as(f64, @floatFromInt(std.time.ns_per_ms));
    try stdout.print("Training complete ({d} epochs). [{d}ms]\n", .{ num_epochs, train_time_ms });

    const train_eval = try evalMnist(fw_arena.allocator(), model, train_dataset);
    const train_acc = train_eval.correct / @as(f32, @floatFromInt(train_eval.n));
    const eval_train_time_ms = @as(f64, @floatFromInt(timer.lap())) / @as(f64, @floatFromInt(std.time.ns_per_ms));
    try stdout.print("Train acc: {d:.2} (n={d}) [{d}ms]\n", .{ train_acc * 100, train_eval.n, eval_train_time_ms });
    train_dataset.deinit();

    try stdout.print("Loading test data...\n", .{});
    const test_dataset = try MnistDataset(T).load(acquired_arena.allocator(), test_path, batch_size);
    defer test_dataset.deinit();
    timer.reset();
    const test_eval = try evalMnist(fw_arena.allocator(), model, test_dataset);
    const eval_test_time_ms = @as(f64, @floatFromInt(timer.lap())) / @as(f64, @floatFromInt(std.time.ns_per_ms));
    const test_acc = test_eval.correct / @as(f32, @floatFromInt(test_eval.n));
    try stdout.print("Test acc: {d:.2} (n={d}) [{d}ms]\n", .{ test_acc * 100, test_eval.n, eval_test_time_ms });
    try stdout.print("Train: {d}ms\n", .{train_time_ms});
    try stdout.print("Eval train: {d}ms\n", .{eval_train_time_ms});
    try stdout.print("Eval test: {d}ms\n", .{eval_test_time_ms});
    try bw.flush();
}

fn evalMnist(allocator: std.mem.Allocator, model: MnistModel(T), dataset: MnistDataset(T)) !struct { correct: f32, n: u32 } {
    zg.rt_grad_enabled = false; // disable gradient tracking
    var n: u32 = 0;
    var correct: f32 = 0;
    var timer = try std.time.Timer.start();
    for (dataset.images, dataset.labels) |image, label| {
        const output = try model.model.forward(image, allocator);
        defer output.deinit();
        const batch_n = try output.data.shape.get(0);
        for (0..batch_n) |j| {
            const start = j * 10;
            const end = start + 10;
            const yh = std.mem.indexOfMax(T, output.data.data[start..end]);
            const y = std.mem.indexOfMax(T, label.data.data[start..end]);
            correct += if (yh == y) 1 else 0;
            n += 1;
        }
        const t1 = @as(f64, @floatFromInt(timer.read()));
        const ms_per_sample = t1 / @as(f64, @floatFromInt(std.time.ns_per_ms * batch_n));
        log.info("ms/sample: {d}", .{ms_per_sample});
    }
    return .{ .correct = correct, .n = n };
}

pub fn main() !void {
    try runMnist("/tmp/zigrad_test_mnist_train_full.csv", "/tmp/zigrad_test_mnist_test_full.csv");
}

test runMnist {
    runMnist("/tmp/zigrad_test_mnist_train_small.csv", "/tmp/zigrad_test_mnist_test_small.csv") catch |err| switch (err) {
        std.fs.File.OpenError.FileNotFound => std.log.warn("{s} error opening test file. Skipping `runMnist` test.", .{@errorName(err)}),
        else => return err,
    };
}
