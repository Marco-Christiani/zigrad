/// Trains a neural network model on the MNIST dataset using a manual training loop.
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

    // Intermediate arena allocator, for forward/backward passes
    var ima = std.heap.ArenaAllocator.init(std.heap.raw_c_allocator);
    defer ima.deinit();

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

    // Train -------------------------------------------------------------------
    try stdout.print("Training...\n", .{});
    const num_epochs = 3;
    for (0..num_epochs) |epoch| {
        var total_loss: f64 = 0;
        for (train_dataset.images, train_dataset.labels, 0..) |image, label, i| {
            const loss = try trainer.trainStep(
                image.setLabel("image_batch"),
                label.setLabel("label_batch"),
                ima.allocator(),
                ima.allocator(),
            );
            total_loss += loss.get(&[_]usize{0});
            log.info("train_loss: {d:<5.5} [{d}/{d}] [ms/sample: {d}]\n", .{
                loss.data.data[0],
                i,
                train_dataset.images.len,
            });
            // Optional: Render an svg of the traced graph
            // if (epoch == 0 and i == 0) {
            //     try zg.utils.renderD2(loss, zg.utils.PrintOptions.plain, fw_arena.allocator(), "./docs/comp_graph_mnist.svg");
            // }

            if (!ima.reset(.retain_capacity)) log.err("Issue in ima reset", .{});
        }
        const avg_loss = total_loss / @as(f32, @floatFromInt(train_dataset.images.len));
        try stdout.print("Epoch {d}: Avg Loss = {d:.4}\n", .{ epoch + 1, avg_loss });
    }
    if (!ima.reset(.free_all)) log.err("Issue in fw arena reset", .{});
    try stdout.print("Training complete ({d} epochs)]\n", .{num_epochs});

    // Eval --------------------------------------------------------------------
    // Eval on train set
    const train_eval = try evalMnist(ima.allocator(), model, train_dataset);
    try stdout.print("Train acc: {d:.2} (n={d})\n", .{ train_eval.acc * 100, train_eval.n });
    train_dataset.deinit();

    // Eval on test set
    try stdout.print("Loading test data...\n", .{});
    const test_dataset = try MnistDataset(T).load(acquired_arena.allocator(), test_path, batch_size);
    defer test_dataset.deinit();
    const test_eval = try evalMnist(ima.allocator(), model, test_dataset);
    try stdout.print("Test acc: {d:.2} (n={d})\n", .{ test_eval.acc * 100, test_eval.n });
    try bw.flush();
}

fn evalMnist(allocator: std.mem.Allocator, model: MnistModel(T), dataset: MnistDataset(T)) !struct { correct: f32, n: u32, acc: f32 } {
    zg.rt_grad_enabled = false; // disable gradient tracking
    var n: u32 = 0;
    var correct: f32 = 0;
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
    }
    return .{ .correct = correct, .n = n, .acc = correct / @as(f32, @floatFromInt(n)) };
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
