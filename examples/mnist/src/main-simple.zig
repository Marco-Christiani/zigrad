/// Trains a neural network model on the MNIST dataset using a manual training loop.
const std = @import("std");
const zg = @import("zigrad");
const MnistDataset = @import("dataset.zig").MnistDataset;

const Optimizer = zg.optim.SGD(T);
const MnistModel = @import("model.zig").MnistModel(f32, Optimizer);

const std_options = .{ .log_level = .info };
const log = std.log.scoped(.mnist);
const T = f32;

pub fn run_mnist(train_path: []const u8, test_path: []const u8) !void {
    var cpu = zg.device.HostDevice.init(std.heap.raw_c_allocator);
    defer cpu.deinit();
    const device = cpu.reference();

    var optim: Optimizer = .{
        .lr = 0.01,
        .grad_clip_max_norm = 10.0,
        .grad_clip_delta = 1e-6,
        .grad_clip_enabled = false,
    };

    var gm = zg.GraphManager(zg.NDTensor(f32)).init(cpu.allocator, .{
        .eager_teardown = true,
    });
    defer gm.deinit();

    var model = try MnistModel.init(device, &optim); // 109_386
    defer model.deinit();

    std.debug.print("Loading train data...\n", .{});
    const batch_size = 64;
    const train_dataset = try MnistDataset(T).load(device, train_path, batch_size);

    // Train -------------------------------------------------------------------
    std.debug.print("Training...\n", .{});
    const num_epochs = 3;
    for (0..num_epochs) |epoch| {
        var total_loss: f64 = 0;
        for (train_dataset.images, train_dataset.labels, 0..) |image, label, i| {
            try image.set_label("image_batch");
            try label.set_label("label_batch");

            const output = try model.forward(image);

            const loss = try zg.loss.softmax_cross_entropy_loss(f32, output, label);
            total_loss += loss.get(0);

            //return loss.print_arrows();

            std.debug.print("train_loss: {d:<5.5} [{d}/{d}]\n", .{
                loss.get(0),
                i,
                train_dataset.images.len,
            });

            try gm.backward(loss);
        }
        const avg_loss = total_loss / @as(f32, @floatFromInt(train_dataset.images.len));
        std.debug.print("Epoch {d}: Avg Loss = {d:.4}\n", .{ epoch + 1, avg_loss });
    }
    std.debug.print("Training complete ({d} epochs)]\n", .{num_epochs});

    //// Eval --------------------------------------------------------------------
    //// Eval on train set
    const train_eval = try eval_mnist(&model, train_dataset);
    std.debug.print("Train acc: {d:.2} (n={d})\n", .{ train_eval.acc * 100, train_eval.n });
    train_dataset.deinit();

    // Eval on test set
    std.debug.print("Loading test data...\n", .{});
    const test_dataset = try MnistDataset(T).load(device, test_path, batch_size);
    defer test_dataset.deinit();
    const test_eval = try eval_mnist(&model, test_dataset);
    std.debug.print("Test acc: {d:.2} (n={d})\n", .{ test_eval.acc * 100, test_eval.n });
}

fn eval_mnist(model: *MnistModel, dataset: MnistDataset(T)) !struct { correct: f32, n: u32, acc: f32 } {
    zg.rt_grad_enabled = false; // disable gradient tracking
    var n: u32 = 0;
    var correct: f32 = 0;
    for (dataset.images, dataset.labels) |image, label| {
        const output = try model.forward(image);
        defer output.deinit();
        const batch_n = output.data.shape.get(0);
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
    var buf1: [1024]u8 = undefined;
    var buf2: [1024]u8 = undefined;
    const data_sub_dir = std.posix.getenv("ZG_DATA_DIR") orelse "data";
    const train_full = try std.fmt.bufPrint(&buf1, "{s}/{s}", .{ data_sub_dir, "mnist_train_full.csv" });
    const test_full = try std.fmt.bufPrint(&buf2, "{s}/{s}", .{ data_sub_dir, "mnist_test_full.csv" });
    try run_mnist(train_full, test_full);
}

//test run_mnist {
//    var buf1: [1024]u8 = undefined;
//    var buf2: [1024]u8 = undefined;
//    const data_sub_dir = std.posix.getenv("ZG_DATA_DIR") orelse "data";
//    const train_small = try std.fmt.bufPrint(&buf1, "{s}/{s}", .{ data_sub_dir, "mnist_train_small.csv" });
//    const test_small = try std.fmt.bufPrint(&buf2, "{s}/{s}", .{ data_sub_dir, "mnist_test_small.csv" });
//    run_mnist(train_small, test_small) catch |err| switch (err) {
//        std.fs.File.OpenError.FileNotFound => std.log.warn("{s} error opening test file. Skipping `runMnist` test.", .{@errorName(err)}),
//        else => return err,
//    };
//}
