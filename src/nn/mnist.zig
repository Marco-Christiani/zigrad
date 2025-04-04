const std = @import("std");
const zg = @import("../zigrad.zig");
const DeviceReference = zg.DeviceReference;

const Model = zg.Model;
const MaxPool2DLayer = zg.layer.MaxPool2DLayer;
const Conv2DLayer = zg.layer.Conv2DLayer;
const LinearLayer = zg.layer.LinearLayer;
const ReLULayer = zg.layer.ReLULayer;
const FlattenLayer = zg.layer.FlattenLayer;
const Trainer = zg.Trainer;
const NDTensor = zg.NDTensor;
const prod = zg.arrayutils.prod;

const log = std.log.scoped(.zg_mnist);
const T = f32;

const MnistModel = struct {
    const Self = @This();
    model: Model(T),
    device: DeviceReference,

    pub fn init_conv(device: DeviceReference) !Self {
        var conv1 = try Conv2DLayer(T).init(device, 1, 6, 5, 1, 0, 1);
        var pool1 = try MaxPool2DLayer(T).init(device, 2, 2, 0);
        var relu1 = try ReLULayer(T).init(device);
        var conv2 = try Conv2DLayer(T).init(device, 6, 16, 5, 1, 0, 1);
        var pool2 = try MaxPool2DLayer(T).init(device, 2, 2, 0);
        var relu2 = try ReLULayer(T).init(device);
        var flatten = try FlattenLayer(T).init(device);
        var fc1 = try LinearLayer(T).init(device, 256, 120);
        var relu3 = try ReLULayer(T).init(device);
        var fc2 = try LinearLayer(T).init(device, 120, 84);
        var relu4 = try ReLULayer(T).init(device);
        var fc3 = try LinearLayer(T).init(device, 84, 10);

        var self = Self{
            .device = device,
            .model = try Model(T).init(device),
        };
        try self.model.add_layer(conv1.as_layer());
        try self.model.add_layer(pool1.as_layer());
        try self.model.add_layer(relu1.as_layer());
        try self.model.add_layer(conv2.as_layer());
        try self.model.add_layer(pool2.as_layer());
        try self.model.add_layer(relu2.as_layer());
        try self.model.add_layer(flatten.as_layer());
        try self.model.add_layer(fc1.as_layer());
        try self.model.add_layer(relu3.as_layer());
        try self.model.add_layer(fc2.as_layer());
        try self.model.add_layer(relu4.as_layer());
        try self.model.add_layer(fc3.as_layer());
        return self;
    }

    pub fn init_simple(device: DeviceReference) !Self {
        var self = Self{
            .device = device,
            .model = try Model(T).init(device),
        };

        var reshape = try FlattenLayer(T).init(device);
        var fc1 = try LinearLayer(T).init(device, 28 * 28, 128);
        var relu1 = try ReLULayer(T).init(device);
        var fc2 = try LinearLayer(T).init(device, 128, 64);
        var relu2 = try ReLULayer(T).init(device);
        var fc3 = try LinearLayer(T).init(device, 64, 10);

        try self.model.add_layer(reshape.as_layer());

        try self.model.add_layer(fc1.as_layer());
        try self.model.add_layer(relu1.as_layer());

        try self.model.add_layer(fc2.as_layer());
        try self.model.add_layer(relu2.as_layer());

        try self.model.add_layer(fc3.as_layer());
        return self;
    }

    pub fn init_simple2(device: DeviceReference) !Self {
        var self = Self{
            .device = device,
            .model = try Model(T).init(device),
        };

        var reshape = try FlattenLayer(T).init(device);
        var fc1 = try LinearLayer(T).init(device, 28 * 28, 28 * 28);
        var relu1 = try ReLULayer(T).init(device);
        var fc2 = try LinearLayer(T).init(device, 28 * 28, 128);
        var relu2 = try ReLULayer(T).init(device);
        var fc3 = try LinearLayer(T).init(device, 128, 10);

        try self.model.add_layer(reshape.as_layer());

        try self.model.add_layer(fc1.as_layer());
        try self.model.add_layer(relu1.as_layer());

        try self.model.add_layer(fc2.as_layer());
        try self.model.add_layer(relu2.as_layer());

        try self.model.add_layer(fc3.as_layer());
        return self;
    }

    pub fn init_conv2(device: DeviceReference) !Self {
        var self = Self{
            .device = device,
            .model = try Model(T).init(device),
        };

        var conv1 = try Conv2DLayer(T).init(device, 1, 3, 5, 1, 0, 1);
        var relu1 = try ReLULayer(T).init(device);
        var conv2 = try Conv2DLayer(T).init(device, 3, 3, 5, 1, 0, 1);
        var relu2 = try ReLULayer(T).init(device);

        var reshape = try FlattenLayer(T).init(device);
        var fc1 = try LinearLayer(T).init(device, 3 * 20 * 20, 128);
        var relu3 = try ReLULayer(T).init(device);
        var fc2 = try LinearLayer(T).init(device, 128, 10);
        var relu4 = try ReLULayer(T).init(device);

        try self.model.add_layer(conv1.as_layer());
        try self.model.add_layer(relu1.as_layer());
        try self.model.add_layer(conv2.as_layer());
        try self.model.add_layer(relu2.as_layer());
        try self.model.add_layer(reshape.as_layer());
        try self.model.add_layer(fc1.as_layer());
        try self.model.add_layer(relu3.as_layer());
        try self.model.add_layer(fc2.as_layer());
        try self.model.add_layer(relu4.as_layer());
        return self;
    }

    pub fn deinit(self: *Self) void {
        log.info("deinit model", .{});
        self.model.deinit();
        self.* = undefined;
    }

    // pending model interface
    // pub fn forward(self: Self, input: NDTensor(T)) !NDTensor(T) {
    //     const a = self.model.forward(input);
    //     log.info("softmaxing and returning", .{});
    //     return try ops.softmax(T, a, self.allocator);
    // }
    //
    // pub fn get_parameters(self: Self) []NDTensor(T) {
    //     return self.model.get_parameters();
    // }

    pub fn count_params(self: Self) usize {
        var total: usize = 0;
        const params = self.model.get_parameters();
        defer self.model.device.allocator.free(params);
        for (params) |p| {
            total += if (p.grad) |*g| g.size() else 0;
        }
        return total;
    }
};

const MnistDataset = struct {
    images: []*NDTensor(T),
    labels: []*NDTensor(T),
    device: DeviceReference,

    pub fn load(device: DeviceReference, csv_path: []const u8, batch_size: usize) !@This() {
        // This example is entirely on the CPU so I'm keeping the
        // allocator directly for all of the data allocations.
        const file = try std.fs.cwd().openFile(csv_path, .{});
        defer file.close();

        const file_size = try file.getEndPos();
        const file_contents = try file.readToEndAlloc(device.allocator, file_size);
        defer device.allocator.free(file_contents);

        var images = std.ArrayList(*NDTensor(T)).init(device.allocator);
        var labels = std.ArrayList(*NDTensor(T)).init(device.allocator);

        var lines = std.mem.splitScalar(u8, file_contents, '\n');
        var batch_images = try device.allocator.alloc(T, batch_size * 784);
        defer device.allocator.free(batch_images);
        var batch_labels = try device.allocator.alloc(T, batch_size * 10);
        defer device.allocator.free(batch_labels);
        var batch_count: usize = 0;

        while (lines.next()) |line| {
            if (line.len == 0) continue; // skip empty lines
            var values = std.mem.splitScalar(u8, line, ',');

            for (0..10) |i| {
                batch_labels[batch_count * 10 + i] = @as(T, @floatFromInt(try std.fmt.parseUnsigned(u8, values.next().?, 10)));
            }

            for (0..784) |i| {
                const pixel_value = try std.fmt.parseFloat(T, values.next().?);
                // pixel_value = pixel_value / 255; // NOTE: Zigrad does NOT need this, but since torch does, ironically adding it messes w convergence here.
                batch_images[batch_count * 784 + i] = pixel_value;
            }

            batch_count += 1;

            if (batch_count == batch_size or lines.peek() == null) {
                const image_tensor = try NDTensor(T).init(batch_images, &[_]usize{ batch_size, 1, 28, 28 }, true, device);
                const label_tensor = try NDTensor(T).init(batch_labels, &[_]usize{ batch_size, 10 }, true, device);
                image_tensor.acquire();
                label_tensor.acquire();
                try images.append(image_tensor);
                try labels.append(label_tensor);
                batch_count = 0;
            }
        }

        if (batch_count > 0) { // remainder
            const image_tensor = try NDTensor(T).init(batch_images[0 .. batch_count * 784], &[_]usize{ batch_count, 1, 28, 28 }, true, device);
            const label_tensor = try NDTensor(T).init(batch_labels[0 .. batch_count * 10], &[_]usize{ batch_count, 10 }, true, device);
            image_tensor.acquire();
            label_tensor.acquire();
            try images.append(image_tensor);
            try labels.append(label_tensor);
        }

        return .{ .images = try images.toOwnedSlice(), .labels = try labels.toOwnedSlice(), .device = device };
    }

    fn deinit(self: @This()) void {
        for (self.images, self.labels) |image, label| {
            image.release();
            label.release();
            image.deinit();
            label.deinit();
        }
        self.device.allocator.free(self.images);
        self.device.allocator.free(self.labels);
    }
};

pub fn run_mnist(train_path: []const u8, test_path: []const u8) !void {
    comptime {
        @setFloatMode(.optimized);
    }

    var cpu = zg.device.HostDevice.init(std.heap.raw_c_allocator);
    defer cpu.deinit();

    const device = cpu.reference();

    var model = try MnistModel.init_simple(device); // 109_386
    // var model = try MnistModel.init_simple2(device); // 717_210
    // var model = try MnistModel.init_conv(device); // 44_426
    // var model = try MnistModel.init_conv2(device); // 155_324
    log.info("n params = {}", .{model.count_params()});
    defer model.deinit();

    log.info("Loading train data...", .{});
    const batch_size = 64;
    const train_dataset = try MnistDataset.load(device, train_path, batch_size);
    const n = train_dataset.images.len;

    var trainer = Trainer(T, .ce).init(
        model.model,
        zg.optim.SGD(T){
            .lr = 0.1,
            .grad_clip_max_norm = 10.0,
            .grad_clip_delta = 1e-6,
            .grad_clip_enabled = false,
        },
        .{ .eager_teardown = true },
    );
    defer trainer.deinit();

    var step_timer = try std.time.Timer.start();
    var timer = try std.time.Timer.start();
    log.info("Training...", .{});

    const num_epochs = 3;
    for (0..num_epochs) |epoch| {
        var total_loss: f64 = 0;
        // TODO: impl/use trainer loop
        for (train_dataset.images, train_dataset.labels, 0..) |image, label, i| {
            try image.set_label("image_batch");
            try label.set_label("label_batch");

            step_timer.reset();
            const loss = try trainer.train_step(image, label);
            const t1 = @as(f64, @floatFromInt(step_timer.read()));
            const ms_per_sample = t1 / @as(f64, @floatFromInt(std.time.ns_per_ms * batch_size));
            total_loss += loss.get(0);

            log.info("train_loss: {d:<5.5} [{d}/{d}] [ms/sample: {d}]", .{
                loss.get(0),
                i,
                n,
                ms_per_sample,
            });
            loss.deinit();

            // Optional: Render a trace of the graph and look at the raw data
            // if (epoch == 0 and i == 0) {
            //     try zg.utils.render_d2(loss, zg.utils.PrintOptions.plain, fw_arena.allocator(), "./docs/comp_graph.svg");
            //     const view = try image.data.slice(0, 0, 1);
            //     defer view.shape.deinit();
            //     std.debug.print("First Value:\n", .{});
            //     view.print();
            //     std.debug.print("\n", .{});
            // }
        }
        const avg_loss = total_loss / @as(f32, @floatFromInt(train_dataset.images.len));
        log.info("Epoch {d}: Avg Loss = {d:.4}", .{ epoch + 1, avg_loss });
    }
    // bw_arena.deinit();
    const train_time_ms = @as(f64, @floatFromInt(timer.lap())) / @as(f64, @floatFromInt(std.time.ns_per_ms));
    log.info("Training complete ({d} epochs). [{d}ms]", .{ num_epochs, train_time_ms });

    // model.model.eval() // TODO: model.eval()
    const train_eval = try eval_mnist(model, train_dataset);
    const train_acc = train_eval.correct / @as(f32, @floatFromInt(train_eval.n));
    const eval_train_time_ms = @as(f64, @floatFromInt(timer.lap())) / @as(f64, @floatFromInt(std.time.ns_per_ms));
    log.info("Train acc: {d:.2} (n={d}) [{d}ms]", .{ train_acc * 100, train_eval.n, eval_train_time_ms });
    train_dataset.deinit();

    log.info("Loading test data...", .{});
    const test_dataset = try MnistDataset.load(device, test_path, batch_size);
    defer test_dataset.deinit();
    timer.reset();
    const test_eval = try eval_mnist(model, test_dataset);
    const eval_test_time_ms = @as(f64, @floatFromInt(timer.lap())) / @as(f64, @floatFromInt(std.time.ns_per_ms));
    const test_acc = test_eval.correct / @as(f32, @floatFromInt(test_eval.n));
    log.info("Test acc: {d:.2} (n={d}) [{d}ms]", .{ test_acc * 100, test_eval.n, eval_test_time_ms });
    log.info("Train: {d}ms", .{train_time_ms});
    log.info("Eval train: {d}ms", .{eval_train_time_ms});
    log.info("Eval test: {d}ms", .{eval_test_time_ms});
}

fn eval_mnist(
    model: MnistModel,
    dataset: MnistDataset,
) !struct { correct: f32, n: u32 } {
    zg.rt_grad_enabled = false;
    // model.model.eval(); // TODO: model.eval()
    var n: u32 = 0;
    var correct: f32 = 0;
    var timer = try std.time.Timer.start();
    for (dataset.images, dataset.labels) |image, label| {
        timer.reset();
        const output = try model.model.forward(image);
        const batch_n = output.data.shape.get(0);
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

fn get_mnist_paths(size: enum { full, small }, allocator: std.mem.Allocator) !struct { train_csv: []u8, test_csv: []u8 } {
    const data_dir = std.posix.getenv("ZG_DATA_DIR") orelse "data";
    return switch (size) {
        .full => .{
            .train_csv = try std.fmt.allocPrint(allocator, "{s}/mnist_train_full.csv", .{data_dir}),
            .test_csv = try std.fmt.allocPrint(allocator, "{s}/mnist_test_full.csv", .{data_dir}),
        },
        .small => .{
            .train_csv = try std.fmt.allocPrint(allocator, "{s}/mnist_train_small.csv", .{data_dir}),
            .test_csv = try std.fmt.allocPrint(allocator, "{s}/mnist_test_small.csv", .{data_dir}),
        },
    };
}

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    const fpaths = try get_mnist_paths(.full, allocator);
    defer {
        allocator.free(fpaths.train_csv);
        allocator.free(fpaths.test_csv);
    }
    log.info("train data: {s}", .{fpaths.train_csv});
    log.info("test data: {s}", .{fpaths.test_csv});
    try run_mnist(fpaths.train_csv, fpaths.test_csv);
}

test run_mnist {
    const allocator = std.heap.c_allocator;
    const fpaths = try get_mnist_paths(.small, allocator);
    defer {
        allocator.free(fpaths.train_csv);
        allocator.free(fpaths.test_csv);
    }
    run_mnist(fpaths.train_csv, fpaths.test_csv) catch |err| switch (err) {
        std.fs.File.OpenError.FileNotFound => std.log.warn("{s} error opening test file. Skipping `run_mnist` test.", .{@errorName(err)}),
        else => return err,
    };
}
