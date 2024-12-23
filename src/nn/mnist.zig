const std = @import("std");
const zg = @import("../zigrad.zig");

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
    allocator: std.mem.Allocator,

    pub fn initConv(allocator: std.mem.Allocator) !Self {
        var conv1 = try Conv2DLayer(T).init(allocator, 1, 6, 5, 1, 0, 1);
        var pool1 = try MaxPool2DLayer(T).init(allocator, 2, 2, 0);
        var relu1 = try ReLULayer(T).init(allocator);
        var conv2 = try Conv2DLayer(T).init(allocator, 6, 16, 5, 1, 0, 1);
        var pool2 = try MaxPool2DLayer(T).init(allocator, 2, 2, 0);
        var relu2 = try ReLULayer(T).init(allocator);
        var flatten = try FlattenLayer(T).init(allocator);
        var fc1 = try LinearLayer(T).init(allocator, 256, 120);
        var relu3 = try ReLULayer(T).init(allocator);
        var fc2 = try LinearLayer(T).init(allocator, 120, 84);
        var relu4 = try ReLULayer(T).init(allocator);
        var fc3 = try LinearLayer(T).init(allocator, 84, 10);

        var self = Self{
            .allocator = allocator,
            .model = try Model(T).init(allocator),
        };
        try self.model.addLayer(conv1.asLayer());
        try self.model.addLayer(pool1.asLayer());
        try self.model.addLayer(relu1.asLayer());
        try self.model.addLayer(conv2.asLayer());
        try self.model.addLayer(pool2.asLayer());
        try self.model.addLayer(relu2.asLayer());
        try self.model.addLayer(flatten.asLayer());
        try self.model.addLayer(fc1.asLayer());
        try self.model.addLayer(relu3.asLayer());
        try self.model.addLayer(fc2.asLayer());
        try self.model.addLayer(relu4.asLayer());
        try self.model.addLayer(fc3.asLayer());
        return self;
    }

    pub fn initSimple(allocator: std.mem.Allocator) !Self {
        var self = Self{
            .allocator = allocator,
            .model = try Model(T).init(allocator),
        };

        var reshape = try FlattenLayer(T).init(allocator);
        var fc1 = try LinearLayer(T).init(allocator, 28 * 28, 128);
        var relu1 = try ReLULayer(T).init(allocator);
        var fc2 = try LinearLayer(T).init(allocator, 128, 64);
        var relu2 = try ReLULayer(T).init(allocator);
        var fc3 = try LinearLayer(T).init(allocator, 64, 10);

        try self.model.addLayer(reshape.asLayer());

        try self.model.addLayer(fc1.asLayer());
        try self.model.addLayer(relu1.asLayer());

        try self.model.addLayer(fc2.asLayer());
        try self.model.addLayer(relu2.asLayer());

        try self.model.addLayer(fc3.asLayer());
        return self;
    }

    pub fn initSimple2(allocator: std.mem.Allocator) !Self {
        var self = Self{
            .allocator = allocator,
            .model = try Model(T).init(allocator),
        };

        var reshape = try FlattenLayer(T).init(allocator);
        var fc1 = try LinearLayer(T).init(allocator, 28 * 28, 28 * 28);
        var relu1 = try ReLULayer(T).init(allocator);
        var fc2 = try LinearLayer(T).init(allocator, 28 * 28, 128);
        var relu2 = try ReLULayer(T).init(allocator);
        var fc3 = try LinearLayer(T).init(allocator, 128, 10);

        try self.model.addLayer(reshape.asLayer());

        try self.model.addLayer(fc1.asLayer());
        try self.model.addLayer(relu1.asLayer());

        try self.model.addLayer(fc2.asLayer());
        try self.model.addLayer(relu2.asLayer());

        try self.model.addLayer(fc3.asLayer());
        return self;
    }

    pub fn initConv2(allocator: std.mem.Allocator) !Self {
        var self = Self{
            .allocator = allocator,
            .model = try Model(T).init(allocator),
        };

        var conv1 = try Conv2DLayer(T).init(allocator, 1, 3, 5, 1, 0, 1);
        var relu1 = try ReLULayer(T).init(allocator);
        var conv2 = try Conv2DLayer(T).init(allocator, 3, 3, 5, 1, 0, 1);
        var relu2 = try ReLULayer(T).init(allocator);

        var reshape = try FlattenLayer(T).init(allocator);
        var fc1 = try LinearLayer(T).init(allocator, 3 * 20 * 20, 128);
        var relu3 = try ReLULayer(T).init(allocator);
        var fc2 = try LinearLayer(T).init(allocator, 128, 10);
        var relu4 = try ReLULayer(T).init(allocator);

        try self.model.addLayer(conv1.asLayer());
        try self.model.addLayer(relu1.asLayer());
        try self.model.addLayer(conv2.asLayer());
        try self.model.addLayer(relu2.asLayer());
        try self.model.addLayer(reshape.asLayer());
        try self.model.addLayer(fc1.asLayer());
        try self.model.addLayer(relu3.asLayer());
        try self.model.addLayer(fc2.asLayer());
        try self.model.addLayer(relu4.asLayer());
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
    // pub fn getParameters(self: Self) []NDTensor(T) {
    //     return self.model.getParameters();
    // }

    pub fn countParams(self: Self) usize {
        var total: usize = 0;
        const params = self.model.getParameters();
        defer self.model.allocator.free(params);
        for (params) |p| {
            total += p.grad.?.size();
        }
        return total;
    }
};

const MnistDataset = struct {
    images: []*NDTensor(T),
    labels: []*NDTensor(T),
    allocator: std.mem.Allocator,

    pub fn load(allocator: std.mem.Allocator, csv_path: []const u8, batch_size: usize) !@This() {
        const file = try std.fs.cwd().openFile(csv_path, .{});
        defer file.close();

        const file_size = try file.getEndPos();
        const file_contents = try file.readToEndAlloc(allocator, file_size);
        defer allocator.free(file_contents);

        var images = std.ArrayList(*NDTensor(T)).init(allocator);
        var labels = std.ArrayList(*NDTensor(T)).init(allocator);

        var lines = std.mem.split(u8, file_contents, "\n");
        var batch_images = try allocator.alloc(T, batch_size * 784);
        defer allocator.free(batch_images);
        var batch_labels = try allocator.alloc(T, batch_size * 10);
        defer allocator.free(batch_labels);
        var batch_count: usize = 0;

        while (lines.next()) |line| {
            if (line.len == 0) continue; // skip empty lines
            var values = std.mem.split(u8, line, ",");

            for (0..10) |i| {
                batch_labels[batch_count * 10 + i] = @as(T, @floatFromInt(try std.fmt.parseUnsigned(u8, values.next().?, 10)));
            }

            for (0..784) |i| {
                const pixel_value = try std.fmt.parseFloat(T, values.next().?);
                batch_images[batch_count * 784 + i] = pixel_value;
            }

            batch_count += 1;

            if (batch_count == batch_size or lines.peek() == null) {
                const image_tensor = try NDTensor(T).init(batch_images, &[_]usize{ batch_size, 1, 28, 28 }, true, allocator);
                const label_tensor = try NDTensor(T).init(batch_labels, &[_]usize{ batch_size, 10 }, true, allocator);
                try images.append(image_tensor);
                try labels.append(label_tensor);
                batch_count = 0;
            }
        }

        if (batch_count > 0) { // remainder
            const image_tensor = try NDTensor(T).init(batch_images[0 .. batch_count * 784], &[_]usize{ batch_count, 1, 28, 28 }, true, allocator);
            const label_tensor = try NDTensor(T).init(batch_labels[0 .. batch_count * 10], &[_]usize{ batch_count, 10 }, true, allocator);
            try images.append(image_tensor);
            try labels.append(label_tensor);
        }

        return .{ .images = try images.toOwnedSlice(), .labels = try labels.toOwnedSlice(), .allocator = allocator };
    }

    fn deinit(self: @This()) void {
        for (self.images) |image| image.deinit();
        for (self.labels) |label| label.deinit();
        self.allocator.free(self.images);
        self.allocator.free(self.labels);
    }
};

pub fn runMnist(train_path: []const u8, test_path: []const u8) !void {
    var model_arena = std.heap.ArenaAllocator.init(std.heap.raw_c_allocator);
    defer model_arena.deinit();

    var dataset_arena = std.heap.ArenaAllocator.init(std.heap.raw_c_allocator);
    defer dataset_arena.deinit();

    var fw_arena = std.heap.ArenaAllocator.init(std.heap.raw_c_allocator);
    defer fw_arena.deinit();

    // bw_arena is torn down before eval
    var bw_arena = std.heap.ArenaAllocator.init(std.heap.raw_c_allocator);

    var model = try MnistModel.initSimple(model_arena.allocator()); // 109_386
    // var model = try MnistModel.initSimple2(model_arena.allocator()); // 717_210
    // var model = try MnistModel.initConv(model_arena.allocator()); // 44_426
    // var model = try MnistModel.initConv2(model_arena.allocator()); // 155_324
    log.info("n params = {}", .{model.countParams()});

    log.info("Loading train data...", .{});
    const batch_size = 64;
    const train_dataset = try MnistDataset.load(dataset_arena.allocator(), train_path, batch_size);
    const n = train_dataset.images.len;

    var trainer = Trainer(T, .ce).init(
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
    log.info("Training...", .{});
    const num_epochs = 3;
    for (0..num_epochs) |epoch| {
        var total_loss: f64 = 0;
        // TODO: impl/use trainer loop
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
            log.info("train_loss: {d:<5.5} [{d}/{d}] [ms/sample: {d}]", .{
                loss.data.data[0],
                i,
                n,
                ms_per_sample,
            });
            // Optional: Render a trace of the graph and look at the raw data
            // if (epoch == 0 and i == 0) {
            //     try zg.utils.renderD2(loss, zg.utils.PrintOptions.plain, fw_arena.allocator(), "./docs/comp_graph.svg");
            //     const view = try image.data.slice(0, 0, 1);
            //     defer view.shape.deinit();
            //     std.debug.print("First Value:\n", .{});
            //     view.print();
            //     std.debug.print("\n", .{});
            // }

            if (!bw_arena.reset(.retain_capacity)) log.warn("Issue in bw arena reset", .{});
            if (!fw_arena.reset(.retain_capacity)) log.warn("Issue in fw arena reset", .{});
        }
        const avg_loss = total_loss / @as(f32, @floatFromInt(train_dataset.images.len));
        log.info("Epoch {d}: Avg Loss = {d:.4}", .{ epoch + 1, avg_loss });
    }
    bw_arena.deinit();
    const train_time_ms = @as(f64, @floatFromInt(timer.lap())) / @as(f64, @floatFromInt(std.time.ns_per_ms));
    log.info("Training complete ({d} epochs). [{d}ms]", .{ num_epochs, train_time_ms });

    // model.model.eval() // TODO: model.eval()
    const train_eval = try evalMnist(&fw_arena, model, train_dataset);
    const train_acc = train_eval.correct / @as(f32, @floatFromInt(train_eval.n));
    const eval_train_time_ms = @as(f64, @floatFromInt(timer.lap())) / @as(f64, @floatFromInt(std.time.ns_per_ms));
    log.info("Train acc: {d:.2} (n={d}) [{d}ms]", .{ train_acc * 100, train_eval.n, eval_train_time_ms });
    train_dataset.deinit();

    log.info("Loading test data...", .{});
    const test_dataset = try MnistDataset.load(dataset_arena.allocator(), test_path, batch_size);
    defer test_dataset.deinit();
    timer.reset();
    const test_eval = try evalMnist(&fw_arena, model, test_dataset);
    const eval_test_time_ms = @as(f64, @floatFromInt(timer.lap())) / @as(f64, @floatFromInt(std.time.ns_per_ms));
    const test_acc = test_eval.correct / @as(f32, @floatFromInt(test_eval.n));
    log.info("Test acc: {d:.2} (n={d}) [{d}ms]", .{ test_acc * 100, test_eval.n, eval_test_time_ms });
    log.info("Train: {d}ms", .{train_time_ms});
    log.info("Eval train: {d}ms", .{eval_train_time_ms});
    log.info("Eval test: {d}ms", .{eval_test_time_ms});
}

fn evalMnist(arena: *std.heap.ArenaAllocator, model: MnistModel, dataset: MnistDataset) !struct { correct: f32, n: u32 } {
    zg.rt_grad_enabled = false;
    // model.model.eval() // TODO: model.eval()
    var n: u32 = 0;
    var correct: f32 = 0;
    var timer = try std.time.Timer.start();
    for (dataset.images, dataset.labels) |image, label| {
        timer.reset();
        const output = try model.model.forward(image, arena.allocator());
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
        if (!arena.reset(.retain_capacity)) log.warn("Issue in arena reset", .{});
    }
    return .{ .correct = correct, .n = n };
}

fn getMnistPaths(size: enum { full, small }, allocator: std.mem.Allocator) !struct { train_csv: []u8, test_csv: []u8 } {
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
    const fpaths = try getMnistPaths(.full, allocator);
    defer {
        allocator.free(fpaths.train_csv);
        allocator.free(fpaths.test_csv);
    }
    log.info("train data: {s}", .{fpaths.train_csv});
    log.info("test data: {s}", .{fpaths.test_csv});
    try runMnist(fpaths.train_csv, fpaths.test_csv);
}

test runMnist {
    const allocator = std.heap.c_allocator;
    const fpaths = try getMnistPaths(.small, allocator);
    defer {
        allocator.free(fpaths.train_csv);
        allocator.free(fpaths.test_csv);
    }
    runMnist(fpaths.train_csv, fpaths.test_csv) catch |err| switch (err) {
        std.fs.File.OpenError.FileNotFound => std.log.warn("{s} error opening test file. Skipping `runMnist` test.", .{@errorName(err)}),
        else => return err,
    };
}
