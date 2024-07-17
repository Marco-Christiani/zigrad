const std = @import("std");
const zg = @import("../root.zig");

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

pub fn MNISTModel(comptime T: type) type {
    return struct {
        const Self = @This();
        model: Model(T),
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) !*Self {
            const self = try allocator.create(Self);
            self.allocator = allocator;
            self.model = try Model(T).init(allocator);

            var conv1 = try Conv2DLayer(T).init(allocator, 1, 6, 5, 1, 0, 1);
            var pool1 = try MaxPool2DLayer(T).init(allocator, 2, 2, 0);
            var relu1 = try ReLULayer(T).init(allocator);
            var conv2 = try Conv2DLayer(T).init(allocator, 6, 16, 5, 1, 0, 1);
            var pool2 = try MaxPool2DLayer(T).init(allocator, 2, 2, 0);
            var relu2 = try ReLULayer(T).init(allocator);
            var flatten = FlattenLayer(T){};
            var fc1 = try LinearLayer(T).init(allocator, 256, 120);
            var relu3 = try ReLULayer(T).init(allocator);
            var fc2 = try LinearLayer(T).init(allocator, 120, 84);
            var relu4 = try ReLULayer(T).init(allocator);
            var fc3 = try LinearLayer(T).init(allocator, 84, 10);

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

            var reshape = FlattenLayer(T){};
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

            var reshape = FlattenLayer(T){};
            var fc1 = try LinearLayer(T).init(allocator, 28 * 28, 28 * 28);
            var relu1 = try ReLULayer(T).init(allocator);
            var fc2 = try LinearLayer(T).init(allocator, 28 * 28, 128);
            var relu2 = try ReLULayer(T).init(allocator);
            var fc3 = try LinearLayer(T).init(allocator, 128, 10);
            var relu3 = try ReLULayer(T).init(allocator);

            try self.model.addLayer(reshape.asLayer());

            try self.model.addLayer(fc1.asLayer());
            try self.model.addLayer(relu1.asLayer());

            try self.model.addLayer(fc2.asLayer());
            try self.model.addLayer(relu2.asLayer());

            try self.model.addLayer(fc3.asLayer());
            try self.model.addLayer(relu3.asLayer());
            return self;
        }

        pub fn initBig(allocator: std.mem.Allocator) !Self {
            var self = Self{
                .allocator = allocator,
                .model = try Model(T).init(allocator),
            };

            var conv1 = try Conv2DLayer(T).init(allocator, 1, 32, 3, 1, 1, 1);
            var relu1 = try ReLULayer(T).init(allocator);
            var conv2 = try Conv2DLayer(T).init(allocator, 32, 64, 3, 1, 1, 1);
            var relu2 = try ReLULayer(T).init(allocator);

            var reshape = FlattenLayer(T){};
            var fc1 = try LinearLayer(T).init(allocator, 64 * 28 * 28, 128);
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
    };
}

pub fn loadMNIST(
    comptime T: type,
    allocator: std.mem.Allocator,
    filepath: []const u8,
    batch_size: usize,
) !struct { images: []*NDTensor(T), labels: []*NDTensor(T) } {
    const file = try std.fs.cwd().openFile(filepath, .{});
    defer file.close();

    const file_size = try file.getEndPos();
    const file_contents = try file.readToEndAlloc(allocator, file_size);
    defer allocator.free(file_contents);

    var images = std.ArrayList(*NDTensor(T)).init(allocator);
    var labels = std.ArrayList(*NDTensor(T)).init(allocator);

    var lines = std.mem.split(u8, file_contents, "\n");
    var batch_images = try allocator.alloc(T, batch_size * 784);
    var batch_labels = try allocator.alloc(T, batch_size * 10);
    var batch_count: usize = 0;

    while (lines.next()) |line| {
        if (line.len == 0) continue; // Skip empty lines
        var values = std.mem.split(u8, line, ",");
        // const label = try std.fmt.parseUnsigned(u8, values.next().?, 10);

        for (0..10) |i| {
            // batch_labels[batch_count * 10 + i] = if (i == label) 1 else 0;
            batch_labels[batch_count * 10 + i] = @as(T, @floatFromInt(try std.fmt.parseUnsigned(u8, values.next().?, 10)));
        }

        for (0..784) |i| {
            const pixel_value = try std.fmt.parseFloat(T, values.next().?);
            // batch_images[batch_count * 784 + i] = pixel_value / 255.0;
            // batch_images[batch_count * 784 + i] = pixel_value / 127.5 - 1;
            // batch_images[batch_count * 784 + i] = (pixel_value - 0.1307) / 0.3081;
            batch_images[batch_count * 784 + i] = pixel_value;
        }

        batch_count += 1;

        if (batch_count == batch_size or lines.peek() == null) {
            const image_tensor = try NDTensor(T).init(batch_images, &[_]usize{ batch_size, 1, 28, 28 }, true, allocator);
            const label_tensor = try NDTensor(T).init(batch_labels, &[_]usize{ batch_size, 10 }, true, allocator);
            try images.append(image_tensor);
            try labels.append(label_tensor);

            batch_images = try allocator.alloc(T, batch_size * 784);
            batch_labels = try allocator.alloc(T, batch_size * 10);
            batch_count = 0;
        }
    }

    if (batch_count > 0) {
        const image_tensor = try NDTensor(T).init(batch_images[0 .. batch_count * 784], &[_]usize{ batch_count, 1, 28, 28 }, true, allocator);
        const label_tensor = try NDTensor(T).init(batch_labels[0 .. batch_count * 10], &[_]usize{ batch_count, 10 }, true, allocator);
        try images.append(image_tensor);
        try labels.append(label_tensor);
    }

    return .{ .images = try images.toOwnedSlice(), .labels = try labels.toOwnedSlice() };
}

pub fn main() !void {
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // defer _ = gpa.deinit();
    // const allocator = gpa.allocator();
    // var allocator = std.heap.page_allocator;
    var acquired_arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer acquired_arena.deinit();

    var fw_arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer fw_arena.deinit();

    var bw_arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer bw_arena.deinit();

    var allocator = std.heap.c_allocator;
    log.info("Loading data...", .{});
    const batch_size = 8;
    const data = try loadMNIST(f64, allocator, "/tmp/mnist_train.csv", batch_size);
    data.labels[0].print();

    defer {
        for (data.images) |image| image.deinit();
        for (data.labels) |label| label.deinit();
        allocator.free(data.images);
        allocator.free(data.labels);
    }

    var model = try MNISTModel(f64).initSimple(acquired_arena.allocator());
    var trainer = Trainer(f64, .ce).init(
        model.model,
        0.1,
        .{
            .grad_clip_max_norm = 10.0,
            .grad_clip_delta = 1e-6,
            .grad_clip_enabled = false,
        },
    );

    defer {
        trainer.deinit();
        model.deinit();
    }

    var timer = try std.time.Timer.start();
    log.info("Training...", .{});
    const num_epochs = 2;
    for (0..num_epochs) |epoch| {
        var total_loss: f64 = 0;
        // TODO: impl/use trainer loop
        for (data.images, data.labels, 0..) |image, label, i| {
            timer.reset();
            const loss = try trainer.trainStep(
                image.setLabel("image_batch"),
                label.setLabel("label_batch"),
                bw_arena.allocator(),
                fw_arena.allocator(),
            );
            const t1 = timer.read();
            const ms_per_sample = @as(f64, @floatFromInt(t1 / std.time.ns_per_ms)) / @as(f64, @floatFromInt(batch_size));
            total_loss += loss.get(&[_]usize{0});
            log.info("Loss: {d:<5.5} [{d}/{d}] [ms/sample: {d}]", .{
                loss.data.data[0],
                i,
                data.images.len,
                ms_per_sample,
            });
            if (!bw_arena.reset(.retain_capacity)) log.warn("Issue in bw arena reset", .{});
            if (!fw_arena.reset(.retain_capacity)) log.warn("Issue in fw arena reset", .{});
        }
        const avg_loss = total_loss / @as(f32, @floatFromInt(data.images.len));
        log.info("Epoch {d}: Avg Loss = {d:.4}", .{ epoch + 1, avg_loss });
    }

    log.info("Training completed.\n", .{});
}

test main {
    try main();
}
