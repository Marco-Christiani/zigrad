const std = @import("std");
const Model = @import("model.zig").Model;
const Conv2DLayer = @import("layer.zig").Conv2DLayer;
const LinearLayer = @import("layer.zig").LinearLayer;
const ReLULayer = @import("layer.zig").ReLULayer;
const ops = @import("ops.zig");
const Trainer = @import("trainer.zig").Trainer;
const NDTensor = @import("tensor.zig").NDTensor;
// const zg = @import("zigrad");
// const NDTensor = zg.tensor.NDTensor;

const log = std.log.scoped(.zg_mnist);

pub fn MNISTModel(comptime T: type) type {
    return struct {
        const Self = @This();
        model: Model(T),
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) !Self {
            var self = Self{
                .allocator = allocator,
                .model = try Model(T).init(allocator),
            };

            var conv1 = try Conv2DLayer(T).init(allocator, 1, 32, 3, 1, 1, 1);
            var relu1 = try ReLULayer(T).init(allocator);
            var conv2 = try Conv2DLayer(T).init(allocator, 32, 64, 3, 1, 1, 1);
            var relu2 = try ReLULayer(T).init(allocator);
            var fc1 = try LinearLayer(T).init(allocator, 64 * 28 * 28, 128);
            var relu3 = try ReLULayer(T).init(allocator);
            var fc2 = try LinearLayer(T).init(allocator, 128, 10);
            var relu4 = try ReLULayer(T).init(allocator);

            try self.model.addLayer(conv1.asLayer());
            try self.model.addLayer(relu1.asLayer());
            try self.model.addLayer(conv2.asLayer());
            try self.model.addLayer(relu2.asLayer());
            try self.model.addLayer(fc1.asLayer());
            try self.model.addLayer(relu3.asLayer());
            try self.model.addLayer(fc2.asLayer());
            try self.model.addLayer(relu4.asLayer());
            return self;
        }

        pub fn deinit(self: *Self) void {
            self.model.deinit();
            self.allocator.destroy(self);
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

pub fn loadMNIST(allocator: std.mem.Allocator, filepath: []const u8, batch_size: usize) !struct { images: []*NDTensor(f32), labels: []*NDTensor(f32) } {
    const file = try std.fs.cwd().openFile(filepath, .{});
    defer file.close();

    const file_size = try file.getEndPos();
    const file_contents = try file.readToEndAlloc(allocator, file_size);
    defer allocator.free(file_contents);

    var images = std.ArrayList(*NDTensor(f32)).init(allocator);
    var labels = std.ArrayList(*NDTensor(f32)).init(allocator);

    var lines = std.mem.split(u8, file_contents, "\n");
    var batch_images = try allocator.alloc(f32, batch_size * 784);
    var batch_labels = try allocator.alloc(f32, batch_size * 10);
    var batch_count: usize = 0;

    while (lines.next()) |line| {
        if (line.len == 0) continue; // Skip empty lines
        var values = std.mem.split(u8, line, ",");
        const label = try std.fmt.parseUnsigned(u8, values.next().?, 10);

        for (0..10) |i| {
            batch_labels[batch_count * 10 + i] = if (i == label) 1 else 0;
        }

        for (0..784) |i| {
            const pixel_value = try std.fmt.parseFloat(f32, values.next().?);
            batch_images[batch_count * 784 + i] = pixel_value / 255.0;
        }

        batch_count += 1;

        if (batch_count == batch_size) {
            const image_tensor = try NDTensor(f32).init(batch_images, &[_]usize{ batch_size, 1, 28, 28 }, true, allocator);
            const label_tensor = try NDTensor(f32).init(batch_labels, &[_]usize{ batch_size, 10 }, true, allocator);
            try images.append(image_tensor);
            try labels.append(label_tensor);

            batch_images = try allocator.alloc(f32, batch_size * 784);
            batch_labels = try allocator.alloc(f32, batch_size * 10);
            batch_count = 0;
        }
    }

    if (batch_count > 0) {
        const image_tensor = try NDTensor(f32).init(batch_images[0 .. batch_count * 784], &[_]usize{ batch_count, 1, 28, 28 }, true, allocator);
        const label_tensor = try NDTensor(f32).init(batch_labels[0 .. batch_count * 10], &[_]usize{ batch_count, 10 }, true, allocator);
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
    var allocator = std.heap.c_allocator;
    log.info("Loading data...", .{});
    const batch_size = 64;
    const data = try loadMNIST(allocator, "/tmp/mnist_train.csv", batch_size);

    defer {
        for (data.images) |image| image.deinit();
        for (data.labels) |label| label.deinit();
        allocator.free(data.images);
        allocator.free(data.labels);
    }

    var model = try MNISTModel(f32).init(allocator);
    var trainer = Trainer(f32, .ce).init(
        model.model,
        0.0003,
        .{
            .grad_clip_max_norm = 10.0,
            .grad_clip_delta = 1e-6,
            .grad_clip_enabled = true,
        },
        allocator,
    );

    defer {
        trainer.deinit();
        model.deinit();
    }

    log.info("Training...", .{});
    const num_epochs = 5;
    for (0..num_epochs) |epoch| {
        var total_loss: f32 = 0;
        // TODO: impl/use trainer loop
        for (data.images, data.labels, 0..) |image, label, i| {
            const loss = try trainer.trainStep(image.setLabel("image"), label.setLabel("label"));
            total_loss += loss.get(&[_]usize{0});
            log.info("Loss: {d:<5.5} {d:<4.2} [{d}/{d}]", .{ loss.data.data[0], @as(f32, @floatFromInt(i / data.images.len)), i, data.images.len });
            // loss.teardown();
            // break;
        }
        const avg_loss = total_loss / @as(f32, @floatFromInt(data.images.len));
        log.info("Epoch {d}: Avg Loss = {d:.4}", .{ epoch + 1, avg_loss });
    }

    log.info("Training completed.\n", .{});
}
