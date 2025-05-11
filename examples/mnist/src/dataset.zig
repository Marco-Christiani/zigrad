const std = @import("std");
const zg = @import("zigrad");
const DeviceReference = zg.DeviceReference;

pub fn MnistDataset(comptime T: type) type {
    return struct {
        images: []*zg.NDTensor(T),
        labels: []*zg.NDTensor(T),
        allocator: std.mem.Allocator,

        pub fn load(gm: *zg.GraphManager, device: DeviceReference, csv_path: []const u8, batch_size: usize) !@This() {
            const file = try std.fs.cwd().openFile(csv_path, .{});
            defer file.close();

            const file_size = try file.getEndPos();
            const file_contents = try file.readToEndAlloc(gm.allocator, file_size);
            defer gm.allocator.free(file_contents);

            var images = std.ArrayList(*zg.NDTensor(T)).init(gm.allocator);
            var labels = std.ArrayList(*zg.NDTensor(T)).init(gm.allocator);

            var lines = std.mem.splitScalar(u8, file_contents, '\n');
            var batch_images = try gm.allocator.alloc(T, batch_size * 784);
            defer gm.allocator.free(batch_images);
            var batch_labels = try gm.allocator.alloc(T, batch_size * 10);
            defer gm.allocator.free(batch_labels);
            var batch_count: usize = 0;

            const config: zg.TensorConfig = .{
                .requires_grad = true,
                .device = device,
                .acquired = true,
                .heap = gm.heap(),
            };

            while (lines.next()) |line| {
                if (line.len == 0) continue; // skip empty lines
                var values = std.mem.splitScalar(u8, line, ',');

                for (0..10) |i| {
                    batch_labels[batch_count * 10 + i] = @as(T, @floatFromInt(try std.fmt.parseUnsigned(u8, values.next().?, 10)));
                }

                for (0..784) |i| {
                    const pixel_value = try std.fmt.parseFloat(T, values.next().?);
                    batch_images[batch_count * 784 + i] = pixel_value;
                }

                batch_count += 1;

                if (batch_count == batch_size or lines.peek() == null) {
                    const image_tensor = try zg.NDTensor(T).from_slice(batch_images, &.{ batch_size, 1, 28, 28 }, config);
                    const label_tensor = try zg.NDTensor(T).from_slice(batch_labels, &.{ batch_size, 10 }, config);
                    try images.append(image_tensor);
                    try labels.append(label_tensor);
                    batch_count = 0;
                }
            }

            if (batch_count > 0) { // remainder
                const image_tensor = try zg.NDTensor(T).from_slice(batch_images[0 .. batch_count * 784], &.{ batch_count, 1, 28, 28 }, config);
                const label_tensor = try zg.NDTensor(T).from_slice(batch_labels[0 .. batch_count * 10], &.{ batch_count, 10 }, config);
                try images.append(image_tensor);
                try labels.append(label_tensor);
            }

            return .{ .images = try images.toOwnedSlice(), .labels = try labels.toOwnedSlice(), .allocator = gm.allocator };
        }

        pub fn deinit(self: @This()) void {
            for (self.images, self.labels) |image, label| {
                image.release();
                label.release();
                image.deinit();
                label.deinit();
            }
            self.allocator.free(self.images);
            self.allocator.free(self.labels);
        }
    };
}
