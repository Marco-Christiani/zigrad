const std = @import("std");
const zg = @import("zigrad");
const DeviceReference = zg.DeviceReference;

pub fn MnistDataset(comptime T: type) type {
    return struct {
        images: []*zg.NDTensor(T),
        labels: []*zg.NDTensor(T),
        device: DeviceReference,

        pub fn load(device: DeviceReference, csv_path: []const u8, batch_size: usize) !@This() {
            const file = try std.fs.cwd().openFile(csv_path, .{});
            defer file.close();

            const file_size = try file.getEndPos();
            const file_contents = try file.readToEndAlloc(device.allocator, file_size);
            defer device.allocator.free(file_contents);

            var images = std.ArrayList(*zg.NDTensor(T)).init(device.allocator);
            var labels = std.ArrayList(*zg.NDTensor(T)).init(device.allocator);

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
                    batch_images[batch_count * 784 + i] = pixel_value;
                }

                batch_count += 1;

                if (batch_count == batch_size or lines.peek() == null) {
                    const image_tensor = try zg.NDTensor(T).init(batch_images, &[_]usize{ batch_size, 1, 28, 28 }, true, device);
                    const label_tensor = try zg.NDTensor(T).init(batch_labels, &[_]usize{ batch_size, 10 }, true, device);
                    image_tensor.acquire();
                    label_tensor.acquire();
                    try images.append(image_tensor);
                    try labels.append(label_tensor);
                    batch_count = 0;
                }
            }

            if (batch_count > 0) { // remainder
                const image_tensor = try zg.NDTensor(T).init(batch_images[0 .. batch_count * 784], &[_]usize{ batch_count, 1, 28, 28 }, true, device);
                const label_tensor = try zg.NDTensor(T).init(batch_labels[0 .. batch_count * 10], &[_]usize{ batch_count, 10 }, true, device);
                image_tensor.acquire();
                label_tensor.acquire();
                try images.append(image_tensor);
                try labels.append(label_tensor);
            }

            return .{ .images = try images.toOwnedSlice(), .labels = try labels.toOwnedSlice(), .device = device };
        }

        pub fn deinit(self: @This()) void {
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
}
