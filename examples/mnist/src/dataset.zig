const std = @import("std");
const zg = @import("zigrad");
const DeviceReference = zg.DeviceReference;

pub fn MnistDataset(comptime T: type) type {
    return struct {
        images: []*zg.NDTensor(T),
        labels: []*zg.NDTensor(T),
        allocator: std.mem.Allocator,

        pub fn load(allocator: std.mem.Allocator, device: DeviceReference, csv_path: []const u8, batch_size: usize) !@This() {
            const file = try std.fs.cwd().openFile(csv_path, .{});
            defer file.close();

            const file_size = try file.getEndPos();
            const file_contents = try file.readToEndAlloc(allocator, file_size);
            defer allocator.free(file_contents);

            var images = std.ArrayList(*zg.NDTensor(T)).init(allocator);
            var labels = std.ArrayList(*zg.NDTensor(T)).init(allocator);

            var lines = std.mem.splitScalar(u8, file_contents, '\n');
            var batch_images = try allocator.alloc(T, batch_size * 784);
            defer allocator.free(batch_images);
            var batch_labels = try allocator.alloc(T, batch_size * 10);
            defer allocator.free(batch_labels);
            var batch_count: usize = 0;

            const config: zg.TensorOpts = .{
                .requires_grad = false,
                .acquired = true,
            };

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
                    const image_tensor = try zg.NDTensor(T).from_slice(device, batch_images, &.{ batch_size, 1, 28, 28 }, config);
                    const label_tensor = try zg.NDTensor(T).from_slice(device, batch_labels, &.{ batch_size, 10 }, config);
                    try images.append(image_tensor);
                    try labels.append(label_tensor);
                    batch_count = 0;
                }
            }

            if (batch_count > 0) { // remainder
                const image_tensor = try zg.NDTensor(T).from_slice(device, batch_images[0 .. batch_count * 784], &.{ batch_count, 1, 28, 28 }, config);
                const label_tensor = try zg.NDTensor(T).from_slice(device, batch_labels[0 .. batch_count * 10], &.{ batch_count, 10 }, config);
                try images.append(image_tensor);
                try labels.append(label_tensor);
            }

            return .{ .images = try images.toOwnedSlice(), .labels = try labels.toOwnedSlice(), .allocator = allocator };
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
