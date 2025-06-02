const std = @import("std");
const zg = @import("zigrad");
const DeviceReference = zg.DeviceReference;

const Tensor = zg.NDTensor;

const FileReader = struct {
    file: std.fs.File,
    buf: []u8,

    pub fn init(device: DeviceReference, path: []const u8) !FileReader {
        const file = try std.fs.cwd().openFile(path, .{});
        const file_size = try file.getEndPos();
        const buf = try file.readToEndAlloc(device.allocator, file_size);
        return .{
            .file = file,
            .buf = buf,
        };
    }

    pub fn deinit(self: *FileReader, device: DeviceReference) void {
        self.file.close();
        device.allocator.free(self.buf);
    }
};

pub fn Dataset(comptime T: type) type {
    return struct {
        const Self = @This();

        x: *Tensor(T),
        y: *Tensor(T),
        edge_index: *Tensor(usize),
        num_features: usize,
        num_classes: usize,
        train_mask: *Tensor(bool),
        eval_mask: *Tensor(bool),
        test_mask: *Tensor(bool),
        device: DeviceReference,

        pub fn load_cora(device: DeviceReference, node_path: []const u8, edge_path: []const u8) !Self {
            var node_file = try FileReader.init(device, node_path);
            defer node_file.deinit(device);
            var edge_file = try FileReader.init(device, edge_path);
            defer edge_file.deinit(device);

            const total_papers = 2708;
            const num_features = 1433;
            const num_classes = 7;
            var papers = try device.allocator.alloc(T, total_papers * num_features);
            defer device.allocator.free(papers);
            var classes = try device.allocator.alloc(T, total_papers * num_classes);
            defer device.allocator.free(classes);
            var masks = try device.allocator.alloc([]bool, 3);
            defer device.allocator.free(masks);
            for (0..3) |i| {
                masks[i] = try device.allocator.alloc(bool, total_papers);
            }
            defer for (0..3) |i| {
                device.allocator.free(masks[i]);
            };

            var i: usize = 0;
            var lines = std.mem.splitScalar(u8, node_file.buf, '\n');
            while (lines.next()) |line| {
                if (line.len == 0 or line[0] == '#') {
                    continue;
                }
                var values = std.mem.splitScalar(u8, line, ',');

                const class_label = try std.fmt.parseUnsigned(usize, values.next().?, 10);
                for (0..num_classes) |j| {
                    classes[i * num_classes + j] = @as(T, 0);
                    if (class_label == j) {
                        classes[i * num_classes + j] = @as(T, 1);
                    }
                }

                for (0..3) |j| {
                    masks[j][i] = (try std.fmt.parseUnsigned(u1, values.next().?, 10)) == 1;
                }

                for (0..num_features) |j| {
                    papers[i * num_features + j] = try std.fmt.parseFloat(T, values.next().?);
                }
                i += 1;
            }

            const total_cites = 10556;
            var cites = try device.allocator.alloc(usize, total_cites * 2);
            defer device.allocator.free(cites);

            i = 0;
            lines = std.mem.splitScalar(u8, edge_file.buf, '\n');
            while (lines.next()) |line| {
                if (line.len == 0 or line[0] == '#') {
                    continue;
                }
                var values = std.mem.splitScalar(u8, line, ',');

                const cited_index = try std.fmt.parseUnsigned(usize, values.next().?, 10);
                const citing_index = try std.fmt.parseUnsigned(usize, values.next().?, 10);

                cites[i * 2] = cited_index;
                cites[i * 2 + 1] = citing_index;
                i += 1;
            }

            return .{
                .x = try Tensor(T).init(papers, &[_]usize{ total_papers, num_features }, true, device),
                .y = try Tensor(T).init(classes, &[_]usize{ total_papers, num_classes }, true, device),
                .edge_index = try Tensor(usize).init(cites, &[_]usize{ 2, total_cites }, true, device),
                .num_features = num_features,
                .num_classes = num_classes,
                .train_mask = try zg.NDTensor(bool).init(masks[0], &[_]usize{ total_papers, 1 }, false, device),
                .eval_mask = try zg.NDTensor(bool).init(masks[1], &[_]usize{ total_papers, 1 }, false, device),
                .test_mask = try zg.NDTensor(bool).init(masks[2], &[_]usize{ total_papers, 1 }, false, device),
                .device = device,
            };
        }

        pub fn deinit(self: @This()) void {
            self.x.deinit();
            self.y.deinit();
            self.edge_index.deinit();

            self.train_mask.deinit();
            self.eval_mask.deinit();
            self.test_mask.deinit();
        }
    };
}
