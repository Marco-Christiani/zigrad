const std = @import("std");
const pb = @import("protobuf");
const tb = @import("gen_proto/tensorboard.pb.zig");
const maskedCrc32c = @import("crc.zig").maskedCrc32c;

const Self = @This();
file: std.fs.File,
writer: std.fs.File.Writer,
buffer: std.io.BufferedWriter(4096, std.fs.File.Writer),
allocator: std.mem.Allocator,

pub fn init(log_dir: []const u8, allocator: std.mem.Allocator) !Self {
    const timestamp = @as(u64, @intCast(std.time.timestamp()));
    const full_log_dir = try std.fmt.allocPrint(allocator, "{s}/run-{d}", .{ log_dir, timestamp });
    defer allocator.free(full_log_dir);

    try std.fs.makeDirAbsolute(full_log_dir);

    const file_name = try std.fmt.allocPrint(allocator, "{s}/events.out.tfevents.{d}", .{ full_log_dir, timestamp });
    defer allocator.free(file_name);

    const file = try std.fs.createFileAbsolute(file_name, .{});
    errdefer file.close();

    var logger = Self{
        .file = file,
        .writer = file.writer(),
        .buffer = std.io.bufferedWriter(file.writer()),
        .allocator = allocator,
    };

    try logger.writeFileVersion();

    return logger;
}

pub fn deinit(self: *Self) void {
    self.buffer.flush() catch {};
    self.file.close();
}

fn writeFileVersion(self: *Self) !void {
    var event = tb.Event.init(self.allocator);
    defer event.deinit();

    event.wall_time = @as(f64, @floatFromInt(std.time.milliTimestamp())) / 1000.0;
    event.what = .{ .file_version = try pb.ManagedString.copy("brain.Event:2", self.allocator) };
    try self.writeEvent(&event);
}

fn writeEvent(self: *Self, event: *tb.Event) !void {
    const bytes = try event.encode(self.allocator);
    defer self.allocator.free(bytes);

    const length = @as(u64, @intCast(bytes.len));
    const length_bytes = std.mem.asBytes(&length);
    const length_crc = maskedCrc32c(length_bytes);
    const data_crc = maskedCrc32c(bytes);

    try self.buffer.writer().writeAll(length_bytes);
    try self.buffer.writer().writeInt(u32, length_crc, .little);
    try self.buffer.writer().writeAll(bytes);
    try self.buffer.writer().writeInt(u32, data_crc, .little);

    try self.buffer.flush();
}

pub fn addScalar(self: *Self, tag: []const u8, value: f32, step: i64) !void {
    var event = tb.Event.init(self.allocator);
    defer event.deinit();

    event.step = step;
    event.wall_time = @as(f64, @floatFromInt(std.time.milliTimestamp())) / 1000.0;

    var summary = tb.Summary.init(self.allocator);

    var summary_value = tb.Summary.Value.init(self.allocator);

    summary_value.tag = try pb.ManagedString.copy(tag, self.allocator);
    summary_value.value = .{ .simple_value = value };

    try summary.value.append(summary_value);
    event.what = .{ .summary = summary };

    try self.writeEvent(&event);
}

pub fn addHistogram(self: *Self, tag: []const u8, values: []const f64, step: i64) !void {
    var event = tb.Event.init(self.allocator);
    defer event.deinit();

    event.step = step;
    event.wall_time = @as(f64, @floatFromInt(std.time.timestamp()));

    var summary = tb.Summary.init(self.allocator);

    var summary_value = tb.Summary.Value.init(self.allocator);

    summary_value.tag = try pb.ManagedString.copy(tag, self.allocator);

    var histo = tb.HistogramProto.init(self.allocator);

    const min, const max = std.mem.minMax(f64, values);
    histo.min = min;
    histo.max = max;
    histo.num = @floatFromInt(values.len);
    histo.sum = blk: {
        var total: f64 = 0;
        for (values) |v| total += v;
        break :blk total;
    };
    histo.sum_squares = blk: {
        var total: f64 = 0;
        for (values) |v| total += v * v;
        break :blk total;
    };

    const num_buckets = 30;
    try computeBuckets(&histo, values, num_buckets);

    summary_value.value = .{ .histo = histo };

    try summary.value.append(summary_value);
    event.what = .{ .summary = summary };

    try self.writeEvent(&event);
}

fn computeBuckets(histo: *tb.HistogramProto, values: []const f64, num_buckets: usize) !void {
    const min = histo.min;
    const max = histo.max;
    const bucket_width = (max - min) / @as(f64, @floatFromInt(num_buckets));

    var buckets = try histo.bucket.allocator.alloc(f64, num_buckets);
    @memset(buckets, 0);

    var limits = try histo.bucket_limit.allocator.alloc(f64, num_buckets);
    for (0..num_buckets) |i| {
        limits[i] = min + bucket_width * @as(f64, @floatFromInt(i + 1));
    }

    for (values) |v| {
        const bucket_index = @min(@as(usize, @intFromFloat((v - min) / bucket_width)), num_buckets - 1);
        buckets[bucket_index] += 1;
    }

    histo.bucket = .{ .items = buckets, .capacity = buckets.len, .allocator = histo.bucket.allocator };
    histo.bucket_limit = .{ .items = limits, .capacity = limits.len, .allocator = histo.bucket_limit.allocator };
}
