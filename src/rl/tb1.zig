const std = @import("std");

pub fn writeVarint(writer: anytype, value: u64) !void {
    var v = value;
    while (v > 0x7F) {
        try writer.writeByte(@intCast((v & 0x7F) | 0x80));
        v >>= 7;
    }
    try writer.writeByte(@intCast(v));
}

// fn maskedCrc32c(data: []const u8) [4]u8 {
//     const x = std.hash.Crc32.hash(data);
//     const masked = (x >> 15 | x << 17) +% 0xa282ead8;
//     return std.mem.toBytes(@as(u32, @truncate(masked)));
// }

// pub fn writeVarint(writer: anytype, value: u64) !void {
//     var v = value;
//     while (v > 0x7F) {
//         try writer.writeByte(@intCast(u8, (v & 0x7F) | 0x80));
//         v >>= 7;
//     }
//     try writer.writeByte(@intCast(u8, v));
// }

fn maskedCrc32c(data: []const u8) u32 {
    const x = std.hash.Crc32.hash(data);
    return (x >> 15 | x << 17) +% 0xa282ead8;
}

pub fn writeDouble(writer: anytype, value: f64) !void {
    const bytes = std.mem.asBytes(&value);
    try writer.writeAll(bytes);
}

pub fn writeString(writer: anytype, value: []const u8) !void {
    try writeVarint(writer, value.len);
    try writer.writeAll(value);
}

pub fn serializeEvent(writer: anytype, wall_time: f64, step: i64, what: enum { file_version, summary }) !void {
    // Event message
    try writeVarint(writer, 1 << 3 | 1); // field 1, wire type 1 (64-bit)
    try writeDouble(writer, wall_time);

    try writeVarint(writer, 2 << 3 | 0); // field 2, wire type 0 (varint)
    try writeVarint(writer, @as(u64, @intCast(step)));

    switch (what) {
        .file_version => {
            try writeVarint(writer, 3 << 3 | 2); // field 3, wire type 2 (length-delimited)
            try writeString(writer, "brain.Event:2");
        },
        .summary => {
            try writeVarint(writer, 5 << 3 | 2); // field 5, wire type 2 (length-delimited)
        },
    }
}

pub fn serializeScalarSummary(writer: anytype, tag: []const u8, value: f32) !void {
    // Summary message
    try writeVarint(writer, 1 << 3 | 2); // field 1, wire type 2 (length-delimited)

    var summary_data = std.ArrayList(u8).init(std.heap.page_allocator);
    defer summary_data.deinit();

    // Value submessage
    try writeVarint(summary_data.writer(), 1 << 3 | 2); // field 1, wire type 2 (length-delimited)

    var value_data = std.ArrayList(u8).init(std.heap.page_allocator);
    defer value_data.deinit();

    // Tag
    try writeVarint(value_data.writer(), 1 << 3 | 2); // field 1, wire type 2 (length-delimited)
    try writeString(value_data.writer(), tag);

    // Simple value
    try writeVarint(value_data.writer(), 2 << 3 | 5); // field 2, wire type 5 (32-bit)
    try value_data.writer().writeAll(std.mem.asBytes(&value));

    try writeVarint(summary_data.writer(), value_data.items.len);
    try summary_data.appendSlice(value_data.items);

    try writeVarint(writer, summary_data.items.len);
    try writer.writeAll(summary_data.items);
}

pub const TensorBoardLogger = struct {
    file: std.fs.File,
    writer: std.fs.File.Writer,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, log_dir: []const u8) !TensorBoardLogger {
        try std.fs.cwd().makePath(log_dir);
        const timestamp = @as(u64, @intCast(std.time.milliTimestamp())) / 1000;
        const file_name = try std.fmt.allocPrint(allocator, "{s}/events.out.tfevents.{d}", .{ log_dir, timestamp });
        defer allocator.free(file_name);

        const file = try std.fs.cwd().createFile(file_name, .{});
        var logger = TensorBoardLogger{
            .file = file,
            .writer = file.writer(),
            .allocator = allocator,
        };

        try logger.writeFileVersion();
        return logger;
    }

    pub fn deinit(self: *TensorBoardLogger) void {
        self.file.close();
    }

    fn writeFileVersion(self: *TensorBoardLogger) !void {
        var event_data = std.ArrayList(u8).init(self.allocator);
        defer event_data.deinit();

        try serializeEvent(event_data.writer(), @as(f64, @floatFromInt(std.time.milliTimestamp())) / 1000.0, 0, .file_version);

        try self.writeEvent(event_data.items);
    }

    fn writeEvent(self: *TensorBoardLogger, event_data: []const u8) !void {
        const event_length = @as(u64, @intCast(event_data.len));
        const length_crc = maskedCrc32c(std.mem.asBytes(&event_length));
        const data_crc = maskedCrc32c(event_data);

        try self.writer.writeInt(u64, event_length, .little);
        try self.writer.writeInt(u32, length_crc, .little);
        try self.writer.writeAll(event_data);
        try self.writer.writeInt(u32, data_crc, .little);

        try self.file.sync();
    }

    pub fn addScalar(self: *TensorBoardLogger, tag: []const u8, value: f32, step: i64) !void {
        var event_data = std.ArrayList(u8).init(self.allocator);
        defer event_data.deinit();

        try serializeEvent(event_data.writer(), @as(f64, @floatFromInt(std.time.milliTimestamp())) / 1000.0, step, .summary);
        try serializeScalarSummary(event_data.writer(), tag, value);

        try self.writeEvent(event_data.items);
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var logger = try TensorBoardLogger.init(allocator, "/tmp/tensorboard_logs");
    defer logger.deinit();

    var i: i64 = 0;
    while (i < 5) : (i += 1) {
        const value = @as(f32, @floatFromInt(i)) * 2.0;
        try logger.addScalar("example/scalar", value, i);
    }

    std.debug.print("TensorBoard log file created in the '/tmp/tensorboard_logs' directory.\n", .{});
}
