const std = @import("std");

pub const EventFileWriter = struct {
    file: std.fs.File,

    pub fn init(path: []const u8) !EventFileWriter {
        const file = try std.fs.cwd().createFile(path, .{ .read = true });
        return EventFileWriter{ .file = file };
    }

    pub fn writeEvent(self: *EventFileWriter, event: Event) !void {
        const data = try event.serialize();
        defer std.heap.page_allocator.free(data);

        const header = std.mem.toBytes(@as(u64, @intCast(data.len)));
        try self.file.writer().writeAll(&header);
        try self.file.writer().writeAll(&maskedCrc32c(&header));
        try self.file.writer().writeAll(data);
        try self.file.writer().writeAll(&maskedCrc32c(data));
    }
    pub fn flush(self: *EventFileWriter) !void {
        try self.file.sync();
    }
    pub fn close(self: *EventFileWriter) void {
        self.file.close();
    }
};

pub const Event = struct {
    wall_time: f64,
    step: i64,
    what: u32 = 1, // New field
    summary: ?Summary = null,

    pub fn serialize(self: Event) ![]u8 {
        var buffer = std.ArrayList(u8).init(std.heap.page_allocator);
        errdefer buffer.deinit();

        try writeField(&buffer, 1, self.wall_time);
        try writeField(&buffer, 2, self.step);
        try writeField(&buffer, 3, self.what); // Write the new field

        if (self.summary) |summary| {
            const summary_data = try summary.serialize();
            defer std.heap.page_allocator.free(summary_data);
            try writeField(&buffer, 4, summary_data);
        }

        return buffer.toOwnedSlice();
    }
};
pub const Summary = struct {
    value: Value,

    pub fn serialize(self: Summary) ![]u8 {
        var buffer = std.ArrayList(u8).init(std.heap.page_allocator);
        errdefer buffer.deinit();

        const value_data = try self.value.serialize();
        defer std.heap.page_allocator.free(value_data);
        try writeField(&buffer, 1, value_data);

        return buffer.toOwnedSlice();
    }
};

pub const Value = struct {
    tag: []const u8,
    simple_value: f32,

    pub fn serialize(self: Value) ![]u8 {
        var buffer = std.ArrayList(u8).init(std.heap.page_allocator);
        errdefer buffer.deinit();

        try writeField(&buffer, 1, self.tag);
        try writeField(&buffer, 2, self.simple_value);

        return buffer.toOwnedSlice();
    }
};

fn writeField(buffer: *std.ArrayList(u8), field_number: u32, value: anytype) !void {
    const T = @TypeOf(value);
    const wire_type: u32 = switch (T) {
        []const u8, []u8 => 2,
        f64 => 1,
        f32 => 5,
        i64, u64, i32, u32 => 0,
        else => @compileError("Unsupported type"),
    };
    const tag = (field_number << 3) | wire_type;
    try writeVarint(buffer, tag);

    switch (T) {
        []const u8, []u8 => {
            try writeVarint(buffer, @intCast(value.len));
            try buffer.appendSlice(value);
        },
        f64 => try buffer.writer().writeInt(u64, @as(u64, @bitCast(value)), .little),
        f32 => try buffer.writer().writeInt(u32, @as(u32, @bitCast(value)), .little),
        i64, u64, i32, u32 => try writeVarint(buffer, value),
        else => unreachable,
    }
}

fn writeVarint(buffer: *std.ArrayList(u8), value: anytype) !void {
    var v = @as(u64, @intCast(value));
    while (v > 127) {
        try buffer.append(@as(u8, @truncate(v)) | 0x80);
        v >>= 7;
    }
    try buffer.append(@as(u8, @truncate(v)));
}

fn maskedCrc32c(data: []const u8) [4]u8 {
    const x = std.hash.Crc32.hash(data);
    const masked = (x >> 15 | x << 17) +% 0xa282ead8;
    return std.mem.toBytes(@as(u32, @truncate(masked)));
}

pub const SummaryWriter = struct {
    event_writer: EventFileWriter,

    pub fn init(logdir: []const u8) !SummaryWriter {
        const path = try std.fmt.allocPrint(std.heap.page_allocator, "{s}/events.out.tfevents.{d}", .{ logdir, std.time.timestamp() });
        defer std.heap.page_allocator.free(path);
        const event_writer = try EventFileWriter.init(path);
        return SummaryWriter{ .event_writer = event_writer };
    }

    pub fn addScalar(self: *SummaryWriter, tag: []const u8, value: f32, step: i64) !void {
        var timer = try std.time.Timer.start();
        const nanos = timer.read();
        const event = Event{
            .wall_time = @as(f64, @floatFromInt(nanos)) / 1e9,
            .step = step,
            .what = 1,
            .summary = Summary{
                .value = Value{
                    .tag = tag,
                    .simple_value = value,
                },
            },
        };
        try self.event_writer.writeEvent(event);
    }

    pub fn flush(self: *SummaryWriter) !void {
        try self.event_writer.flush();
    }

    pub fn close(self: *SummaryWriter) void {
        self.event_writer.close();
    }
};

pub fn main() !void {
    var writer = try SummaryWriter.init("/tmp/tensorboard_logs");
    defer writer.close();

    for (0..5) |i| {
        try writer.addScalar("test_metric", @as(f32, @floatFromInt(i * 2)), @intCast(i));
    }
    try writer.flush();

    std.debug.print("Finished writing events. Check the log directory.\n", .{});
}
