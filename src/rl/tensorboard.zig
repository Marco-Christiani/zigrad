const std = @import("std");
const pb = @import("protobuf");
const tb = @import("proto/tensorboard.pb.zig");

fn maskedCrc32c(data: []const u8) u32 {
    const x = std.hash.Crc32.hash(data);
    return (x >> 15 | x << 17) +% 0xa282ead8;
}

fn writeEvent(writer: anytype, event: *tb.Event, alloc: std.mem.Allocator) !void {
    const bytes = try event.encode(alloc);
    defer alloc.free(bytes);

    const length = @as(u64, @intCast(bytes.len));
    const length_bytes = std.mem.asBytes(&length);
    const length_crc = maskedCrc32c(length_bytes);
    const data_crc = maskedCrc32c(bytes);

    try writer.writeAll(length_bytes);
    try writer.writeInt(u32, length_crc, .little);
    try writer.writeAll(bytes);
    try writer.writeInt(u32, data_crc, .little);
}

fn addScalar(event: *tb.Event, tag: []const u8, value: f32, step: i64, alloc: std.mem.Allocator) !void {
    event.step = step;
    event.wall_time = @as(f64, @floatFromInt(std.time.milliTimestamp())) / 1000.0;

    var summary = tb.Summary.init(alloc);
    var summary_value = tb.Summary.Value.init(alloc);
    summary_value.tag = try pb.ManagedString.copy(tag, alloc);

    // SummaryMetadata
    var metadata = tb.SummaryMetadata.init(alloc);
    var plugin_data = tb.SummaryMetadata.PluginData.init(alloc);
    plugin_data.plugin_name = try pb.ManagedString.copy("scalars", alloc);
    try metadata.plugin_data.append(plugin_data);
    summary_value.metadata = metadata;

    // TensorProto for scalar
    var tensor = tb.TensorProto.init(alloc);
    tensor.dtype = .DT_FLOAT;
    try tensor.float_val.append(value);
    var tensor_shape = tb.TensorShapeProto.init(alloc);
    try tensor_shape.dim.append(.{ .size = 1 });
    tensor.tensor_shape = tensor_shape;

    summary_value.value = .{ .tensor = tensor };
    try summary.value.append(summary_value);

    event.what = .{ .summary = summary };
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const timestamp = @as(u64, @intCast(std.time.milliTimestamp())) / 1000;
    const file_name = try std.fmt.allocPrint(alloc, "/tmp/tensorboard_logs/events.out.tfevents.{d}", .{timestamp});
    defer alloc.free(file_name);

    const file = try std.fs.cwd().createFile(file_name, .{});
    defer file.close();

    var buffered_writer = std.io.bufferedWriter(file.writer());
    const writer = buffered_writer.writer();

    // write version
    var version_event = tb.Event.init(alloc);
    defer version_event.deinit();
    version_event.wall_time = @as(f64, @floatFromInt(std.time.milliTimestamp())) / 1000.0;
    version_event.what = .{ .file_version = try pb.ManagedString.copy("brain.Event:2", alloc) };
    try writeEvent(writer, &version_event, alloc);

    // write scalars
    var event = tb.Event.init(alloc);
    defer event.deinit();

    for (0..5) |i| {
        const step: i64 = @intCast(i);
        const value: f32 = @as(f32, @floatFromInt(i)) * 2.0;
        try addScalar(&event, "example/scalar", value, step, alloc);
        try writeEvent(writer, &event, alloc);
    }

    try buffered_writer.flush();
    std.debug.print("TensorBoard log file created: {s}\n", .{file_name});
}

pub fn example() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();
    var e = tb.Event.init(alloc);
    defer e.deinit();
    e.wall_time = @as(f64, @floatFromInt(std.time.timestamp()));
    // e.wall_time = @as(u64, @intCast(std.time.milliTimestamp())) / 1000;
    e.step = 0;

    var what = tb.Summary.init(alloc);

    var value = tb.Summary.Value.init(alloc);

    value.node_name = .Empty;
    value.metadata = null;
    value.tag = pb.ManagedString.static("something");
    value.value = .{ .simple_value = 32.1 };
    try what.value.append(value);
    e.what = .{ .summary = what };

    const bytes = try e.encode(alloc);
    defer alloc.free(bytes);
    std.debug.print("encoded: {X}\n", .{bytes});
    const decoded = try tb.Event.decode(bytes, alloc);
    defer decoded.deinit();
    std.debug.print("decoded: {any}\n", .{decoded});
}
