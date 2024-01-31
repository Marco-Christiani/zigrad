const std = @import("std");
const zigrad = @import("grad.zig");

pub fn main() !void {
    std.debug.print("This is a debug message.", .{});

    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);
    const stdout = bw.writer();

    try stdout.print("Running...\n", .{});
    _ = try zigrad.linearModel(&std.heap.page_allocator, epoch_callback);
    try stdout.print("Done.\n", .{});

    try bw.flush(); // don't forget to flush!
}

fn epoch_callback(value: *const zigrad.Value, epoch_i: usize) anyerror!void {
    var allocator = std.heap.page_allocator;
    const graphJson = try zigrad.serializeValueToJson(allocator, value);
    const filename = try std.fmt.allocPrint(allocator, "outputs/graph_epoch_{}.json", .{epoch_i});
    defer allocator.free(filename);
    std.fs.cwd().makeDir("outputs") catch |err| switch (err) {
        error.PathAlreadyExists => std.debug.print("output/ already exists\n", .{}),
        else => |e| return e,
    };

    const file = try std.fs.cwd().createFile(filename, .{});
    defer file.close();
    const fileWriter = file.writer();
    try std.json.stringify(graphJson, .{}, fileWriter);
}

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}
