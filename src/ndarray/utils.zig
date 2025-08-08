const std = @import("std");

pub fn prod(dims: []const usize) usize {
    if (dims.len == 0) return 0;
    var s: usize = 1;
    for (dims) |f| s *= f;
    return s;
}

fn _print_slice(comptime T: type, arr: []const T, shape: []const usize, writer: anytype, index: *usize) !void {
    if (shape.len == 1) {
        try writer.writeAll("[");
        var i: usize = 0;
        while (i < shape[0]) : (i += 1) {
            try writer.print("{d}", .{arr[index.*]});
            index.* += 1;
            if (i < shape[0] - 1) {
                try writer.writeAll(", ");
            }
        }
        try writer.writeAll("]");
        return;
    }

    try writer.writeAll("[");
    var i: usize = 0;
    while (i < shape[0]) : (i += 1) {
        try _print_slice(T, arr, shape[1..], writer, index);
        if (i < shape[0] - 1) {
            try writer.writeAll(", ");
        }
    }
    try writer.writeAll("]");
}

pub fn print_ndslice(comptime T: type, arr: []const T, shape: []const usize, writer: anytype) !void {
    var index: usize = 0;
    try _print_slice(T, arr, shape, writer, &index);
}

test print_ndslice {
    const allocator = std.testing.allocator;

    // 2D array
    {
        const arr1 = [_]f64{ 1, 2, 3, 4, 5, 6 };
        const shape1 = [_]usize{ 3, 2 };
        var buffer1 = std.ArrayList(u8).init(allocator);
        defer buffer1.deinit();
        try print_ndslice(f64, &arr1, &shape1, buffer1.writer());
        try std.testing.expectEqualStrings("[[1, 2], [3, 4], [5, 6]]", buffer1.items);
    }

    // 3D array
    {
        const arr2 = [_]f64{ 1, 2, 3, 4, 5, 6 };
        const shape2 = [_]usize{ 3, 2, 1 };
        var buffer2 = std.ArrayList(u8).init(allocator);
        defer buffer2.deinit();
        try print_ndslice(f64, &arr2, &shape2, buffer2.writer());
        try std.testing.expectEqualStrings("[[[1], [2]], [[3], [4]], [[5], [6]]]", buffer2.items);
    }
}
