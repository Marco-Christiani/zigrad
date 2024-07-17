const std = @import("std");

pub fn prod(dims: []const usize) usize {
    if (dims.len == 0) return 0;
    var s: usize = 1;
    for (dims) |f| s *= f;
    return s;
}

pub fn calculateBroadcastedShape(shape1: []const usize, shape2: []const usize, allocator: std.mem.Allocator) ![]usize {
    const dims = @max(shape1.len, shape2.len);
    const result_shape = try allocator.alloc(usize, dims);
    var i: usize = 0;
    while (i < dims) : (i += 1) {
        const dimA = if (i < shape1.len) shape1[shape1.len - 1 - i] else 1;
        const dimB = if (i < shape2.len) shape2[shape2.len - 1 - i] else 1;
        if (dimA != dimB and dimA != 1 and dimB != 1) {
            allocator.free(result_shape);
            return error.Unbroadcastable;
        }
        result_shape[dims - 1 - i] = @max(dimA, dimB);
    }
    return result_shape;
}

/// Flexibly select index allowing for indices.len > shape.len
/// Effectively mocks batch dimensions (e.g. shape=(2,2), indices=(0, 0) == indices=(1, 0, 0) == indices= (..., 0, 0))
fn flexSelectOffset(shape: []const usize, indices: []const usize) !usize {
    if (indices.len < shape.len) {
        return error.InvalidIndex; // should be slicing, not selecting a single index
    }
    var index: usize = 0;
    var stride: usize = 1;
    for (0..shape.len) |i| {
        const shape_i = shape.len - i - 1;
        const indices_i = indices.len - i - 1;
        const dimSize = shape[shape_i];
        const idx = indices[indices_i];
        std.debug.assert(idx < dimSize);

        index += idx * stride;
        stride *= dimSize;
    }
    return index;
}

fn _printSlice(comptime T: type, arr: []const T, shape: []const usize, writer: anytype, index: *usize) !void {
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
        try _printSlice(T, arr, shape[1..], writer, index);
        if (i < shape[0] - 1) {
            try writer.writeAll(", ");
        }
    }
    try writer.writeAll("]");
}

pub fn printNDSlice(comptime T: type, arr: []const T, shape: []const usize, writer: anytype) !void {
    var index: usize = 0;
    try _printSlice(T, arr, shape, writer, &index);
}

test printNDSlice {
    const allocator = std.testing.allocator;

    // 2D array
    {
        const arr1 = [_]f64{ 1, 2, 3, 4, 5, 6 };
        const shape1 = [_]usize{ 3, 2 };
        var buffer1 = std.ArrayList(u8).init(allocator);
        defer buffer1.deinit();
        try printNDSlice(f64, &arr1, &shape1, buffer1.writer());
        try std.testing.expectEqualStrings("[[1, 2], [3, 4], [5, 6]]", buffer1.items);
    }

    // 3D array
    {
        const arr2 = [_]f64{ 1, 2, 3, 4, 5, 6 };
        const shape2 = [_]usize{ 3, 2, 1 };
        var buffer2 = std.ArrayList(u8).init(allocator);
        defer buffer2.deinit();
        try printNDSlice(f64, &arr2, &shape2, buffer2.writer());
        try std.testing.expectEqualStrings("[[[1], [2]], [[3], [4]], [[5], [6]]]", buffer2.items);
    }
}

test "calculateBroadcastedShape" {
    const shape1 = [_]usize{ 5, 3, 4, 2 };
    const shape2 = [_]usize{ 4, 2 };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var result_shape = try calculateBroadcastedShape(&shape1, &shape2, alloc);
    try std.testing.expectEqualSlices(usize, &shape1, result_shape);

    result_shape = try calculateBroadcastedShape(&shape2, &shape1, alloc);
    try std.testing.expectEqualSlices(usize, &shape1, result_shape);

    result_shape = try calculateBroadcastedShape(&shape1, &shape1, alloc);
    try std.testing.expectEqualSlices(usize, &shape1, result_shape);

    result_shape = try calculateBroadcastedShape(&shape2, &shape2, alloc);
    try std.testing.expectEqualSlices(usize, &shape2, result_shape);

    try std.testing.expectError(error.Unbroadcastable, calculateBroadcastedShape(&shape1, &[_]usize{ 4, 3 }, alloc));
    try std.testing.expectError(error.Unbroadcastable, calculateBroadcastedShape(&shape1, &[_]usize{ 3, 2 }, alloc));
    try std.testing.expectError(error.Unbroadcastable, calculateBroadcastedShape(&shape1, &[_]usize{ 3, 3 }, alloc));
}

test "flexPosToIndex" {
    // arr = [[0, 1, 2]
    //        [3, 4, 5]]
    // arr[1, 1] == 4
    try std.testing.expectEqual(4, flexSelectOffset(&[_]usize{ 2, 3 }, &[_]usize{ 1, 1 }));
    // arr[1, 1, 1] == 4
    try std.testing.expectEqual(4, flexSelectOffset(&[_]usize{ 2, 3 }, &[_]usize{ 1, 1, 1 }));

    // arr = [[[0, 1, 2]
    //        [3, 4, 5]]]
    // arr[1, 1, 1] == 4
    try std.testing.expectEqual(4, flexSelectOffset(&[_]usize{ 1, 2, 3 }, &[_]usize{ 0, 1, 1 }));
}
