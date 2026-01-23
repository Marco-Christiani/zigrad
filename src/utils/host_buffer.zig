/// Host Buffer Utils
///
/// Host-side buffer management with shape and dtype metadata.
/// No device buffer operations, those belong to backend.Buffer.
const std = @import("std");

pub const DType = enum {
    f32,
    f64,
    i32,
    i64,
    u32,
    u64,

    pub fn size_in_bytes(self: DType) usize {
        return switch (self) {
            .f32, .i32, .u32 => 4,
            .f64, .i64, .u64 => 8,
        };
    }

    pub fn name(self: DType) []const u8 {
        return switch (self) {
            .f32 => "f32",
            .f64 => "f64",
            .i32 => "i32",
            .i64 => "i64",
            .u32 => "u32",
            .u64 => "u64",
        };
    }
};

pub const Shape = struct {
    dims: []const usize,

    pub fn num_elements(self: Shape) usize {
        var count: usize = 1;
        for (self.dims) |d| count *= d;
        return count;
    }

    pub fn rank(self: Shape) usize {
        return self.dims.len;
    }

    pub fn format(self: Shape, allocator: std.mem.Allocator) ![]const u8 {
        if (self.dims.len == 0) return try allocator.dupe(u8, "scalar");

        // Build format string with all dimensions
        var result = std.ArrayList(u8).initCapacity(allocator, 32) catch
            return try allocator.dupe(u8, "[...]");
        defer result.deinit(allocator);

        const writer = result.writer(allocator);
        try writer.writeAll("[");
        for (self.dims, 0..) |d, i| {
            if (i > 0) try writer.writeAll(", ");
            try writer.print("{d}", .{d});
        }
        try writer.writeAll("]");

        return result.toOwnedSlice(allocator);
    }
};

pub const HostBuffer = struct {
    data: []align(8) u8,
    shape: Shape,
    dtype: DType,
    allocator: std.mem.Allocator,
    owns_dims: bool,

    pub fn init(allocator: std.mem.Allocator, shape: Shape, dtype: DType) !HostBuffer {
        const num_bytes = shape.num_elements() * dtype.size_in_bytes();
        const data = try allocator.alignedAlloc(u8, .@"8", num_bytes);

        // Allocate and copy shape dims
        const dims = try allocator.dupe(usize, shape.dims);

        return HostBuffer{
            .data = data,
            .shape = Shape{ .dims = dims },
            .dtype = dtype,
            .allocator = allocator,
            .owns_dims = true,
        };
    }

    pub fn from_slice(allocator: std.mem.Allocator, data: anytype, shape: Shape, dtype: DType) !HostBuffer {
        const DataType = @TypeOf(data);
        const data_info = @typeInfo(DataType);

        if (data_info != .pointer) {
            @compileError("from_slice requires a pointer or slice type, got: " ++ @typeName(DataType));
        }

        // Accept both slices and pointers to arrays
        if (data_info.pointer.size != .slice and data_info.pointer.size != .one) {
            @compileError("from_slice requires a slice or pointer-to-array, got pointer size: " ++ @tagName(data_info.pointer.size));
        }

        const num_bytes = shape.num_elements() * dtype.size_in_bytes();
        const data_bytes = try allocator.alignedAlloc(u8, .@"8", num_bytes);

        // Copy data as bytes (works for both slices and pointer-to-array)
        const src_bytes = std.mem.sliceAsBytes(data);
        if (src_bytes.len < num_bytes) return error.InsufficientData;
        @memcpy(data_bytes, src_bytes[0..num_bytes]);

        const dims = try allocator.dupe(usize, shape.dims);

        return HostBuffer{
            .data = data_bytes,
            .shape = Shape{ .dims = dims },
            .dtype = dtype,
            .allocator = allocator,
            .owns_dims = true,
        };
    }

    pub fn deinit(self: *HostBuffer) void {
        self.allocator.free(self.data);
        if (self.owns_dims) {
            self.allocator.free(self.shape.dims);
        }
    }

    /// Fill buffer with a scalar value
    pub fn fill(self: *HostBuffer, comptime T: type, value: T) void {
        const count = self.shape.num_elements();
        const data_slice: []T = @alignCast(std.mem.bytesAsSlice(T, self.data));
        for (data_slice[0..count]) |*elem| {
            elem.* = value;
        }
    }

    /// View buffer as typed slice
    pub fn as_slice(self: *HostBuffer, comptime T: type) []T {
        return @alignCast(std.mem.bytesAsSlice(T, self.data));
    }

    /// Print buffer contents (for debugging)
    pub fn print(self: *HostBuffer, writer: anytype) !void {
        const shape_str = try self.shape.format(self.allocator);
        defer self.allocator.free(shape_str);

        try writer.print("HostBuffer({s}, {s}): ", .{ self.dtype.name(), shape_str });

        const max_print = @min(self.shape.num_elements(), 16);

        switch (self.dtype) {
            .f32 => {
                const slice = self.as_slice(f32);
                try writer.writeAll("[");
                for (slice[0..max_print], 0..) |val, i| {
                    if (i > 0) try writer.writeAll(", ");
                    try writer.print("{d:.2}", .{val});
                }
                if (max_print < self.shape.num_elements()) {
                    try writer.writeAll(", ...");
                }
                try writer.writeAll("]");
            },
            .f64 => {
                const slice = self.as_slice(f64);
                try writer.writeAll("[");
                for (slice[0..max_print], 0..) |val, i| {
                    if (i > 0) try writer.writeAll(", ");
                    try writer.print("{d:.2}", .{val});
                }
                if (max_print < self.shape.num_elements()) {
                    try writer.writeAll(", ...");
                }
                try writer.writeAll("]");
            },
            .i32 => {
                const slice = self.as_slice(i32);
                try writer.writeAll("[");
                for (slice[0..max_print], 0..) |val, i| {
                    if (i > 0) try writer.writeAll(", ");
                    try writer.print("{d}", .{val});
                }
                if (max_print < self.shape.num_elements()) {
                    try writer.writeAll(", ...");
                }
                try writer.writeAll("]");
            },
            else => try writer.writeAll("<unsupported dtype for print>"),
        }

        try writer.writeAll("\n");
    }
};

test "HostBuffer basic operations" {
    const allocator = std.testing.allocator;

    var buf = try HostBuffer.init(allocator, .{ .dims = &[_]usize{ 2, 3 } }, .f32);
    defer buf.deinit();

    try std.testing.expectEqual(@as(usize, 6), buf.shape.num_elements());
    try std.testing.expectEqual(@as(usize, 24), buf.data.len);

    buf.fill(f32, 42.0);
    const slice = buf.as_slice(f32);
    try std.testing.expectEqual(@as(f32, 42.0), slice[0]);
    try std.testing.expectEqual(@as(f32, 42.0), slice[5]);
}

test "HostBuffer from_slice" {
    const allocator = std.testing.allocator;

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var buf = try HostBuffer.from_slice(allocator, &data, .{ .dims = &[_]usize{4} }, .f32);
    defer buf.deinit();

    const slice = buf.as_slice(f32);
    try std.testing.expectEqual(@as(f32, 1.0), slice[0]);
    try std.testing.expectEqual(@as(f32, 4.0), slice[3]);
}
