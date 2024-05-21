const std = @import("std");
const Op = @import("grad.zig").Op;
const generateASCIIUUID = @import("grad.zig").generateASCIIUUID;

pub fn NDTensor(comptime T: type, comptime shape: []const u32) type {
    const size: u32 = blk: {
        var n: u32 = 1;
        for (shape) |d| {
            n *= d;
        }
        break :blk n;
    };

    return struct {
        const Self = @This();
        data: [size]T,
        op: ?Op = null,
        children: [2]*anyopaque = undefined,
        label: ?[]const u8 = null,

        pub fn init(values: [size]T, allocator: std.mem.Allocator) *Self {
            var data: [size]T = undefined;
            for (values, 0..) |val, i| {
                data[i] = val;
            }
            const result = allocator.create(Self) catch {
                std.debug.panic("NDTensor allocation failed.\n", .{});
            };
            result.* = Self{ .data = data };
            return result;
        }

        pub fn setLabel(self: *Self, comptime label: ?[]const u8) *Self {
            const uid = generateASCIIUUID(4);
            if (label) |l| {
                var buf: [l.len + 4]u8 = undefined;
                _ = std.fmt.bufPrint(&buf, "{?s}{s}", .{ l, uid }) catch {
                    @panic("Failed to set label");
                };
                self.label = &buf;
            } else {
                self.label = uid;
            }
            return self;
        }

        pub fn fill(self: *Self, val: T) void {
            for (self.data) |*elem| {
                elem.* = val;
            }
        }

        pub fn get(self: Self, indices: [shape.len]u32) T {
            const index = self.posToIndex(indices);
            return self.data[index];
        }

        pub fn set(self: *Self, indices: [shape.len]u32, value: T) void {
            const index = self.posToIndex(indices);
            self.data[index] = value;
        }

        fn posToIndex(self: Self, indices: [shape.len]u32) u32 {
            _ = self;
            std.debug.assert(indices.len == shape.len);
            var index: u32 = 0;
            var stride: u32 = 1;
            for (0..shape.len) |i| {
                const dim = shape.len - i - 1;
                const dimSize = shape[dim];
                const idx = indices[dim];
                std.debug.assert(idx < dimSize);

                index += idx * stride;
                stride *= dimSize;
            }
            return index;
        }

        fn indexToPos(self: Self, index: u32) [shape.len]u32 {
            _ = self;
            var pos: [shape.len]u32 = undefined;
            var remainingIndex = index;
            var stride: u32 = 1;
            for (0..shape.len) |i| {
                stride *= shape[i];
            }

            for (0..shape.len) |i| {
                const dim = shape.len - i - 1;
                stride /= shape[dim];
                pos[dim] = remainingIndex / stride;
                remainingIndex %= stride;
            }

            return pos;
        }

        pub fn print(self: Self) void {
            var shapeStr: [shape.len * 2 - 1]u8 = undefined;
            var j: usize = 0;
            inline for (shape) |s| {
                const b = std.fmt.formatIntBuf(shapeStr[j..shapeStr.len], s, 10, .lower, .{});
                if (j + b < shapeStr.len - 1) shapeStr[j + b] = 'x';
                j += 2;
            }

            std.debug.print("NDTensor<{any},{s}>[", .{ T, shapeStr });
            for (0..size - 1) |i| {
                std.debug.print("{d}, ", .{self.data[i]});
            }
            std.debug.print("{d}]\n", .{self.data[size - 1]});
        }

        pub fn add(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const v1: @Vector(size, T) = self.data;
            const v2: @Vector(size, T) = other.data;
            const v3: [size]T = v1 + v2;
            return Self.init(v3, allocator);
        }

        pub fn sub(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const v1: @Vector(size, T) = self.data;
            const v2: @Vector(size, T) = other.data;
            const v3: [size]T = v1 - v2;
            return Self.init(v3, allocator);
        }

        pub fn mul(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const v1: @Vector(size, T) = self.data;
            const v2: @Vector(size, T) = other.data;
            const v3: [size]T = v1 * v2;
            return Self.init(v3, allocator);
        }

        pub fn div(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const v1: @Vector(size, T) = self.data;
            const v2: @Vector(size, T) = other.data;
            const v3: [size]T = v1 / v2;
            return Self.init(v3, allocator);
        }
    };
}

test "tensor/NDTensor index, add, div" {
    const alloc = std.testing.allocator;
    const Tensor = NDTensor(f32, &[_]u32{ 2, 3 });

    // 1 2 3
    // 4 5 6
    var t1 = Tensor.init([_]f32{ 1, 2, 3, 4, 5, 6 }, alloc);
    defer alloc.destroy(t1);
    t1.print();

    // 1 2 3
    // 4 5 23
    t1.set([_]u32{ 1, 2 }, 23);
    std.debug.print("{d}\n", .{t1.get([_]u32{ 1, 2 })});
    std.debug.print("{d}\n", .{t1.indexToPos(5)});

    const t2 = Tensor.init([_]f32{ 10, 20, 30, 40, 50, 60 }, alloc);
    defer alloc.destroy(t2);

    var t3 = try t1.add(t2, alloc);
    defer alloc.destroy(t3);
    t3.print();

    var t4 = try t3.div(t3, alloc);
    defer alloc.destroy(t4);
    t4.print();
}
