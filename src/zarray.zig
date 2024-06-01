// TODO: Add SIMD ops (e.g. matmul) to support arbitrary precision types
// TODO: generate tests for all supported types
// TODO: support batch ops
const std = @import("std");
const c = @cImport(@cInclude("Accelerate/Accelerate.h"));
const root = @import("root");

const ZarrayError = error{ InvalidShape, Unbroadcastable, InvalidIndex, ShapeOutOfBounds };

/// lib-wide options that can be overridden by the root file.
pub const settings: Settings = if (@hasDecl(root, "zigrad_settings")) root.zigrad_settings else .{};

const Settings = struct {
    grad_enabled: bool = true,
    max_dim: usize = 4,
};

pub fn NDArray(comptime T: type) type {
    return struct {
        const Self = @This();
        shape: []const usize,
        data: []T,

        pub fn init(values: []const T, shape: ?[]const usize, allocator: std.mem.Allocator) ZarrayError!*Self {
            const owned_shape: []const usize = blk: {
                if (shape) |s| {
                    if (s.len > settings.max_dim) return ZarrayError.InvalidShape;
                    const owned_shape = allocator.alloc(usize, s.len) catch {
                        std.debug.panic("NDArray allocation failed trying to own shape slice.\n", .{});
                    };
                    for (owned_shape, 0..) |*e, i| e.* = s[i];
                    break :blk owned_shape;
                } else {
                    const owned_shape = allocator.alloc(usize, 1) catch {
                        std.debug.panic("NDArray allocation failed trying to own shape slice.\n", .{});
                    };
                    owned_shape[0] = values.len;
                    break :blk owned_shape;
                }
            };
            const result = allocator.create(Self) catch {
                std.debug.panic("NDArray allocation failed.\n", .{});
            };
            const owned_slice = allocator.alloc(T, values.len) catch {
                std.debug.panic("NDArray allocation failed trying to own values slice.\n", .{});
            };
            for (owned_slice, 0..) |*e, i| e.* = values[i];

            result.* = Self{
                .data = owned_slice,
                .shape = owned_shape,
            };
            return result;
        }

        pub fn initFill(val: T, shape: ?[]const usize, allocator: std.mem.Allocator) ZarrayError!*Self {
            if (shape) |s| {
                if (s.len > settings.max_dim) return ZarrayError.InvalidShape;
            }
            var data: []T = undefined;
            for (&data) |*elem| {
                elem.* = val;
            }
            return Self.init(data, shape, allocator);
        }

        pub fn fill(self: *Self, val: T) void {
            for (0..self.data.len) |i| {
                self.data[i] = val;
            }
        }

        pub fn reshape(self: *Self, shape: []const usize) ZarrayError!void {
            if (shape.len > settings.max_dim or shape.len == 0) return ZarrayError.InvalidShape;
            const requested_size = blk: {
                var s: usize = 1;
                for (shape) |e| s *= e;
                break :blk s;
            };
            if (requested_size > self.data.len) return ZarrayError.ShapeOutOfBounds;
            self.shape = shape;
        }

        pub fn get(self: Self, indices: []const usize) T {
            const index = self.posToIndex(indices);
            return self.data[index];
        }

        pub fn slice(self: Self, indices: []const usize) []T {
            _ = self;
            _ = indices;
            @compileError("Not yet implemented.\n");
        }

        pub fn set(self: *Self, indices: []const usize, value: T) ZarrayError!void {
            if (indices.len != self.shape.len) {
                return ZarrayError.InvalidShape;
            }
            std.debug.assert(indices.len == self.shape.len);
            const index = self.posToIndex(indices);
            self.data[index] = value;
        }

        fn posToIndex(self: Self, indices: []const usize) usize {
            std.debug.assert(indices.len == self.shape.len);
            var index: usize = 0;
            var stride: usize = 1;
            for (0..self.shape.len) |i| {
                const dim = self.shape.len - i - 1;
                const dimSize = self.shape[dim];
                const idx = indices[dim];
                std.debug.assert(idx < dimSize);

                index += idx * stride;
                stride *= dimSize;
            }
            return index;
        }

        /// See flexSelectIndex
        fn flexPosToIndex(self: Self, indices: []const usize) ZarrayError.InvalidIndex!usize {
            return flexSelectIndex(self.shape, indices);
        }

        fn indexToPos(self: Self, index: usize, allocator: std.mem.Allocator) []const usize {
            const pos = allocator.alloc(T, self.len) catch std.debug.panic("Failed to allocate shape slice.\n", .{});
            var remainingIndex = index;
            var stride: usize = 1;
            for (0..self.shape.len) |i| {
                stride *= self.shape[i];
            }

            for (0..self.shape.len) |i| {
                const dim = self.shape.len - i - 1;
                stride /= self.shape[dim];
                pos[dim] = remainingIndex / stride;
                remainingIndex %= stride;
            }

            return pos;
        }

        pub fn print(self: *const Self) void {
            const alloc = std.heap.page_allocator;
            var shapeStr: []u8 = alloc.alloc(u8, self.shape.len * 2 - 1) catch unreachable;
            defer alloc.free(shapeStr);
            var j: usize = 0;
            for (self.shape) |s| {
                const b = std.fmt.formatIntBuf(shapeStr[j..shapeStr.len], s, 10, .lower, .{});
                if (j + b < shapeStr.len - 1) shapeStr[j + b] = 'x';
                j += 2;
            }

            std.debug.print("NDArray<{any},{s}>", .{ T, shapeStr });
            std.debug.print("{d}", .{self.data});
        }

        /// Vec-vec: Dot product
        /// Mat-vec: 2D x 1D
        /// (...)-Mat-Mat: ND x KD (N,K>2) and broadcastable
        /// Simple dim rules: (M, K) x (K, N) = (M, N)
        /// Computes self*other without tranposing
        pub fn matmul(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            if (self.shape.len < 2 or other.shape.len < 2) {
                std.debug.panic("Input tensors must have at least two dimensions.\n", .{});
            }

            const M: usize = self.shape[self.shape.len - 2];
            const K: usize = self.shape[self.shape.len - 1];
            const N: usize = other.shape[other.shape.len - 1];

            if (other.shape[other.shape.len - 2] != K) {
                std.debug.panic("Inner dimensions must match for matrix multiplication. (K from A is {}, but K from B is {})\n", .{ K, other.shape[other.shape.len - 2] });
            }

            const output: []T = allocator.alignedAlloc(T, null, M * N) catch {
                std.debug.panic("Output array allocation failed.\n", .{});
            };
            errdefer allocator.free(output);
            blas_matmul(T, self.data, other.data, output, M, N, K);
            return Self.init(output, &[_]usize{ M, N }, allocator);
        }

        pub fn dot(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            if (self.shape.len > 1 or other.shape.len > 1) std.debug.panic("Dot product only valid for 1d vectors even if there are dummy outer dimensions.\n", .{});
            if (self.data.len != other.data.len) std.debug.panic("Incompatible lengths for dot product: {d} and {d}\n", .{ self.data.len, other.data.len });

            const output: T = blas_dot(T, self.data, other.data);
            return Self.init(&[_]T{output}, null, allocator);
        }

        pub fn matvec(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            _ = self;
            _ = other;
            _ = allocator;
            @compileError("Not yet implemented.\n");
        }
    };
}

/// Computes dot product assuming a stride of 1. (N,) x (N,) = (1,)
pub fn blas_dot(T: type, A: []T, B: []T) T {
    switch (T) {
        f32 => return c.cblas_sdot(@intCast(A.len), A.ptr, 1, B.ptr, 1),
        f64 => return c.cblas_ddot(@intCast(A.len), A.ptr, 1, B.ptr, 1),
        else => std.debug.panic("Unsupported type {}\n", .{@typeName(T)}),
    }
}

/// (M, K) x (K, N) = (M, N)
pub fn blas_matmul(T: type, A: []T, B: []T, C: []T, M: usize, N: usize, K: usize) void {
    const lda = K;
    const ldb = N;
    const ldc = N;
    switch (T) {
        f32 => {
            c.cblas_sgemm(
                c.CblasRowMajor,
                c.CblasNoTrans,
                c.CblasNoTrans,
                @intCast(M),
                @intCast(N),
                @intCast(K),
                1.0,
                A.ptr,
                @intCast(lda),
                B.ptr,
                @intCast(ldb),
                0.0,
                C.ptr,
                @intCast(ldc),
            );
        },
        f64 => {
            c.cblas_dgemm(
                c.CblasRowMajor,
                c.CblasNoTrans,
                c.CblasNoTrans,
                @intCast(M),
                @intCast(N),
                @intCast(K),
                1.0,
                A.ptr,
                @intCast(lda),
                B.ptr,
                @intCast(ldb),
                0.0,
                C.ptr,
                @intCast(ldc),
            );
        },
        else => std.debug.panic("Unsupported type {}\n", .{@typeName(T)}),
    }
}

fn calculateBroadcastedShape(shape1: []const usize, shape2: []const usize, allocator: std.mem.Allocator) error{ OutOfMemory, Unbroadcastable }![]usize {
    const dims = @max(shape1.len, shape2.len);
    const result_shape = try allocator.alloc(usize, dims);
    var i: usize = 0;
    while (i < dims) {
        const dimA = if (i < shape1.len) shape1[shape1.len - 1 - i] else 1;
        const dimB = if (i < shape2.len) shape2[shape2.len - 1 - i] else 1;
        if (dimA != dimB and dimA != 1 and dimB != 1) {
            allocator.free(result_shape);
            return ZarrayError.Unbroadcastable;
        }
        result_shape[dims - 1 - i] = @max(dimA, dimB);
        i += 1;
    }
    return result_shape;
}

/// Flexibly select index allowing for indices.len > shape.len
/// Effectively mocks batch dimensions (e.g. shape=(2,2), indices=(0, 0) == indices=(1, 0, 0) == indices= (..., 0, 0))
fn flexSelectIndex(shape: []const usize, indices: []const usize) !usize {
    if (indices.len < shape.len) {
        return ZarrayError.InvalidIndex; // should be slicing, not selecting a single index
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

    try std.testing.expectError(ZarrayError.Unbroadcastable, calculateBroadcastedShape(&shape1, &[_]usize{ 4, 3 }, alloc));
    try std.testing.expectError(ZarrayError.Unbroadcastable, calculateBroadcastedShape(&shape1, &[_]usize{ 3, 2 }, alloc));
    try std.testing.expectError(ZarrayError.Unbroadcastable, calculateBroadcastedShape(&shape1, &[_]usize{ 3, 3 }, alloc));
}

test "flexPosToIndex" {
    // arr = [[0, 1, 2]
    //        [3, 4, 5]]
    // arr[1, 1] == 4
    try std.testing.expectEqual(4, flexSelectIndex(&[_]usize{ 2, 3 }, &[_]usize{ 1, 1 }));
    // arr[1, 1, 1] == 4
    try std.testing.expectEqual(4, flexSelectIndex(&[_]usize{ 2, 3 }, &[_]usize{ 1, 1, 1 }));

    // arr = [[[0, 1, 2]
    //        [3, 4, 5]]]
    // arr[1, 1, 1] == 4
    try std.testing.expectEqual(4, flexSelectIndex(&[_]usize{ 1, 2, 3 }, &[_]usize{ 0, 1, 1 }));
}

test "NDArray.reshape" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f64;
    const Array = NDArray(T);

    var A = try Array.init(@constCast(&[_]T{ 1, 2, 3, 4, 5, 6 }), null, alloc);
    try A.reshape(&[_]usize{ 2, 3 });
    try std.testing.expectError(ZarrayError.ShapeOutOfBounds, A.reshape(&[_]usize{9}));
    const v1 = A.get(&[_]usize{ 0, 0 });
    try A.reshape(&[_]usize{ 2, 3 });
    const v2 = A.get(&[_]usize{ 0, 0 });
    try std.testing.expectEqual(v1, v2);
}

test "NDArray.dot" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f64;
    const Array = NDArray(T);

    const a = try Array.init(@constCast(&[_]T{ 1, 2, 3 }), null, alloc);
    const b = try Array.init(@constCast(&[_]T{ 1, 1, 1 }), null, alloc);
    const result = try a.dot(b, alloc);
    const expected = Array.init(&[_]T{6}, null, alloc);
    try std.testing.expectEqualDeep(expected, result);
}

test "NDArray.matmul" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f64;
    const Array = NDArray(T);

    // multiply a 2x2 and a 2x2 to get a 2x2
    // [[1, 2]  * [[1, 1]  = [[3, 3]
    //  [0, 1]]    [1, 1]]    [1, 1]]
    const A1 = try Array.init(@constCast(&[_]T{ 1, 2, 0, 1 }), &[_]usize{ 2, 2 }, alloc);
    const B1 = try Array.init(@constCast(&[_]T{ 1, 1, 1, 1 }), &[_]usize{ 2, 2 }, alloc);
    const C1 = try A1.matmul(B1, alloc);
    const expected1 = Array.init(&[_]T{ 3, 3, 1, 1 }, &[_]usize{ 2, 2 }, alloc);
    try std.testing.expectEqualDeep(expected1, C1);

    const shape1 = &[_]usize{ 2, 3 };
    const shape2 = &[_]usize{ 3, 2 };

    // multiply a 2x3 and a 3x2 to get a 2x2
    // [[1, 2, 3]  * [[1, 0]  = [[1, 2]
    //  [4, 5, 6]]    [0, 1]     [4, 5]]
    //                [0, 0]]
    var A2 = try Array.init(@constCast(&[_]T{ 1, 2, 3, 4, 5, 6 }), shape1, alloc);
    const B2 = try Array.init(@constCast(&[_]T{ 1, 0, 0, 1, 0, 0 }), shape2, alloc);
    const C2 = try A2.matmul(B2, alloc);
    const expected2 = Array.init(&[_]T{ 1, 2, 4, 5 }, &[_]usize{ 2, 2 }, alloc);
    try std.testing.expectEqualDeep(expected2, C2);
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const shape1 = &[_]usize{ 2, 3 };
    const shape2 = &[_]usize{ 3, 2 };
    const T = f64;
    const Array = NDArray(T);

    // multiply a 2x3 and a 3x2 to get a 2x2
    var A = try Array.init(@constCast(&[_]T{ 1, 2, 3, 4, 5, 6 }), shape1, alloc);
    const B = try Array.init(@constCast(&[_]T{ 1, 0, 0, 1, 0, 0 }), shape2, alloc);
    var C = try A.matmul(B, alloc);
    C.print();
}
