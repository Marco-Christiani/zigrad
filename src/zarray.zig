// TODO: Add SIMD ops (e.g. matmul) to support arbitrary precision types
const std = @import("std");
const c = @cImport(@cInclude("Accelerate/Accelerate.h"));
const root = @import("root");

const NDArrayError = error{InvalidShape};

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

        pub fn init(values: []T, shape: ?[]const usize, allocator: std.mem.Allocator) NDArrayError!*Self {
            if (shape) |s| {
                if (s.len > settings.max_dim) return NDArrayError.InvalidShape;
            }
            const result = allocator.create(Self) catch {
                std.debug.panic("NDArray allocation failed.\n", .{});
            };
            const owned_slice = allocator.alloc(T, values.len) catch {
                std.debug.panic("NDArray allocation failed trying to own slice.\n", .{});
            };
            for (owned_slice, 0..) |*e, i| e.* = values[i];
            result.* = Self{
                .data = owned_slice,
                .shape = if (shape) |s| s else &[_]usize{},
            };
            return result;
        }

        pub fn initFill(val: T, shape: ?[]const usize, allocator: std.mem.Allocator) NDArrayError!*Self {
            if (shape) |s| {
                if (s.len > settings.max_dim) return NDArrayError.InvalidShape;
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

        pub fn reshape(self: *Self, shape: []const usize) NDArrayError!void {
            // TODO: check shape bounds (e.g. the total size)
            if (shape.len > settings.max_dim) return NDArrayError.InvalidShape;
            self.shape = shape;
        }

        pub fn get(self: Self, indices: []const usize) T {
            const index = self.posToIndex(indices);
            return self.data[index];
        }

        pub fn set(self: *Self, indices: []const usize, value: T) NDArrayError!void {
            if (indices.len != self.shape.len) {
                return NDArrayError.InvalidShape;
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

        fn indexToPos(self: Self, index: usize) []const usize {
            var pos: [settings.max_dim]usize = undefined;
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

            return pos[0..self.shape.len];
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
        pub fn matmul(self: *const Self, other: *const Self, allocator: std.mem.Allocator) *Self {
            if (self.shape.len < 2 or other.shape.len < 2) {
                @panic("Input tensors must have at least two dimensions.");
            }

            const M: usize = self.shape[self.shape.len - 2]; // Second last dimension of self
            const K: usize = self.shape[self.shape.len - 1]; // Last dimension of self
            const N: usize = other.shape[other.shape.len - 1]; // Last dimension of other
            const lda = K;
            const ldb = N;
            const ldc = N;

            if (other.shape[other.shape.len - 2] != K) {
                std.debug.panic("Inner dimensions must match for matrix multiplication. (K from A is {}, but K from B is {})\n", .{ K, other.shape[other.shape.len - 2] });
            }

            std.debug.print("(M={}, K={}) x (K={}, N={}) = (M={}, N={}) [M*N={}]\n", .{ M, K, K, N, M, N, M * N });

            const output = allocator.alignedAlloc(T, null, M * N) catch {
                std.debug.panic("Output matrix allocation failed.\n", .{});
            };

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
                        self.data.ptr,
                        @intCast(lda),
                        other.data.ptr,
                        @intCast(ldb),
                        0.0,
                        output.ptr,
                        @intCast(ldc),
                    );
                },
                f64 => {
                    c.dblas_sgemm(
                        c.CblasRowMajor,
                        c.CblasNoTrans,
                        c.CblasNoTrans,
                        @intCast(M),
                        @intCast(N),
                        @intCast(K),
                        1.0,
                        self.data.ptr,
                        @intCast(lda),
                        other.data.ptr,
                        @intCast(ldb),
                        0.0,
                        output.ptr,
                        @intCast(ldc),
                    );
                },
                else => std.debug.panic("Unsupported type {}", .{@typeName(T)}),
            }

            // Constructing the result tensor
            const result = allocator.create(Self) catch {
                std.debug.panic("Tensor allocation failed.\n", .{});
            };
            const shape = allocator.alloc(usize, 2) catch {
                @panic("Failed to allocate shape array.\n");
            };
            shape[0] = M;
            shape[1] = N;
            result.* = .{
                .shape = shape,
                .data = output,
            };
            return result;
        }
    };
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const shape1 = &[_]usize{ 2, 3 };
    const shape2 = &[_]usize{ 3, 2 };
    const Array = NDArray(f32);

    // multiply a 2x3 and a 3x2 to get a 2x2
    var A = try Array.init(@constCast(&[_]f32{ 1, 2, 3, 4, 5, 6 }), shape1, alloc);
    const B = try Array.init(@constCast(&[_]f32{ 1, 0, 0, 1, 0, 0 }), shape2, alloc);
    var C = A.matmul(B, alloc);
    C.print();
}
