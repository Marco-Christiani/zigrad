// TODO:Scalar ops
// TODO: support ND ops. finish broadcast integration
// TODO: element-wise ops to blas
// TODO: Shape memory management, put it + tests somewhere
// TODO: exp
// TODO: document bcast rules and shape rules for inplace ewise ops
const std = @import("std");
const c = @cImport(@cInclude("Accelerate/Accelerate.h"));

pub const ZarrayError = error{ InvalidShape, Unbroadcastable, InvalidIndex, ShapeOutOfBounds, IncompatibleShapes };

pub fn NDArray(comptime T: type) type {
    return struct {
        const Self = @This();
        shape: *Shape,
        data: []T,

        pub fn init(values: []const T, shape: ?[]const usize, allocator: std.mem.Allocator) !*Self {
            const result = try allocator.create(Self);

            result.* = Self{
                .data = try allocator.dupe(T, values),
                .shape = try Shape.init(shape orelse &[_]usize{values.len}, allocator),
            };
            if (result.shape.size() != result.data.len) {
                std.log.err("Invalid shape: result.shape.size()={d} result.data.len={d}", .{ result.shape.size(), result.data.len });
                return ZarrayError.InvalidShape;
            }
            return result;
        }

        pub fn initNoAlloc(values: []T, shape: []const usize, allocator: std.mem.Allocator) !*Self {
            const result = allocator.create(Self) catch {
                std.debug.panic("NDArray allocation failed.", .{});
            };
            const result_shape = try allocator.create(Shape);
            result_shape.* = Shape{
                .shape = @constCast(shape),
                .alloc = allocator,
            };

            result.* = Self{
                .data = values,
                .shape = result_shape,
            };
            if (result.shape.size() > result.data.len) return ZarrayError.InvalidShape;
            return result;
        }

        pub fn deinit(self: *const Self, allocator: std.mem.Allocator) void {
            allocator.free(self.data);
            self.shape.deinit();
            allocator.destroy(self);
        }

        pub fn zerosLike(self: *const Self, allocator: std.mem.Allocator) !*Self {
            const result = try allocator.create(Self);
            result.* = .{
                .data = try allocator.alloc(T, self.shape.size()),
                .shape = try Shape.init(self.shape.shape, allocator),
            };
            @memset(result.data, 0);
            return result;
        }

        pub fn empty(shape: []const usize, allocator: std.mem.Allocator) !*Self {
            const result = try allocator.create(Self);
            const empty_shape = try Shape.init(shape, allocator);
            result.* = .{
                .data = try allocator.alloc(T, empty_shape.size()),
                .shape = empty_shape,
            };
            return result;
        }

        pub fn fill(self: *Self, val: T) void {
            @memset(self.data, val);
        }

        pub fn reshape(self: *Self, shape: []const usize) !void {
            try self.shape.reshape(shape);
        }

        pub fn get(self: Self, indices: []const usize) T {
            return self.data[self.posToOffset(indices)];
        }

        pub fn set(self: *Self, indices: []const usize, value: T) ZarrayError!void {
            if (indices.len != self.shape.len()) {
                return ZarrayError.InvalidShape;
            }
            std.debug.assert(indices.len == self.shape.len());
            const index = self.posToOffset(indices);
            self.data[index] = value;
        }

        pub fn posToOffset(self: Self, indices: []const usize) usize {
            std.debug.assert(indices.len == self.shape.len());
            var index: usize = 0;
            var stride: usize = 1;
            for (0..self.shape.len()) |i| {
                const dim = self.shape.len() - i - 1;
                const dimSize = self.shape.get(dim) catch unreachable;
                const idx = indices[dim];
                std.debug.assert(idx < dimSize);

                index += idx * stride;
                stride *= dimSize;
            }
            return index;
        }

        /// See flexSelectOffset
        pub fn flexPosToOffset(self: Self, indices: []const usize) ZarrayError.InvalidIndex!usize {
            return flexSelectOffset(self.shape, indices);
        }

        pub fn offsetToPos(self: Self, offset: usize, allocator: std.mem.Allocator) []const usize {
            const pos = allocator.alloc(usize, self.data.len) catch std.debug.panic("Failed to allocate shape slice.\n", .{});
            var remainingIndex = offset;
            var stride: usize = 1;
            for (0..self.shape.len()) |i| {
                stride *= self.shape.get(i);
            }

            for (0..self.shape.len()) |i| {
                const dim = self.shape.len() - i - 1;
                stride /= self.shape.get(dim);
                pos[dim] = remainingIndex / stride;
                remainingIndex %= stride;
            }

            return pos;
        }

        pub fn size(self: *const Self) !void {
            return self.data.len;
        }

        pub fn print(self: *const Self) void {
            const alloc = std.heap.page_allocator;
            var shapeStr: []u8 = alloc.alloc(u8, self.shape.len() * @sizeOf(usize)) catch unreachable;
            defer alloc.free(shapeStr);
            var j: usize = 0;
            var bytes_written: usize = 0;
            for (self.shape.shape, 0..) |s, i| {
                const b = std.fmt.formatIntBuf(shapeStr[j..shapeStr.len], s, 10, .lower, .{});
                bytes_written += b;
                if (i < self.shape.len() - 1 and j + b < shapeStr.len - 1) {
                    shapeStr[j + b] = 'x';
                    bytes_written += 1;
                } else {
                    break;
                }
                j += 2;
            }

            std.debug.print("NDArray<{any},{s}>", .{ T, shapeStr[0..bytes_written] });
            std.debug.print("{d}", .{self.data});
        }

        pub fn slice(self: Self, ranges: []const Range) !Self {
            if (ranges.len != self.shape.len()) {
                return ZarrayError.InvalidShape;
            }

            var new_shape = try self.shape.alloc.alloc(usize, self.shape.len());
            var start_index: usize = 0;
            var total_elements: usize = 1;

            for (ranges, 0..) |range, i| {
                if (range.end > self.shape.shape[i]) {
                    return ZarrayError.ShapeOutOfBounds;
                }
                new_shape[i] = range.end - range.start;
                total_elements *= new_shape[i];
                start_index += range.start * self.getStride(i);
            }

            const shape = try self.shape.alloc.create(Shape);
            shape.* = Shape{ .shape = new_shape, .alloc = self.shape.alloc };

            return Self{
                .shape = shape,
                .data = self.data[start_index .. start_index + total_elements],
            };
        }

        pub fn setSlice(self: *Self, ranges: []const Range, values: Self) !void {
            const slice_ = try self.slice(ranges);
            defer slice_.shape.deinit();
            if (!slice_.shape.eq(values.shape.*, .{ .strict = true })) {
                return ZarrayError.IncompatibleShapes;
            }

            var i: usize = 0;
            while (i < slice_.data.len) : (i += 1) {
                slice_.data[i] = values.data[i];
            }
        }

        fn getStride(self: Self, dim: usize) usize {
            var s: usize = 1;
            var i: usize = dim + 1;
            while (i < self.shape.len()) : (i += 1) {
                s *= self.shape.shape[i];
            }
            return s;
        }

        /// Element-wise addition
        pub fn add(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const bshape = try self.shape.broadcast(other.shape);
            const values = try allocator.alloc(T, bshape.size());
            for (0..values.len) |i| {
                values[i] = self.data[i % self.data.len] + other.data[i % other.data.len];
            }
            return Self.initNoAlloc(values, bshape.shape, allocator);
        }

        /// In-place element-wise addition
        pub fn _add(self: *Self, other: *const Self) !*Self {
            if (!Shape.eq(self.shape.*, other.shape.*, .{}) or other.shape.size() > self.shape.size()) {
                std.log.err("self.shape={d} other.shape={d}", .{ self.shape.shape, other.shape.shape });
                return ZarrayError.IncompatibleShapes;
            }
            for (0..self.data.len) |i| self.data[i] += other.data[i % other.data.len];
            return self;
        }

        /// Element-wise subtraction
        pub fn sub(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const bshape = try self.shape.broadcast(other.shape);
            const values = try allocator.alloc(T, bshape.size());
            for (0..values.len) |i| values[i] = self.data[i % self.data.len] - other.data[i % other.data.len];
            return Self.initNoAlloc(values, bshape.shape, allocator);
        }

        /// In-place element-wise subtraction
        pub fn _sub(self: *Self, other: *const Self) !*Self {
            if (!Shape.eq(self.shape.*, other.shape.*, .{}) or other.shape.size() > self.shape.size()) {
                std.log.err("self.shape={d} other.shape={d}", .{ self.shape.shape, other.shape.shape });
                return ZarrayError.IncompatibleShapes;
            }
            for (0..self.data.len) |i| self.data[i] -= other.data[i % other.data.len];
            return self;
        }

        /// Element-wise multiplication
        pub fn mul(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const bshape = try self.shape.broadcast(other.shape);
            const values = try allocator.alloc(T, bshape.size());
            for (0..values.len) |i| values[i] = self.data[i % self.data.len] * other.data[i % other.data.len];
            return Self.initNoAlloc(values, bshape.shape, allocator);
        }

        /// In-place element-wise multiplication
        pub fn _mul(self: *Self, other: *const Self) !*Self {
            if (!Shape.eq(self.shape.*, other.shape.*, .{}) or other.shape.size() > self.shape.size()) {
                std.log.err("self.shape={d} other.shape={d}", .{ self.shape.shape, other.shape.shape });
                return ZarrayError.IncompatibleShapes;
            }
            for (0..self.data.len) |i| self.data[i] *= other.data[i % other.data.len];
            return self;
        }

        /// Element-wise division
        pub fn div(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const bshape = try self.shape.broadcast(other.shape);
            const values = try allocator.alloc(T, bshape.size());
            for (0..values.len) |i| values[i] = self.data[i % self.data.len] / other.data[i % other.data.len];
            return Self.initNoAlloc(values, bshape.shape, allocator);
        }

        /// In-place element-wise division
        pub fn _div(self: *Self, other: *const Self) !*Self {
            if (!Shape.eq(self.shape.*, other.shape.*, .{}) or other.shape.size() > self.shape.size()) {
                std.log.err("self.shape={d} other.shape={d}", .{ self.shape.shape, other.shape.shape });
                return ZarrayError.IncompatibleShapes;
            }
            for (0..self.data.len) |i| self.data[i] /= other.data[i % other.data.len];
            return self;
        }

        /// Element-wise sum
        pub fn sum(self: *Self, allocator: std.mem.Allocator) !*Self {
            var s: T = 0;
            for (self.data) |e| s += e;
            return Self.init(&[_]T{s}, null, allocator);
        }

        /// (...)-Mat-Mat: ND x KD (N,K>2) and broadcastable
        /// Simple dim rules: (M, K) x (K, N) = (M, N)
        pub fn matmul(self: *const Self, other: *const Self, trans_a: bool, trans_b: bool, allocator: std.mem.Allocator) !*Self {
            if (self.shape.len() < 2 or other.shape.len() < 2) {
                std.debug.panic("Input tensors must have at least two dimensions.", .{});
            }
            var a_rows: usize = try self.shape.rget(-2);
            var a_cols: usize = try self.shape.rget(-1);
            var b_rows: usize = try other.shape.rget(-2);
            var b_cols: usize = try other.shape.rget(-1);

            const orig_a_cols = a_cols;
            const orig_b_cols = b_cols;

            if (trans_a) {
                std.mem.swap(usize, &a_rows, &a_cols);
            }
            if (trans_b) {
                std.mem.swap(usize, &b_rows, &b_cols);
            }

            const M: usize = a_rows;
            const K: usize = a_cols;
            const N: usize = b_cols;

            const output: []T = try allocator.alignedAlloc(T, null, M * N);
            errdefer allocator.free(output);

            blas_matmul(T, self.data, other.data, output, M, N, K, trans_a, trans_b, orig_a_cols, orig_b_cols, N);
            const result = try allocator.create(Self);
            result.* = Self{
                .data = output,
                .shape = try Shape.init(&[_]usize{ M, N }, allocator),
            };
            return result;
        }

        pub fn dot(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            if (self.shape.len() > 1 or other.shape.len() > 1) std.debug.panic("Dot product only valid for 1d vectors even if there are dummy outer dimensions.\n", .{});
            if (self.data.len != other.data.len) std.debug.panic("Incompatible lengths for dot product: {d} and {d}\n", .{ self.data.len, other.data.len });

            const output: T = blas_dot(T, self.data, other.data);
            return Self.init(&[_]T{output}, &[_]usize{1}, allocator);
        }

        pub fn outer(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            if (self.shape.len() != 1 or other.shape.len() != 1) std.debug.panic("Outer product only valid for 1d vectors even if there are dummy outer dimensions.\n", .{});
            const M: usize = try self.shape.get(0);
            const N: usize = try other.shape.get(0);
            const output: []T = try allocator.alignedAlloc(T, null, M * N);
            @memset(output, 0);
            errdefer allocator.free(output);
            blas_outer(T, self.data, other.data, output, 1);
            const result = try allocator.create(Self);
            result.* = Self{
                .data = output,
                .shape = try Shape.init(&[_]usize{ M, N }, allocator),
            };
            return result;
        }

        pub fn matvec(self: *const Self, other: *const Self, trans_a: bool, allocator: std.mem.Allocator) !*Self {
            // TODO: shape checks for matvec
            const output_size = if (trans_a) try self.shape.get(1) else try self.shape.get(0);
            const output: []T = try allocator.alignedAlloc(T, null, output_size);
            errdefer allocator.free(output);
            blas_matvec(T, self.data, other.data, output, try self.shape.get(0), try self.shape.get(1), 1, 0, trans_a);

            const result = try allocator.create(Self);
            result.* = Self{
                .data = output,
                .shape = try Shape.init(other.shape.shape, allocator),
            };
            return result;
        }

        pub const SumOpts = struct {
            dim: usize,
            keep_dims: bool = false,
        };

        pub fn sum_along(self: *Self, allocator: std.mem.Allocator, opts: SumOpts) !*Self {
            const input_shape = self.shape.shape;
            const input_dims = input_shape.len;
            if (opts.dim >= input_dims) return ZarrayError.ShapeOutOfBounds;

            var output_shape = try allocator.alloc(usize, if (opts.keep_dims) input_dims else input_dims - 1);
            defer allocator.free(output_shape);
            var idx: usize = 0;
            for (0..input_dims) |i| {
                if (i != opts.dim) {
                    output_shape[idx] = input_shape[i];
                    idx += 1;
                } else if (opts.keep_dims) {
                    output_shape[idx] = 1;
                }
            }
            const output = try Self.empty(output_shape, allocator);
            output.fill(0);

            const sum_dim_size = input_shape[opts.dim];

            var slice_size: usize = 1;
            for (opts.dim + 1..input_dims) |i| {
                slice_size *= input_shape[i];
            }

            var num_slices: usize = 1;
            for (0..opts.dim) |i| {
                num_slices *= input_shape[i];
            }

            for (0..output.data.len) |i| {
                var total: T = 0;
                const base_idx = (i / slice_size) * (slice_size * sum_dim_size) + (i % slice_size);
                for (0..sum_dim_size) |j| {
                    const curr_idx = base_idx + j * slice_size;
                    total += self.data[curr_idx];
                }
                output.data[i] = total;
            }

            return output;
        }

        fn l2_norm(self: *const Self) T {
            return blas_nrm2(T, self.data, 1);
        }

        pub fn clip_norm(self: *Self, max_norm: T, delta: T) void {
            const norm = self.l2_norm();
            if (norm > max_norm) {
                const scale = max_norm / (norm + delta);
                for (self.data) |*value| {
                    value.* *= scale;
                }
            }
        }

        /// Unbroadcast to target shape
        pub fn unbroadcast(self: *Self, target_shape: *Shape, allocator: std.mem.Allocator) !*Self {
            var result: *Self = self;
            while (result.shape.len() > target_shape.len()) {
                const temp = result;
                result = try temp.sum_along(allocator, .{ .dim = 0 });
                if (temp != self) temp.deinit(allocator);
            }

            if (result.shape.len() == target_shape.len()) {
                for (target_shape.shape, 0..) |s, dimi| {
                    if (s == 1 and result.shape.shape[dimi] != 1) {
                        const temp = result;
                        result = try temp.sum_along(allocator, .{ .dim = dimi, .keep_dims = true });
                        if (temp != self) temp.deinit(allocator);
                    }
                }
            }

            return result;
        }
        // pub fn unbroadcast(self: *Self, target_shape: *Shape, allocator: std.mem.Allocator) !*Self {
        //     if (self.shape.eq(target_shape.*, .{ .strict = true })) return self;
        //     var result: *Self = self;
        //     while (result.shape.len() > target_shape.len()) {
        //         const temp = result;
        //         result = try temp.sum_along(allocator, .{ .dim = 0 });
        //         // temp.deinit(allocator);
        //     }
        //     if (result.shape.len() == target_shape.len()) {
        //         for (target_shape.shape, 0..) |s, dimi| {
        //             if (s == 1) {
        //                 const temp = result;
        //                 result = try temp.sum_along(allocator, .{ .dim = dimi, .keep_dims = true });
        //                 // temp.deinit(allocator);
        //             }
        //         }
        //     }
        //     return result;
        // }
    };
}

/// Computes dot product assuming a stride of 1 and row-major. (N,) x (N,) = (1,)
pub fn blas_dot(T: type, A: []T, B: []T) T {
    switch (T) {
        f32 => return c.cblas_sdot(@intCast(A.len), A.ptr, 1, B.ptr, 1),
        f64 => return c.cblas_ddot(@intCast(A.len), A.ptr, 1, B.ptr, 1),
        else => std.debug.panic("Unsupported type {}\n", .{@typeName(T)}),
    }
}

/// Computes mat-vec assuming a stride of 1 for the vec and row-major.
/// a * (M, N) x (N,) + b * (N,) = (M,)
/// Y = aAX + bY
pub fn blas_matvec(T: type, A: []T, X: []T, Y: []T, M: usize, N: usize, alpha: T, beta: T, trans_a: bool) void {
    const lda = N;
    const ta = if (trans_a) c.CblasTrans else c.CblasNoTrans;
    switch (T) {
        f32 => c.cblas_sgemv(c.CblasRowMajor, @intCast(ta), @intCast(M), @intCast(N), alpha, A.ptr, @intCast(lda), X.ptr, 1, beta, Y.ptr, 1),
        f64 => c.cblas_dgemv(c.CblasRowMajor, @intCast(ta), @intCast(M), @intCast(N), alpha, A.ptr, @intCast(lda), X.ptr, 1, beta, Y.ptr, 1),
        else => std.debug.panic("Unsupported type {}\n", .{@typeName(T)}),
    }
}

///  Assumes row-major.
///  (M, K) x (K, N) = (M, N)
pub fn blas_matmul(T: type, A: []T, B: []T, C: []T, M: usize, N: usize, K: usize, trans_a: bool, trans_b: bool, lda: usize, ldb: usize, ldc: usize) void {
    const ta = if (trans_a) c.CblasTrans else c.CblasNoTrans;
    const tb = if (trans_b) c.CblasTrans else c.CblasNoTrans;
    switch (T) {
        f32 => c.cblas_sgemm(c.CblasRowMajor, @intCast(ta), @intCast(tb), @intCast(M), @intCast(N), @intCast(K), 1.0, A.ptr, @intCast(lda), B.ptr, @intCast(ldb), 0.0, C.ptr, @intCast(ldc)),
        f64 => c.cblas_dgemm(c.CblasRowMajor, @intCast(ta), @intCast(tb), @intCast(M), @intCast(N), @intCast(K), 1.0, A.ptr, @intCast(lda), B.ptr, @intCast(ldb), 0.0, C.ptr, @intCast(ldc)),
        else => std.debug.panic("Unsupported type {}\n", .{@typeName(T)}),
    }
}

/// Outer product: A = alpha(xy') + A
/// A: (M, N)
pub fn blas_outer(T: type, x: []T, y: []T, A: []T, alpha: T) void {
    switch (T) {
        f32 => c.cblas_sger(c.CblasRowMajor, @intCast(x.len), @intCast(y.len), alpha, x.ptr, 1, y.ptr, 1, A.ptr, @intCast(y.len)),
        f64 => c.cblas_dger(c.CblasRowMajor, @intCast(x.len), @intCast(y.len), alpha, x.ptr, 1, y.ptr, 1, A.ptr, @intCast(y.len)),
        else => std.debug.panic("Unsupported type {}\n", .{@typeName(T)}),
    }
}
/// Outer product: A = alpha(xy') + A
/// A: (M, N)
pub fn blas_nrm2(T: type, x: []T, stride: usize) T {
    return switch (T) {
        f32 => c.cblas_snrm2(@intCast(x.len), x.ptr, @intCast(stride)),
        f64 => c.cblas_dnrm2(@intCast(x.len), x.ptr, @intCast(stride)),
        else => std.debug.panic("Unsupported type {}\n", .{@typeName(T)}),
    };
}

pub const Shape = struct {
    const Self = @This();
    shape: []usize,
    alloc: std.mem.Allocator,

    pub fn init(shape: []const usize, allocator: std.mem.Allocator) !*Self {
        if (shape.len == 0) return ZarrayError.InvalidShape;
        const self = try allocator.create(Self);
        self.alloc = allocator;
        self.shape = try allocator.dupe(usize, shape);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.alloc.free(self.shape);
        self.alloc.destroy(self);
    }

    pub fn broadcast(self: *Self, other: *const Self) !Self {
        const dims = @max(self.len(), other.len());
        const result_shape = try self.alloc.alloc(usize, dims);
        var i: usize = 0;
        while (i < dims) {
            const dimA = if (i < self.shape.len) self.shape[self.shape.len - 1 - i] else 1;
            const dimB = if (i < other.shape.len) other.shape[other.shape.len - 1 - i] else 1;
            if (dimA != dimB and dimA != 1 and dimB != 1) return ZarrayError.Unbroadcastable;
            result_shape[dims - 1 - i] = @max(dimA, dimB);
            i += 1;
        }
        // self.alloc.free(self.shape);
        // self.shape = result_shape;
        // return self;
        return Self{ .shape = result_shape, .alloc = self.alloc };
    }

    // TODO: usage should be minimized (or size stored but rarely needed this was)
    pub fn size(self: Self) usize {
        if (self.shape.len == 0) return 0;
        var total: usize = 1;
        for (self.shape) |b| total *= b;
        return total;
    }

    pub fn len(self: Self) usize {
        return self.shape.len;
    }

    /// Functional number of dims if the shape was squeezed
    pub fn realdims(self: Self) usize {
        var ndim: usize = 1;
        for (self.shape) |e| {
            if (e != 1) ndim += 1;
        }
        return ndim;
    }

    pub fn squeeze(self: *Self) !*Self {
        var newshape = try self.alloc.alloc(usize, self.realdims());
        var j: usize = 0;
        for (0..self.shape.len) |i| {
            if (self.shape[i] != 1) {
                newshape[j] = self.shape[i];
                j += 1;
            }
        }
        self.alloc.free(self.shape);
        self.shape = newshape;
        return self;
    }

    pub fn reshape(self: *Self, shape: []const usize) !void {
        // TODO: reshape()
        if (shape.len == 0) return ZarrayError.InvalidShape;
        const requested_size = blk: {
            var s: usize = 1;
            for (shape) |e| s *= e;
            break :blk s;
        };
        if (requested_size > self.size()) {
            std.log.warn("ShapeOutOfBounds requested_size={d} self.size={d} self.shape={d} requested_shape={d}", .{ requested_size, self.size(), self.shape, shape });
            return ZarrayError.ShapeOutOfBounds;
        }
        // for (0.., shape) |i, s| self.shape[i] = s;
        self.alloc.free(self.shape);
        self.shape = try self.alloc.dupe(usize, shape);
        // @compileError("Not yet implemented");
    }

    pub fn get(self: Self, dim: usize) !usize {
        if (dim >= self.len()) return error.DimOutOfBounds;
        return self.shape[dim];
    }

    pub fn rget(self: Self, dim: i32) !usize {
        if (@abs(dim) > self.len()) return error.DimOutOfBounds;
        if (dim < 0) {
            return self.shape[self.shape.len - @abs(dim)];
        }
        return self.shape[@abs(dim)];
    }

    const EqualOptions = struct { strict: bool = false };

    pub fn eq(a: Self, b: Self, options: EqualOptions) bool {
        const dims = @max(a.len(), b.len());
        var i: usize = 0;
        while (i < dims) {
            const dimA = if (i < a.len()) a.shape[a.shape.len - 1 - i] else 1;
            const dimB = if (i < b.len()) b.shape[b.shape.len - 1 - i] else 1;
            if (dimA != dimB) {
                if (options.strict) {
                    // require exactly equal, eq to std.mem.eq tho TODO: sub this later
                    return false;
                }
                if (dimA != 1 and dimB != 1) return false;
            }
            i += 1;
        }
        return true;
    }
};

pub const Range = struct {
    start: usize,
    end: usize,
};

fn calculateBroadcastedShape(shape1: []const usize, shape2: []const usize, allocator: std.mem.Allocator) ![]usize {
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
fn flexSelectOffset(shape: []const usize, indices: []const usize) !usize {
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
    try std.testing.expectEqual(4, flexSelectOffset(&[_]usize{ 2, 3 }, &[_]usize{ 1, 1 }));
    // arr[1, 1, 1] == 4
    try std.testing.expectEqual(4, flexSelectOffset(&[_]usize{ 2, 3 }, &[_]usize{ 1, 1, 1 }));

    // arr = [[[0, 1, 2]
    //        [3, 4, 5]]]
    // arr[1, 1, 1] == 4
    try std.testing.expectEqual(4, flexSelectOffset(&[_]usize{ 1, 2, 3 }, &[_]usize{ 0, 1, 1 }));
}

test "NDArray.clip_norm" {
    const alloc = std.testing.allocator;
    const shape = &[_]usize{ 2, 2 };
    const T = f64;
    const Array = NDArray(T);

    // [2, 2, 1, 4] => 4+4+1+16 = 25 => 5
    var array = try Array.init(&[_]T{ 2, 2, 1, 4 }, shape, alloc);
    defer array.deinit(alloc);
    std.debug.print("Original array: {d}\n", .{array.data});

    const initial_norm = array.l2_norm();
    std.debug.print("Initial norm: {d}\n", .{initial_norm});
    try std.testing.expectEqual(@as(T, 5.0), initial_norm);

    const max_norm: T = 4.0;
    const delta: T = 1e-6;

    array.clip_norm(max_norm, delta);

    const clipped_norm = array.l2_norm();
    std.debug.print("Clipped norm: {d}\n", .{clipped_norm});
    try std.testing.expect(clipped_norm <= max_norm);

    const expected_values = &[_]T{
        2.0 * (max_norm / initial_norm),
        2.0 * (max_norm / initial_norm),
        1.0 * (max_norm / initial_norm),
        4.0 * (max_norm / initial_norm),
    };

    for (0..array.data.len) |i| {
        try std.testing.expectApproxEqAbs(expected_values[i], array.data[i], 1e-5);
    }
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

test "NDArray.add" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f64;
    const Array = NDArray(T);
    const a = try Array.init(@constCast(&[_]T{1}), null, alloc);
    const b = try Array.init(@constCast(&[_]T{ 1, 1, 1 }), null, alloc);
    const sum = try a.add(b, alloc);
    std.debug.print("{d}\n", .{sum.data});
    const b2 = try Array.init(@constCast(&[_]T{3}), null, alloc);
    const sum2 = try a.add(b2, alloc);
    std.debug.print("{d}\n", .{sum2.data});

    const b3 = try Array.init(@constCast(&[_]T{ 3, 2, 1 }), null, alloc);
    const diff = try a.sub(b3, alloc);
    std.debug.print("{d}\n", .{diff.data});
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
    const C1 = try A1.matmul(B1, false, false, alloc);
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
    const C2 = try A2.matmul(B2, false, false, alloc);
    const expected2 = Array.init(&[_]T{ 1, 2, 4, 5 }, &[_]usize{ 2, 2 }, alloc);
    try std.testing.expectEqualDeep(expected2, C2);
}

test "NDArray.matvec" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f64;
    const Array = NDArray(T);

    // multiply a 2x2 and a 2x to get a 2x
    // [[1, 2]  * [[1]  = [[13]
    //  [0, 1]]    [6]]    [6]]
    const A1 = try Array.init(@constCast(&[_]T{ 1, 2, 0, 1 }), &[_]usize{ 2, 2 }, alloc);
    const X1 = try Array.init(@constCast(&[_]T{ 1, 6 }), null, alloc);
    const Y1 = try A1.matvec(X1, false, alloc);
    const expected1 = Array.init(&[_]T{ 13, 6 }, null, alloc);
    try std.testing.expectEqualDeep(expected1, Y1);
}

test "NDArray.sum_along" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f64;
    const Array = NDArray(T);

    // sum a 2x2 along dim=0
    // [[3, 2],    = [7, 3]
    //  [4, 1]]
    const a1 = try Array.init(@constCast(&[_]T{ 3, 2, 4, 1 }), &[_]usize{ 2, 2 }, alloc);
    const a2 = try a1.sum_along(alloc, .{ .dim = 0 });
    const expected1 = try Array.init(&[_]T{ 7, 3 }, null, alloc);
    try std.testing.expectEqualDeep(expected1, a2);
    try std.testing.expect(a2.shape.eq(expected1.shape.*, .{}));
    a2.print();
}

test "NDArray.slice" {
    const alloc = std.testing.allocator;
    var arr = try NDArray(f32).init(&[_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, &[_]usize{ 3, 3 }, alloc);
    defer arr.deinit(alloc);

    // get 2x2 slice from the top-left corner
    const slice = try arr.slice(&[_]Range{ Range{ .start = 0, .end = 2 }, Range{ .start = 0, .end = 2 } });
    defer slice.shape.deinit();
    slice.print();

    // mutate
    const new_values = try NDArray(f32).init(&[_]f32{ 10, 20, 30, 40 }, &[_]usize{ 2, 2 }, alloc);
    defer new_values.deinit(alloc);
    try arr.setSlice(&[_]Range{ Range{ .start = 0, .end = 2 }, Range{ .start = 0, .end = 2 } }, new_values.*);
    new_values.print();
    arr.print();
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        if (gpa.deinit() == .leak) {
            std.debug.print("Leak detected.", .{});
        }
    }
    // var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    const alloc = gpa.allocator();
    // const alloc = arena.allocator();
    // defer arena.deinit();
    const shape1 = &[_]usize{ 2, 3 };
    const shape2 = &[_]usize{ 3, 2 };
    const T = f64;
    const Array = NDArray(T);

    // multiply a 2x3 and a 3x2 to get a 2x2
    var A = try Array.init(@constCast(&[_]T{ 1, 2, 3, 4, 5, 6 }), shape1, alloc);
    const B = try Array.init(@constCast(&[_]T{ 1, 0, 0, 1, 0, 0 }), shape2, alloc);
    var C = try A.matmul(B, false, false, alloc);
    defer A.deinit(alloc);
    defer B.deinit(alloc);
    defer C.deinit(alloc);
    C.print();
}
