// TODO:Scalar ops
// TODO: support ND ops. finish broadcast integration
// TODO: element-wise ops to blas if not then vector
// TODO: Shape memory management, put it + tests somewhere
// TODO: exp
// TODO: document bcast rules and shape rules for inplace ewise ops
const std = @import("std");
const blas = @import("../backend/blas.zig");
const log = std.log.scoped(.zg_zarray);

pub const ZarrayError = error{
    InvalidShape,
    Unbroadcastable,
    InvalidIndex,
    IndexOutOfBounds,
    ShapeOutOfBounds,
    DimOutOfBounds,
    IncompatibleShapes,
    RangeOutOfBounds,
};

pub fn NDArray(comptime T: type) type {
    return struct {
        const Self = @This();
        shape: *Shape,
        data: []T,
        view: bool = false,

        /// Values and shape are copied. COM.
        pub fn init(values: []const T, shape: ?[]const usize, allocator: std.mem.Allocator) !*Self {
            const result = try allocator.create(Self);
            result.* = Self{
                .data = try allocator.dupe(T, values),
                .shape = try Shape.init(shape orelse &[_]usize{values.len}, allocator),
            };
            if (result.shape.size() != result.data.len) {
                log.err("Invalid shape: result.shape.size()={d} result.data.len={d}", .{ result.shape.size(), result.data.len });
                return ZarrayError.InvalidShape;
            }
            return result;
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            if (!self.view) allocator.free(self.data);
            self.shape.deinit();
            allocator.destroy(self);
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

        pub fn zeros(shape: []const usize, allocator: std.mem.Allocator) !*Self {
            const result = try Self.empty(shape, allocator);
            result.fill(0);
            return result;
        }

        pub fn copy(self: Self, allocator: std.mem.Allocator) !*Self {
            const result = try allocator.create(Self);
            result.* = .{
                .data = try allocator.dupe(T, self.data),
                .shape = try self.shape.copy(allocator),
            };
            return result;
        }

        pub fn fill(self: Self, val: T) void {
            @memset(self.data, val);
        }

        pub fn logShape(self: Self, comptime msg: ?[]const u8) void {
            log.debug("{s} shape: {d}", .{
                if (msg) |n| n else "",
                self.shape.shape,
            });
        }

        // TODO: make a copying reshape like tensor does
        pub fn _reshape(self: *const Self, shape: []const usize) !void {
            try self.shape._reshape(shape);
        }

        pub fn get(self: Self, indices: []const usize) T {
            return self.data[self.posToOffset(indices)];
        }

        pub fn set(self: Self, indices: []const usize, value: T) ZarrayError!void {
            if (indices.len != self.shape.len()) return ZarrayError.InvalidShape;
            std.debug.assert(indices.len == self.shape.len());
            self.data[self.posToOffset(indices)] = value;
        }

        pub fn posToOffset(self: Self, indices: []const usize) usize {
            if (indices.len < self.shape.len()) log.warn("Hope you know what you are doing.", .{});
            std.debug.assert(indices.len >= self.shape.len());
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
            const pos = allocator.alloc(usize, self.data.len) catch @panic("Failed to allocate shape slice.");
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

        pub fn size(self: Self) !void {
            return self.data.len;
        }

        pub fn print(self: Self) void {
            const alloc = std.heap.page_allocator;
            var shapeStr: []u8 = alloc.alloc(u8, self.shape.len() * @sizeOf(usize)) catch @panic("allocation failed in print");
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

        /// View into contiguous slice along a single dim. Shape is allocated COM.
        pub fn slice(self: Self, dim: usize, start: usize, end: usize) !Self {
            var new_shape = try self.shape.copy(self.shape.alloc);
            new_shape.shape[dim] = end - start;
            return .{
                .shape = new_shape,
                .data = try self.sliceRawNoAlloc(dim, start, end),
                .view = true,
            };
        }

        /// Completely mutable into a contiguous slice along a single dim, better know what you are doing.
        /// if you touch the shape youre most likely up schitts creek.
        pub fn sliceUnsafeNoAlloc(self: Self, dim: usize, start: usize, end: usize) !Self {
            return .{
                .shape = self.shape,
                .data = try self.sliceRawNoAlloc(dim, start, end),
                .view = true,
            };
        }

        /// View into a raw contiguous slice along a single dim.
        pub fn sliceRawNoAlloc(self: Self, dim: usize, start: usize, end: usize) ![]T {
            if (start > end or end > self.shape.shape[dim]) {
                return ZarrayError.RangeOutOfBounds;
            }

            const stride = try self.getStride(dim);
            const start_index = start * stride;
            const slice_size = (end - start) * stride;

            return self.data[start_index .. start_index + slice_size];
        }

        /// Yields an arbitrary view into a an array. May be non-contiguous. Shape is allocated COM.
        pub fn sliceRanges(self: Self, ranges: []const Range) !Self {
            if (ranges.len != self.shape.len()) {
                return ZarrayError.InvalidShape;
            }
            if (ranges.len > self.shape.len()) {
                return ZarrayError.RangeOutOfBounds;
            }

            var buf: [64]usize = undefined; // imposing max ndim here to avoid alloc.
            var new_shape = buf[0..self.shape.len()];
            var start_index: usize = 0;
            var total_elements: usize = 1;
            var stride: usize = 1;

            for (ranges, 0..) |range, i| {
                if (range.end > self.shape.shape[i]) {
                    return ZarrayError.IndexOutOfBounds;
                }
                new_shape[i] = range.end - range.start;
                total_elements *= new_shape[i];
                start_index += range.start * stride;
                stride *= self.shape.shape[i];
            }

            return .{
                .shape = try Shape.init(new_shape, self.shape.alloc),
                .data = self.data[start_index .. start_index + total_elements],
                .view = true,
            };
        }

        pub fn setSliceRanges(self: Self, ranges: []const Range, values: Self) !void {
            const slice_ = try self.sliceRanges(ranges);
            defer slice_.shape.deinit();
            if (!slice_.shape.eq(values.shape.*, .{ .strict = true })) {
                return ZarrayError.IncompatibleShapes;
            }

            var i: usize = 0;
            while (i < slice_.data.len) : (i += 1) {
                slice_.data[i] = values.data[i];
            }
        }

        fn getStride(self: Self, dim: usize) !usize {
            if (dim >= self.shape.len()) return ZarrayError.DimOutOfBounds;
            var s: usize = 1;
            var i: usize = dim + 1;
            while (i < self.shape.len()) : (i += 1) {
                s *= self.shape.shape[i];
            }
            return s;
        }

        /// Element-wise addition
        pub fn add(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const bshape = try self.shape.broadcast(other.shape.*);
            const values = try allocator.alloc(T, bshape.size());
            for (0..values.len) |i| {
                values[i] = self.data[i % self.data.len] + other.data[i % other.data.len];
            }
            const result = try allocator.create(Self);
            result.* = Self{
                .data = values,
                .shape = bshape,
            };
            return result;
        }

        /// In-place element-wise addition
        pub fn _add(self: *const Self, other: *const Self) !*const Self {
            if (!Shape.eq(self.shape.*, other.shape.*, .{}) or other.shape.size() > self.shape.size()) {
                log.err("_add() self.shape={d} other.shape={d}", .{ self.shape.shape, other.shape.shape });
                return ZarrayError.IncompatibleShapes;
            }
            for (0..self.data.len) |i| self.data[i] += other.data[i % other.data.len];
            return self;
        }

        /// In-place scalar add
        pub fn _add_scalar(self: Self, scalar: T) void {
            for (self.data) |*val| val.* += scalar;
        }

        /// Element-wise subtraction
        pub fn sub(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const bshape = try self.shape.broadcast(other.shape.*);
            const values = try allocator.alloc(T, bshape.size());
            for (0..values.len) |i| values[i] = self.data[i % self.data.len] - other.data[i % other.data.len];
            const result = try allocator.create(Self);
            result.* = Self{
                .data = values,
                .shape = bshape,
            };
            return result;
        }

        /// In-place element-wise subtraction
        pub fn _sub(self: *const Self, other: *const Self) !*const Self {
            if (!Shape.eq(self.shape.*, other.shape.*, .{}) or other.shape.size() > self.shape.size()) {
                log.err("_sub() self.shape={d} other.shape={d}", .{ self.shape.shape, other.shape.shape });
                return ZarrayError.IncompatibleShapes;
            }
            for (0..self.data.len) |i| self.data[i] -= other.data[i % other.data.len];
            return self;
        }

        /// Element-wise multiplication
        pub fn mul(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const bshape = try self.shape.broadcast(other.shape.*);
            const values = try allocator.alloc(T, bshape.size());
            for (0..values.len) |i| values[i] = self.data[i % self.data.len] * other.data[i % other.data.len];
            const result = try allocator.create(Self);
            result.* = Self{
                .data = values,
                .shape = bshape,
            };
            return result;
        }

        /// In-place element-wise multiplication
        pub fn _mul(self: *const Self, other: *const Self) !*const Self {
            if (!Shape.eq(self.shape.*, other.shape.*, .{}) or other.shape.size() > self.shape.size()) {
                log.err("_mul() self.shape={d} other.shape={d}", .{ self.shape.shape, other.shape.shape });
                return ZarrayError.IncompatibleShapes;
            }
            for (0..self.data.len) |i| self.data[i] *= other.data[i % other.data.len];
            return self;
        }

        /// Element-wise division
        pub fn div(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const bshape = try self.shape.broadcast(other.shape.*);
            const values = try allocator.alloc(T, bshape.size());
            for (0..values.len) |i| values[i] = self.data[i % self.data.len] / other.data[i % other.data.len];
            const result = try allocator.create(Self);
            result.* = Self{
                .data = values,
                .shape = bshape,
            };
            return result;
        }

        /// In-place element-wise division
        pub fn _div(self: *const Self, other: *const Self) !*const Self {
            if (!Shape.eq(self.shape.*, other.shape.*, .{}) or other.shape.size() > self.shape.size()) {
                log.err("_div() self.shape={d} other.shape={d}", .{ self.shape.shape, other.shape.shape });
                return ZarrayError.IncompatibleShapes;
            }
            for (0..self.data.len) |i| self.data[i] /= other.data[i % other.data.len];
            return self;
        }

        /// Element-wise sum. COM.
        pub fn sum(self: *const Self, allocator: std.mem.Allocator) !*Self {
            return try Self.init(&[_]T{self.sumNoAlloc()}, &.{1}, allocator);
        }

        /// Element-wise sum.
        pub fn sumNoAlloc(self: *const Self) T {
            // TODO: tired rn
            // const VT = @Vector(16, T);
            // var s: T = 0;
            // var i: usize = 0;
            // while (i < self.data.len - 16) : (i += 16) s += @reduce(.Add, @as(VT, self.data[i .. i + 16].*));
            // for (i..self.data.len) |j| s += self.data[j];
            // return s;
            var s: T = 0;
            for (0..self.data.len) |i| s += self.data[i];
            return s;
        }

        /// COM.
        pub fn max(self: *const Self, allocator: std.mem.Allocator) !*Self {
            if (self.data.len == 0) return error.EmptyArray;
            return Self.init(&[_]T{std.mem.max(T, self.data)}, null, allocator);
        }

        // TODO: naive
        /// Copies. COM.
        pub fn exp(self: *const Self, allocator: std.mem.Allocator) !*Self {
            const result = try Self.empty(self.shape.shape, allocator);
            for (self.data, 0..) |val, i| {
                result.data[i] = @exp(val);
            }
            return result;
        }

        /// In-place e^x
        pub fn _exp(self: *const Self) void {
            for (self.data) |*val| val.* = @exp(val.*);
        }

        /// In-place scaling
        pub fn _scale(self: *const Self, scalar: T) void {
            blas.blas_scale(T, scalar, self.data);
        }

        /// (...)-Mat-Mat: ND x KD (N,K>2) and broadcastable
        /// Simple dim rules: (M, K) x (K, N) = (M, N)
        pub fn matmul(self: *const Self, other: *const Self, trans_a: bool, trans_b: bool, allocator: std.mem.Allocator) !*Self {
            if (self.shape.len() < 2) {
                try self._reshape(&[_]usize{ 1, try self.shape.get(0) });
            }
            if (other.shape.len() < 2) {
                try other._reshape(&[_]usize{ 1, try other.shape.get(0) });
            }

            const a_matrix_dims = self.shape.shape[self.shape.len() - 2 ..];
            const b_matrix_dims = other.shape.shape[other.shape.len() - 2 ..];
            const a_batch_dims = if (self.shape.len() > 2) self.shape.shape[0 .. self.shape.len() - 2] else &[_]usize{1};
            const b_batch_dims = if (other.shape.len() > 2) other.shape.shape[0 .. other.shape.len() - 2] else &[_]usize{1};

            const broadcast_batch_dims = try calculateBroadcastedShape(a_batch_dims, b_batch_dims, allocator);
            defer allocator.free(broadcast_batch_dims);

            const a_rows = a_matrix_dims[0];
            const a_cols = a_matrix_dims[1];
            const b_rows = b_matrix_dims[0];
            const b_cols = b_matrix_dims[1];

            const M = if (trans_a) a_cols else a_rows;
            const K = if (trans_a) a_rows else a_cols;
            const N = if (trans_b) b_rows else b_cols;

            if ((if (trans_a) a_rows else a_cols) != (if (trans_b) b_cols else b_rows)) {
                std.debug.panic("Incompatible matrix dimensions for matmul: {}x{} and {}x{} bcasted batch dims {d} (trans_a={}, trans_b={})", .{
                    a_rows, a_cols, b_rows, b_cols, broadcast_batch_dims, trans_a, trans_b,
                });
            }

            const batch_size = prod(broadcast_batch_dims);
            const output: []T = try allocator.alignedAlloc(T, null, batch_size * M * N);
            errdefer allocator.free(output);

            var output_shape = try allocator.alloc(usize, broadcast_batch_dims.len + 2);
            defer allocator.free(output_shape);
            @memcpy(output_shape[0..broadcast_batch_dims.len], broadcast_batch_dims);
            output_shape[broadcast_batch_dims.len] = M;
            output_shape[broadcast_batch_dims.len + 1] = N;

            const a_batch_size = prod(a_batch_dims);
            const b_batch_size = prod(b_batch_dims);

            const lda = a_cols;
            const ldb = b_cols;

            for (0..batch_size) |i| {
                const a_index = i % a_batch_size;
                const b_index = i % b_batch_size;

                const a_start = a_index * a_rows * a_cols;
                const b_start = b_index * b_rows * b_cols;
                const out_start = i * M * N;

                const a_slice = self.data[a_start .. a_start + a_rows * a_cols];
                const b_slice = other.data[b_start .. b_start + b_rows * b_cols];
                const out_slice = output[out_start .. out_start + M * N];

                blas.blas_matmul(T, a_slice, b_slice, out_slice, M, N, K, trans_a, trans_b, lda, ldb, N);
            }
            var result = try allocator.create(Self);
            result.* = Self{
                .data = output,
                .shape = try Shape.init(output_shape, allocator),
            };
            errdefer result.deinit(allocator);
            // NOTE: squeezing here means an unnecessary allocation and we should optimize this out later since
            // we mocked an outer batch dim of 1 for the inputs, so the output will have an outer batch dim of 1
            // but the inputs were both 2x2 so we should return a 2x2, thus squeeze it out
            if (self.shape.len() == 2 and other.shape.len() == 2) try result.shape._squeeze(); // FIXME: artifact of failed re-re-factor
            return result;
        }

        pub fn dot(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            if (self.shape.len() > 1 or other.shape.len() > 1) std.debug.panic("Dot product only valid for 1d vectors even if there are dummy outer dimensions.\n", .{});
            if (self.data.len != other.data.len) std.debug.panic("Incompatible lengths for dot product: {d} and {d}\n", .{ self.data.len, other.data.len });

            const output: T = blas.blas_dot(T, self.data, other.data);
            return try Self.init(&[_]T{output}, &[_]usize{1}, allocator);
        }

        pub fn outer(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            if (try self.shape.realdims() != 1 or try other.shape.realdims() != 1) {
                std.debug.panic(
                    "Outer product only valid for 1d vectors even if there are dummy outer dimensions. Got {d} {d}\n",
                    .{ self.shape.shape, other.shape.shape },
                );
            }
            if (self.shape.len() != 1 or other.shape.len() != 1) {
                try self.shape._squeeze();
                try other.shape._squeeze();
            }
            const M: usize = try self.shape.get(0);
            const N: usize = try other.shape.get(0);
            const output: []T = try allocator.alignedAlloc(T, null, M * N);
            errdefer allocator.free(output);
            @memset(output, 0);

            blas.blas_outer(T, self.data, other.data, output, 1);

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
            @memset(output, 0);

            blas.blas_matvec(T, self.data, other.data, output, try self.shape.get(0), try self.shape.get(1), 1, 0, trans_a);

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

            // TODO: rm alloc
            var output_shape = try allocator.alloc(usize, if (opts.keep_dims) input_dims else input_dims - 1);
            defer allocator.free(output_shape);
            var idx: usize = 0;
            for (0..input_dims) |i| {
                if (i != opts.dim) {
                    output_shape[idx] = input_shape[i];
                    idx += 1;
                } else if (opts.keep_dims) {
                    output_shape[idx] = 1;
                    idx += 1;
                }
            }
            const output = try Self.empty(output_shape, allocator);
            errdefer output.deinit(allocator);
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

        pub fn l2_norm(self: *const Self) T {
            return blas.blas_nrm2(T, self.data, 1);
        }

        pub fn clip_norm(self: *const Self, max_norm: T, delta: T) void {
            const norm = self.l2_norm();
            if (norm > max_norm) {
                const scale = max_norm / (norm + delta);
                for (self.data) |*value| {
                    value.* *= scale;
                }
            }
        }

        /// Unbroadcast to target shape
        /// TODO: think ab this later but could pointer swap (may give compiler more freedom to optimize)
        pub fn unbroadcast(self: *Self, target_shape: *Shape, allocator: std.mem.Allocator) !*Self {
            var result: *Self = self;

            while (result.shape.len() > target_shape.len()) {
                const temp = result;
                result = try temp.sum_along(allocator, .{ .dim = 0 });
                if (temp != result) {
                    temp.deinit(allocator);
                }
            }

            if (result.shape.len() == target_shape.len()) {
                for (target_shape.shape, 0..) |s, dimi| {
                    if (s == 1 and result.shape.shape[dimi] != 1) {
                        const temp = result;
                        result = try temp.sum_along(allocator, .{ .dim = dimi, .keep_dims = true });
                        if (temp != self) {
                            temp.deinit(allocator);
                        }
                    }
                }
            }

            return result;
        }

        // TODO: inplace transpose. Naive, optimize.
        pub fn transpose(self: *Self, allocator: std.mem.Allocator) !*Self {
            if (self.shape.len() == 1) return self;
            const new_shape = [_]usize{ self.shape.shape[1], self.shape.shape[0] };
            var result = try Self.empty(&new_shape, allocator);
            for (0..self.shape.shape[0]) |i| {
                for (0..self.shape.shape[1]) |j| {
                    result.data[j * self.shape.shape[0] + i] = self.data[i * self.shape.shape[1] + j];
                }
            }
            return result;
        }
    };
}

pub const Shape = struct {
    const Self = @This();
    shape: []usize,
    alloc: std.mem.Allocator,

    pub fn init(shape: []const usize, allocator: std.mem.Allocator) !*Self {
        if (shape.len == 0) return ZarrayError.InvalidShape;
        const self = try allocator.create(Self);
        self.* = Self{
            .alloc = allocator,
            .shape = try allocator.dupe(usize, shape),
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.alloc.free(self.shape);
        self.alloc.destroy(self);
    }

    pub fn copy(self: Shape, allocator: std.mem.Allocator) !*Shape {
        return try Self.init(self.shape, allocator);
    }

    pub fn broadcast(self: Self, other: Self) !*Self {
        const dims = @max(self.len(), other.len());
        const result_shape = try self.alloc.alloc(usize, dims);
        errdefer self.alloc.free(result_shape);
        var i: usize = 0;
        while (i < dims) : (i += 1) {
            const dim_a = if (i < self.shape.len) self.shape[self.shape.len - 1 - i] else 1;
            const dim_b = if (i < other.shape.len) other.shape[other.shape.len - 1 - i] else 1;
            if (dim_a != dim_b and dim_a != 1 and dim_b != 1) {
                log.err("Cannot broadcast {d} and {d}", .{ self.shape, other.shape });
                return ZarrayError.Unbroadcastable;
            }
            result_shape[dims - 1 - i] = @max(dim_a, dim_b);
        }
        const result = try self.alloc.create(Self);
        result.* = Self{ .shape = result_shape, .alloc = self.alloc };
        return result;
    }

    // TODO: usage should be minimized (or size stored but rarely needed this was)
    pub fn size(self: Self) usize {
        return prod(self.shape);
    }

    pub fn len(self: Self) usize {
        return self.shape.len;
    }

    /// Functional number of dims if the shape was squeezed
    pub fn realdims(self: Self) !usize {
        if (self.shape.len == 0) return error.EmptyShape;
        if (self.size() == 1) return 1; // scalar
        var ndim: usize = 0;
        for (self.shape) |e| {
            if (e != 1) ndim += 1;
        }
        return ndim;
    }

    pub fn _squeeze(self: *Self) !void {
        // TODO: realloc is a thing
        var newshape = try self.alloc.alloc(usize, try self.realdims());
        // scalar case
        if (self.size() == 1) {
            newshape[0] = 1;
        } else {
            var j: usize = 0;
            for (0..self.shape.len) |i| {
                if (self.shape[i] != 1) {
                    newshape[j] = self.shape[i];
                    j += 1;
                }
            }
        }
        self.alloc.free(self.shape);
        self.shape = newshape;
    }

    pub fn _reshape(self: *Self, shape: []const usize) !void {
        // TODO: realloc is a thing
        const requested_size = prod(shape);
        if (requested_size != self.size()) {
            log.info("ShapeOutOfBounds requested_size={d} self.size={d} self.shape={d} requested_shape={d}", .{ requested_size, self.size(), self.shape, shape });
            return ZarrayError.ShapeOutOfBounds;
        }
        self.alloc.free(self.shape);
        self.shape = try self.alloc.dupe(usize, shape);
    }

    pub fn get(self: Self, dim: usize) !usize {
        if (dim >= self.len()) return ZarrayError.DimOutOfBounds;
        return self.shape[dim];
    }

    pub fn rget(self: Self, dim: i32) !usize {
        if (@abs(dim) > self.len()) return ZarrayError.DimOutOfBounds;
        if (dim >= 0) return error.InvalidDim; // must be a negative number
        return self.shape[self.shape.len - @abs(dim)];
    }

    const EqualOptions = struct {
        /// require exactly equal
        strict: bool = false,
    };

    pub fn eq(a: Self, b: Self, options: EqualOptions) bool {
        if (options.strict) return std.mem.eql(usize, a.shape, b.shape);
        const dims = @max(a.len(), b.len());
        var i: usize = 0;
        while (i < dims) : (i += 1) {
            const dim_a = if (i < a.len()) a.shape[a.shape.len - 1 - i] else 1;
            const dim_b = if (i < b.len()) b.shape[b.shape.len - 1 - i] else 1;
            if (dim_a != dim_b and dim_a != 1 and dim_b != 1) return false;
        }
        return true;
    }
};

pub const Range = struct {
    start: usize,
    end: usize,
};

pub fn prod(dims: []const usize) usize {
    if (dims.len == 0) return 0;
    var s: usize = 1;
    for (dims) |f| s *= f;
    return s;
}

fn calculateBroadcastedShape(shape1: []const usize, shape2: []const usize, allocator: std.mem.Allocator) ![]usize {
    const dims = @max(shape1.len, shape2.len);
    const result_shape = try allocator.alloc(usize, dims);
    var i: usize = 0;
    while (i < dims) : (i += 1) {
        const dimA = if (i < shape1.len) shape1[shape1.len - 1 - i] else 1;
        const dimB = if (i < shape2.len) shape2[shape2.len - 1 - i] else 1;
        if (dimA != dimB and dimA != 1 and dimB != 1) {
            allocator.free(result_shape);
            return ZarrayError.Unbroadcastable;
        }
        result_shape[dims - 1 - i] = @max(dimA, dimB);
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

    const initial_norm = array.l2_norm();
    try std.testing.expectEqual(@as(T, 5.0), initial_norm);

    const max_norm: T = 4.0;
    const delta: T = 1e-6;

    array.clip_norm(max_norm, delta);

    const clipped_norm = array.l2_norm();
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

    var A = try Array.init(&[_]T{ 1, 2, 3, 4, 5, 6 }, null, alloc);
    try A._reshape(&[_]usize{ 2, 3 });
    try std.testing.expectError(ZarrayError.ShapeOutOfBounds, A._reshape(&[_]usize{9}));
    const v1 = A.get(&[_]usize{ 0, 0 });
    try A._reshape(&[_]usize{ 2, 3 });
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
    try std.testing.expectEqualSlices(T, &[_]T{ 2, 2, 2 }, sum.data);
    const b2 = try Array.init(@constCast(&[_]T{3}), null, alloc);
    const sum2 = try a.add(b2, alloc);
    try std.testing.expectEqualSlices(T, &[_]T{4}, sum2.data);

    const b3 = try Array.init(@constCast(&[_]T{ 3, 2, 1 }), null, alloc);
    const diff = try a.sub(b3, alloc);
    try std.testing.expectEqualSlices(T, &[_]T{ -2, -1, 0 }, diff.data);
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
    const alloc = std.testing.allocator;
    const T = f64;
    const Array = NDArray(T);

    // sum a 2x2 along dim=0
    // [[3, 2],    = [7, 3]
    //  [4, 1]]
    const a1 = try Array.init(@constCast(&[_]T{ 3, 2, 4, 1 }), &[_]usize{ 2, 2 }, alloc);
    defer a1.deinit(alloc);
    const a2 = try a1.sum_along(alloc, .{ .dim = 0 });
    defer a2.deinit(alloc);
    const expected1 = try Array.init(&[_]T{ 7, 3 }, null, alloc);
    defer expected1.deinit(alloc);
    try std.testing.expectEqualDeep(expected1, a2);
    try std.testing.expect(a2.shape.eq(expected1.shape.*, .{}));
}

test "NDArray.slice" {
    const alloc = std.testing.allocator;
    var arr = try NDArray(f32).init(&[_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, &[_]usize{ 3, 3 }, alloc);
    defer arr.deinit(alloc);

    // get 2x2 slice from the top-left corner
    var slice = try arr.sliceRanges(&[_]Range{ Range{ .start = 0, .end = 2 }, Range{ .start = 0, .end = 2 } });
    defer slice.shape.deinit();

    // mutate
    const new_values = try NDArray(f32).init(&[_]f32{ 10, 20, 30, 40 }, &[_]usize{ 2, 2 }, alloc);
    defer new_values.deinit(alloc);
    try slice.setSliceRanges(&[_]Range{ Range{ .start = 0, .end = 2 }, Range{ .start = 0, .end = 2 } }, new_values.*);
    const exp = try NDArray(f32).init(&[_]f32{ 10, 20, 30, 40, 5, 6, 7, 8, 9 }, &[_]usize{ 3, 3 }, alloc);
    defer exp.deinit(alloc);
    try std.testing.expectEqualDeep(exp, arr);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        if (gpa.deinit() == .leak) {
            std.debug.print("Leak detected.", .{});
        }
    }
    const alloc = gpa.allocator();
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
