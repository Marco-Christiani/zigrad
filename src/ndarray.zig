// TODO:Scalar ops
// TODO: support ND ops. finish broadcast integration
// TODO: element-wise ops to blas if not then vector
// TODO: Shape memory management, put it + tests somewhere
// TODO: exp
// TODO: document bcast rules and shape rules for inplace ewise ops
const std = @import("std");
pub const utils = @import("ndarray/utils.zig");
pub const Shape = @import("ndarray/Shape.zig");
const blas = @import("backend/blas.zig");
const log = std.log.scoped(.zg_ndarray);

pub const MaxOverDimOptions = struct {
    dim: usize,
    keep_dims: bool = false,
    return_offsets: bool = false,
};

pub const GatherOptions = struct {
    indices: *const NDArray(usize),
    dim: usize,
    return_offsets: bool = false,
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
                return error.InvalidShape;
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

        pub fn cast(self: *Self, K: type, allocator: std.mem.Allocator) !*NDArray(K) {
            _ = allocator;
            _ = self;
            @compileError("Not implemented");
            // defer allocator.destroy(self);
            // const result = try allocator.create(NDArray(K));
            // result.* = .{
            //     .data = ,
            //     .shape = self.shape,
            // };
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

        pub fn set(self: Self, indices: []const usize, value: T) !void {
            if (indices.len != self.shape.len()) return error.InvalidShape;
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

        pub fn offsetToPos(self: Self, offset: usize, allocator: std.mem.Allocator) ![]usize {
            const n = self.shape.len();
            const pos = try allocator.alloc(usize, n);
            errdefer allocator.free(pos);

            var remainingIndex = offset;
            for (0..n) |i| {
                pos[i] = remainingIndex / self.shape.strides[i];
                remainingIndex %= self.shape.strides[i];
            }

            return pos;
        }

        pub fn size(self: Self) usize {
            return self.data.len;
        }

        pub fn print(self: Self) void {
            // self.printToWriter(std.io.getStdOut().writer());
            self.printToWriter(std.io.getStdErr().writer()) catch @panic("print failure");
        }

        pub fn printToWriter(self: Self, writer: anytype) !void {
            const alloc = std.heap.page_allocator;
            var shapeStr: []u8 = alloc.alloc(u8, self.shape.len() * @sizeOf(usize)) catch @panic("allocation failed in print");
            defer alloc.free(shapeStr);
            var j: usize = 0;
            var bytes_written: usize = 0;
            for (self.shape.shape, 0..) |s, i| {
                const b = std.fmt.formatIntBuf(shapeStr[bytes_written..shapeStr.len], s, 10, .lower, .{});
                bytes_written += b;
                if (i < self.shape.len() - 1 and j + b < shapeStr.len - 1) {
                    shapeStr[j + b] = 'x';
                    bytes_written += 1;
                } else {
                    break;
                }
                j += 2;
            }
            const preamble = std.fmt.allocPrint(alloc, "NDArray<{any},{s}>", .{ T, shapeStr[0..bytes_written] }) catch @panic("allocation failed in print");
            try writer.writeAll(preamble);
            try utils.printNDSlice(T, self.data, self.shape.shape, writer);
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
                return error.RangeOutOfBounds;
            }

            const stride = try self.getStride(dim);
            const start_index = start * stride;
            const slice_size = (end - start) * stride;

            return self.data[start_index .. start_index + slice_size];
        }

        /// Yields an arbitrary view into a an array. May be non-contiguous. Shape is allocated COM.
        pub fn sliceRanges(self: Self, ranges: []const Range) !Self {
            if (ranges.len != self.shape.len()) {
                return error.InvalidShape;
            }
            if (ranges.len > self.shape.len()) {
                return error.RangeOutOfBounds;
            }

            var buf: [64]usize = undefined; // imposing max ndim here to avoid alloc.
            var new_shape = buf[0..self.shape.len()];
            var start_index: usize = 0;
            var total_elements: usize = 1;
            var stride: usize = 1;

            for (ranges, 0..) |range, i| {
                if (range.end > self.shape.shape[i]) {
                    return error.IndexOutOfBounds;
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

        /// Should be analogous to: https://pytorch.org/docs/stable/generated/torch.slice_scatter.html#torch.slice_scatter
        pub fn setSliceRanges(self: Self, ranges: []const Range, values: Self) !void {
            const slice_ = try self.sliceRanges(ranges);
            defer slice_.shape.deinit();
            if (!slice_.shape.eq(values.shape.*, .{ .strict = true })) {
                return error.IncompatibleShapes;
            }

            var i: usize = 0;
            while (i < slice_.data.len) : (i += 1) {
                slice_.data[i] = values.data[i];
            }
        }

        pub fn getStride(self: Self, dim: usize) !usize {
            if (dim >= self.shape.len()) return error.DimOutOfBounds;
            var s: usize = 1;
            var i: usize = dim + 1;
            while (i < self.shape.len()) : (i += 1) {
                s *= self.shape.shape[i];
            }
            return s;
        }

        /// Element-wise addition
        pub fn add(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            // TODO: standardize this shape creation logic elsewhere (its a micro-optimization)
            const bshape = if (self.shape.size() != other.shape.size()) try self.shape.broadcast(other.shape.*) else try self.shape.copy(allocator);
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
                return error.IncompatibleShapes;
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
                return error.IncompatibleShapes;
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
                return error.IncompatibleShapes;
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
                return error.IncompatibleShapes;
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
        pub fn bmm(self: *const Self, other: *const Self, trans_a: bool, trans_b: bool, allocator: std.mem.Allocator) !*Self {
            return try self._bmmAcc(other, null, 1.0, 0.0, trans_a, trans_b, allocator);
        }

        pub fn _bmmAcc(self: *const Self, other: *const Self, accumulator: ?*Self, alpha: T, beta: T, trans_a: bool, trans_b: bool, allocator: std.mem.Allocator) !*Self {
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

            // Even if nothing needs broadcasting and we have two 2d inputs, we still get a 3d shape w a batch of 1
            // in that case, we dont have any extra broadcasted batch dims
            const broadcast_batch_dims = blk: {
                if (self.shape.len() == 2 and other.shape.len() == 2) break :blk try allocator.alloc(usize, 0);
                break :blk try utils.calculateBroadcastedShape(a_batch_dims, b_batch_dims, allocator);
            };
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

            const n_batch_dims = broadcast_batch_dims.len;
            var output_shape = try allocator.alloc(usize, n_batch_dims + 2);
            defer allocator.free(output_shape);
            if (n_batch_dims > 0) @memcpy(output_shape[0..n_batch_dims], broadcast_batch_dims);
            output_shape[n_batch_dims] = M;
            output_shape[n_batch_dims + 1] = N;

            const n_batches = if (n_batch_dims > 0) utils.prod(broadcast_batch_dims) else 1;
            var result: *Self = blk: {
                if (accumulator) |acc| {
                    if (!Shape.eqRaw(acc.shape.shape, output_shape, .{ .strict = true })) return error.IncompatibleAccumulatorShape;
                    break :blk acc;
                } else break :blk try Self.zeros(output_shape, allocator);
            };
            errdefer if (accumulator == null) result.deinit(allocator);

            const n_batches_a = utils.prod(a_batch_dims);
            const n_batches_b = utils.prod(b_batch_dims);

            const lda = a_cols;
            const ldb = b_cols;
            const ldc = N;

            for (0..n_batches) |i| {
                const a_index = i % n_batches_a;
                const b_index = i % n_batches_b;

                const a_start = a_index * a_rows * a_cols;
                const b_start = b_index * b_rows * b_cols;
                const out_start = i * M * N;

                const a_slice = self.data[a_start .. a_start + a_rows * a_cols];
                const b_slice = other.data[b_start .. b_start + b_rows * b_cols];
                const out_slice = result.data[out_start .. out_start + M * N];

                blas.blas_matmul(T, a_slice, b_slice, out_slice, M, N, K, trans_a, trans_b, lda, ldb, ldc, alpha, beta);
            }
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
            const M = if (trans_a) try self.shape.get(1) else try self.shape.get(0);
            const N = if (trans_a) try self.shape.get(0) else try self.shape.get(1);
            std.debug.assert(N == other.shape.size());
            const output: []T = try allocator.alignedAlloc(T, null, M);
            errdefer allocator.free(output);
            @memset(output, 0);

            blas.blas_matvec(T, self.data, other.data, output, M, N, 1, 0, trans_a);

            const result = try allocator.create(Self);
            result.* = Self{
                .data = output,
                .shape = try Shape.init(&.{M}, allocator),
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
            if (opts.dim >= input_dims) return error.ShapeOutOfBounds;

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

        pub const MaxOverDimResult = struct {
            values: *Self,
            offsets: ?[]usize,
        };

        pub fn maxOverDim(self: *const Self, allocator: std.mem.Allocator, opts: MaxOverDimOptions) !MaxOverDimResult {
            const dim = opts.dim;
            const keep_dims = opts.keep_dims;
            const input_shape = self.shape.shape;
            const input_dims = input_shape.len;
            if (dim >= input_dims) return error.DimOutOfBounds;

            var output_shape = try allocator.alloc(usize, if (keep_dims) input_dims else input_dims - 1);
            defer allocator.free(output_shape);
            var idx: usize = 0;
            for (0..input_dims) |i| {
                if (i != dim) {
                    output_shape[idx] = input_shape[i];
                    idx += 1;
                } else if (keep_dims) {
                    output_shape[idx] = 1;
                    idx += 1;
                }
            }
            const output = try Self.empty(output_shape, allocator);
            errdefer output.deinit(allocator);
            const offsets = if (opts.return_offsets) try allocator.alloc(usize, output.size()) else null;
            errdefer if (offsets) |o| allocator.free(o);

            const max_dim_size = input_shape[dim];
            var slice_size: usize = 1;
            for (dim + 1..input_dims) |i| {
                slice_size *= input_shape[i];
            }

            for (0..output.data.len) |i| {
                var max_val: T = -std.math.inf(T);
                var max_offs: usize = 0;
                const base_offs = (i / slice_size) * (slice_size * max_dim_size) + (i % slice_size);
                for (0..max_dim_size) |j| { // can be optimized if the view along this dim is contiguous (just check dim stride)
                    const curr_offs = base_offs + j * slice_size;
                    const curr_val = self.data[curr_offs];
                    if (curr_val > max_val) {
                        max_val = curr_val;
                        max_offs = curr_offs;
                    }
                }
                output.data[i] = max_val;
                if (offsets) |o| o[i] = max_offs;
            }
            return .{ .values = output, .offsets = offsets };
        }

        pub const GatherResult = struct {
            values: *Self,
            offsets: ?[]usize, // offsets taken, so they dont have to be recomputed
        };

        pub fn gather(self: *const Self, allocator: std.mem.Allocator, opts: GatherOptions) !GatherResult {
            const indices = opts.indices;
            const dim = opts.dim;
            std.debug.assert(self.shape.len() == indices.shape.len());
            for (self.shape.shape, indices.shape.shape, 0..) |src_dim, idx_dim, i| {
                if (i != dim and idx_dim > src_dim) return error.InvalidIndexShape;
            }

            var output = try Self.zeros(indices.shape.shape, allocator);
            errdefer output.deinit(allocator);
            const offsets = if (opts.return_offsets) try allocator.alloc(usize, indices.data.len) else null;
            errdefer if (offsets) |o| allocator.free(o);

            for (0..indices.data.len) |i| {
                const idx_coord = try indices.offsetToPos(i, allocator);
                defer allocator.free(idx_coord);

                const src_coord = try allocator.dupe(usize, idx_coord);
                defer allocator.free(src_coord);

                src_coord[dim] = indices.data[i];
                if (src_coord[dim] >= self.shape.shape[dim]) return error.IndexOutOfBounds;

                const src_offset = self.posToOffset(src_coord);
                const src_value = self.data[src_offset];
                try output.set(idx_coord, src_value);
                if (offsets) |o| o[i] = src_offset;
            }
            return .{ .values = output, .offsets = offsets };
        }

        /// COM
        pub fn take(self: *Self, offsets: *const NDArray(usize), allocator: std.mem.Allocator) !*Self {
            const result = try Self.empty(&.{offsets.data.len}, allocator);
            for (result.data, offsets) |*r, o| r.* = self.data[o];
            return result;
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

        // TODO: copies. Naive, optimize. Support >2d
        pub fn transpose(self: *Self, allocator: std.mem.Allocator) !*Self {
            std.debug.assert(self.shape.len() < 3);
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

pub const Range = struct {
    start: usize,
    end: usize,
};

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
    try std.testing.expectError(error.ShapeOutOfBounds, A._reshape(&[_]usize{9}));
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

    // Run a few tests with matching dims and mismatching dims that are broadcastable
    // 3D batching (1): 1x2x3 and a 1x3x2 to get a 1x2x2
    // 3D batching (2): 2x2x3 and a 2x3x2 to get a 2x2x2
    // 3D batching + broadcasting (1): 2x2x3 and a 3x2 to get a 2x2x2
    // 3D batching + broadcasting (2): 2x2x3 and a 1x3x2 to get a 2x2x2
    // 4D batching + broadcasting (1): 1x2x2x3 and a 3x2 to get a 1x2x2x2
    // 4D batching + broadcasting (2): 2x2x2x3 and a 2x3x2 to get a 2x2x2x2
    // 4D batching + broadcasting (3): 2x2x2x3 and a 2x1x3x2 to get a 2x2x2x2

    // Test 1: 2x2 * 2x2 = 2x2
    {
        // multiply a 2x2 and a 2x2 to get a 2x2
        // [[1, 2]  * [[1, 1]  = [[3, 3]
        //  [0, 1]]    [1, 1]]    [1, 1]]
        const A1 = try Array.init(&.{ 1, 2, 0, 1 }, &[_]usize{ 2, 2 }, alloc);
        const B1 = try Array.init(&.{ 1, 1, 1, 1 }, &[_]usize{ 2, 2 }, alloc);
        const C1 = try A1.bmm(B1, false, false, alloc);
        const expected1 = try Array.init(&.{ 3, 3, 1, 1 }, &[_]usize{ 2, 2 }, alloc);
        try std.testing.expectEqualDeep(expected1.data, C1.data);
        try std.testing.expectEqualDeep(expected1.shape.shape, C1.shape.shape);
        try std.testing.expectEqualDeep(expected1, C1);
    }

    // Test 2: 2x3 * 3x2 = 2x2
    {

        // multiply a 2x3 and a 3x2 to get a 2x2
        // [[1, 2, 3]  * [[1, 0]  = [[1, 2]
        //  [4, 5, 6]]    [0, 1]     [4, 5]]
        //                [0, 0]]
        const A2 = try Array.init(&.{ 1, 2, 3, 4, 5, 6 }, &[_]usize{ 2, 3 }, alloc);
        const B2 = try Array.init(&.{ 1, 0, 0, 1, 0, 0 }, &[_]usize{ 3, 2 }, alloc);
        const C2 = try A2.bmm(B2, false, false, alloc);
        const expected2 = try Array.init(&.{ 1, 2, 4, 5 }, &[_]usize{ 2, 2 }, alloc);
        try std.testing.expectEqualDeep(expected2.data, C2.data);
        try std.testing.expectEqualDeep(expected2.shape.shape, C2.shape.shape);
        try std.testing.expectEqualDeep(expected2, C2);
    }

    // Test 3: 1x2x3 * 1x3x2 = 1x2x2
    {
        const A3 = try Array.init(&.{ 1, 2, 3, 4, 5, 6 }, &[_]usize{ 1, 2, 3 }, alloc);
        const B3 = try Array.init(&.{ 1, 2, 3, 4, 5, 6 }, &[_]usize{ 1, 3, 2 }, alloc);
        const C3 = try A3.bmm(B3, false, false, alloc);
        const expected3 = try Array.init(&.{ 22, 28, 49, 64 }, &[_]usize{ 1, 2, 2 }, alloc);
        try std.testing.expectEqualDeep(expected3.data, C3.data);
        try std.testing.expectEqualDeep(expected3.shape.shape, C3.shape.shape);
        try std.testing.expectEqualDeep(expected3, C3);
    }

    // Test 4: 2x2x3 * 2x3x2 = 2x2x2
    {
        const A4 = try Array.init(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, &[_]usize{ 2, 2, 3 }, alloc);
        const B4 = try Array.init(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, &[_]usize{ 2, 3, 2 }, alloc);
        const C4 = try A4.bmm(B4, false, false, alloc);
        const expected4 = try Array.init(&.{ 22, 28, 49, 64, 220, 244, 301, 334 }, &[_]usize{ 2, 2, 2 }, alloc);
        try std.testing.expectEqualDeep(expected4.data, C4.data);
        try std.testing.expectEqualDeep(expected4.shape.shape, C4.shape.shape);
        try std.testing.expectEqualDeep(expected4, C4);
    }

    // Test 5: 2x2x3 * 3x2 = 2x2x2 (broadcasting)
    {
        const A5 = try Array.init(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, &[_]usize{ 2, 2, 3 }, alloc);
        const B5 = try Array.init(&.{ 1, 2, 3, 4, 5, 6 }, &[_]usize{ 3, 2 }, alloc);
        const C5 = try A5.bmm(B5, false, false, alloc);
        const expected5 = try Array.init(&.{ 22, 28, 49, 64, 76, 100, 103, 136 }, &[_]usize{ 2, 2, 2 }, alloc);
        try std.testing.expectEqualDeep(expected5.data, C5.data);
        try std.testing.expectEqualDeep(expected5.shape.shape, C5.shape.shape);
        try std.testing.expectEqualDeep(expected5, C5);
    }

    // Test 6: 2x2x3 * 1x3x2 = 2x2x2 (broadcasting)
    {
        const A6 = try Array.init(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, &[_]usize{ 2, 2, 3 }, alloc);
        const B6 = try Array.init(&.{ 1, 2, 3, 4, 5, 6 }, &[_]usize{ 1, 3, 2 }, alloc);
        const C6 = try A6.bmm(B6, false, false, alloc);
        const expected6 = try Array.init(&.{ 22, 28, 49, 64, 76, 100, 103, 136 }, &[_]usize{ 2, 2, 2 }, alloc);
        try std.testing.expectEqualDeep(expected6.data, C6.data);
        try std.testing.expectEqualDeep(expected6.shape.shape, C6.shape.shape);
        try std.testing.expectEqualDeep(expected6, C6);
    }

    // Test 7: 1x2x2x3 * 3x2 = 1x2x2x2 (4D broadcasting)
    {
        const A7 = try Array.init(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, &[_]usize{ 1, 2, 2, 3 }, alloc);
        const B7 = try Array.init(&.{ 1, 2, 3, 4, 5, 6 }, &[_]usize{ 3, 2 }, alloc);
        const C7 = try A7.bmm(B7, false, false, alloc);
        const expected7 = try Array.init(&.{ 22, 28, 49, 64, 76, 100, 103, 136 }, &[_]usize{ 1, 2, 2, 2 }, alloc);
        try std.testing.expectEqualDeep(expected7.data, C7.data);
        try std.testing.expectEqualDeep(expected7.shape.shape, C7.shape.shape);
        try std.testing.expectEqualDeep(expected7, C7);

        // NOTE: This is different behavior than torch which treats this as a different case in broadcasting
        const expected7b = try Array.init(&.{ 22, 28, 49, 64, 76, 100, 103, 136 }, &[_]usize{ 2, 1, 2, 2 }, alloc);
        try A7._reshape(&.{ 2, 1, 2, 3 });
        const C7b = try A7.bmm(B7, false, false, alloc);
        try std.testing.expectEqualDeep(expected7b.data, C7b.data);
        try std.testing.expectEqualDeep(expected7b.shape.shape, C7b.shape.shape);
        try std.testing.expectEqualDeep(expected7b, C7b);
    }

    // Test 8: 2x2x2x3 * 2x3x2 = 2x2x2x2 (4D broadcasting)
    {
        const A8 = try Array.init(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 }, &[_]usize{ 2, 2, 2, 3 }, alloc);
        const B8 = try Array.init(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, &[_]usize{ 2, 3, 2 }, alloc);
        const C8 = try A8.bmm(B8, false, false, alloc);
        const expected8 = try Array.init(&.{ 22, 28, 49, 64, 220, 244, 301, 334, 130, 172, 157, 208, 544, 604, 625, 694 }, &[_]usize{ 2, 2, 2, 2 }, alloc);
        try std.testing.expectEqualDeep(expected8.data, C8.data);
        try std.testing.expectEqualDeep(expected8.shape.shape, C8.shape.shape);
        try std.testing.expectEqualDeep(expected8, C8);
    }

    // Test 9: 2x2x2x3 * 2x1x3x2 = 2x2x2x2 (4D broadcasting)
    // FIXME: mm edge case pytorch parity, when this works test 7b should break
    // {
    //     const A9 = try Array.init(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 }, &[_]usize{ 2, 2, 2, 3 }, alloc);
    //     const B9 = try Array.init(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, &[_]usize{ 2, 1, 3, 2 }, alloc);
    //     const C9 = try A9.matmul(B9, false, false, alloc);
    //     const expected9 = try Array.init(&.{ 22, 28, 49, 64, 76, 100, 103, 136, 382, 424, 463, 514, 544, 604, 625, 694 }, &[_]usize{ 2, 2, 2, 2 }, alloc);
    //     expected9.print();
    //     C9.print();
    //     try std.testing.expectEqualDeep(expected9.data, C9.data);
    //     try std.testing.expectEqualDeep(expected9.shape.shape, C9.shape.shape);
    //     try std.testing.expectEqualDeep(expected9, C9);
    // }
}

test "NDArray.matmul with accumulation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f64;
    const Array = NDArray(T);

    // Test: 2x2 * 2x2 with accumulation
    {
        // A = [[1, 2], [3, 4]]
        const A = try Array.init(&.{ 1, 2, 3, 4 }, &[_]usize{ 2, 2 }, alloc);

        // B = [[5, 6], [7, 8]]
        const B = try Array.init(&.{ 5, 6, 7, 8 }, &[_]usize{ 2, 2 }, alloc);

        // C (accumulator) = [[1, 1], [1, 1]]
        const C = try Array.init(&.{ 1, 1, 1, 1 }, &[_]usize{ 2, 2 }, alloc);

        // Perform C = 2 * A * B + 3 * C
        const alpha: T = 2.0;
        const beta: T = 3.0;
        const result = try A._bmmAcc(B, C, alpha, beta, false, false, alloc);

        // Expected result:
        // 2 * [[1, 2], [3, 4]] * [[5, 6], [7, 8]] + 3 * [[1, 1], [1, 1]]
        // = 2 * [[19, 22], [43, 50]] + [[3, 3], [3, 3]]
        // = [[41, 47], [89, 103]]
        const expected = try Array.init(&.{ 41, 47, 89, 103 }, &[_]usize{ 2, 2 }, alloc);

        try std.testing.expectEqualDeep(expected.data, result.data);
        try std.testing.expectEqualDeep(expected.shape.shape, result.shape.shape);

        // Check that C was modified in-place
        try std.testing.expectEqualDeep(expected.data, C.data);
        try std.testing.expectEqualDeep(expected.shape.shape, C.shape.shape);
    }
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

test "NDArray.maxOverDim" {
    const alloc = std.testing.allocator;
    const shape = &[_]usize{ 2, 3, 4 };
    const T = f64;
    const Array = NDArray(T);

    // [[[ 1,  2,  3,  4],
    //   [ 5,  6,  7,  8],
    //   [ 9, 10, 11, 12]],
    //  [[13, 14, 15, 16],
    //   [17, 18, 19, 20],
    //   [21, 22, 23, 24]]]
    var array = try Array.init(@constCast(&[_]T{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24,
    }), shape, alloc);
    defer array.deinit(alloc);

    // max over dim 0
    var result1 = (try array.maxOverDim(alloc, .{ .dim = 0, .keep_dims = true })).values;
    defer result1.deinit(alloc);
    const expected1 = try Array.init(&[_]T{
        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24,
    }, &[_]usize{ 1, 3, 4 }, alloc);
    defer expected1.deinit(alloc);
    try std.testing.expectEqualDeep(expected1, result1);

    // max over dim 1
    const result2 = (try array.maxOverDim(alloc, .{ .dim = 1, .keep_dims = false })).values;
    defer result2.deinit(alloc);
    const expected2 = try Array.init(&[_]T{
        9,  10, 11, 12,
        21, 22, 23, 24,
    }, &[_]usize{ 2, 4 }, alloc);
    defer expected2.deinit(alloc);
    try std.testing.expectEqualDeep(expected2, result2);

    // max over dim 2
    const result3 = (try array.maxOverDim(alloc, .{ .dim = 2, .keep_dims = true })).values;
    defer result3.deinit(alloc);
    const expected3 = try Array.init(&[_]T{
        4,  8,  12,
        16, 20, 24,
    }, &[_]usize{ 2, 3, 1 }, alloc);
    defer expected3.deinit(alloc);
    try std.testing.expectEqualDeep(expected3, result3);
}

test "NDArray.gather" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f32;

    // Case 1
    const input_data = [_]T{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const input_shape = [_]usize{ 3, 3 };
    var input = try NDArray(T).init(&input_data, &input_shape, alloc);
    defer input.deinit(alloc);

    const index_data = [_]usize{ 0, 1, 1, 2, 0, 2 };
    const index_shape = [_]usize{ 3, 2 };
    var index = try NDArray(usize).init(&index_data, &index_shape, alloc);
    defer index.deinit(alloc);

    const output = (try input.gather(alloc, .{ .indices = index, .dim = 1 })).values;
    defer output.deinit(alloc);

    try std.testing.expectEqualSlices(T, &[_]T{ 1, 2, 5, 6, 7, 9 }, output.data);
    try std.testing.expectEqualSlices(usize, &index_shape, output.shape.shape);

    // Case 2: out of bounds
    try index.set(&.{ 0, 0 }, 3);
    try std.testing.expectError(error.IndexOutOfBounds, input.gather(alloc, .{ .indices = index, .dim = 1 }));

    // // Case 3: wrong shape
    // const invalid_index_shape = [_]usize{4};
    // var invalid_index = try NDArray(usize).init(&index_data, &invalid_index_shape, alloc);
    // defer invalid_index.deinit(alloc);
    //
    // try std.testing.expectError(error.InvalidIndexShape, input.gather(invalid_index, 1, alloc));
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
    var C = try A.bmm(B, false, false, alloc);
    defer A.deinit(alloc);
    defer B.deinit(alloc);
    defer C.deinit(alloc);
    C.print();
}
