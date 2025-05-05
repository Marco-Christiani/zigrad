const std = @import("std");
pub const utils = @import("ndarray/utils.zig");
pub const Shape = @import("ndarray/shape.zig");
const log = std.log.scoped(.zg_ndarray);
const zg = @import("zigrad.zig");
const DeviceReference = zg.DeviceReference;
const backend = zg.backend;
const builtin = @import("builtin");

const opspec = zg.opspec;

pub const MaxAlongOptions = struct {
    dim: usize,
    keep_dims: bool = false,
};

pub const GatherOptions = struct {
    indices: NDArray(usize),
    dim: usize,
    return_offsets: bool = false,
};

pub fn NDArray(comptime T: type) type {
    // TODO: document bcast rules and shape rules for inplace ewise ops
    return struct {
        const Self = @This();
        pub const Mode = enum { none, view, shared };
        /// Can be safely accessed. See `Shape`
        shape: Shape,

        /// Cannot be safely accessed. Despite its appearance, the data may lie
        /// on device memory. Therefore, direct access is unsafe unless data is
        /// known to reside on host memory.
        data: []T,

        /// Whether `data` is a view into another array.
        mode: Mode = .none,

        /// Values and shape are copied. COM.
        pub fn init(values: []const T, shape: ?[]const usize, device: DeviceReference) !Self {
            const _shape = Shape.init(shape orelse &.{values.len});

            const n = _shape.size();
            if (n != values.len or n == 0 or values.len == 0) {
                log.err("Invalid shape: result.shape.size()={} result.data.len={}", .{ _shape.size(), values.len });
                return error.InvalidShape;
            }
            return .{
                .data = try device.mem_dupe(T, values),
                .shape = _shape,
            };
        }

        pub fn deinit(self: *Self, device: DeviceReference) void {
            if (self.mode == .none) device.mem_free(self.data);
            self.* = undefined;
        }

        pub fn empty(shape: []const usize, device: DeviceReference) !Self {
            const _shape = Shape.init(shape);
            const _size = _shape.size();
            std.debug.assert(shape.len > 0 and _size > 0);
            return .{ .data = try device.mem_alloc(T, _size), .shape = _shape };
        }

        pub fn zeros(shape: []const usize, device: DeviceReference) !Self {
            const result = try Self.empty(shape, device);
            result.fill(0, device);
            return result;
        }

        pub fn random(shape: []const usize, device: DeviceReference, op: zg.RandType) !Self {
            const self = try Self.empty(shape, device);
            device.mem_random(T, self.data, op, zg.random);
            return self;
        }

        pub fn copy(self: Self, device: DeviceReference) !Self {
            return .{ .data = try device.mem_dupe(T, self.data), .shape = self.shape };
        }

        pub fn share(self: Self) Self {
            return .{ .data = self.data, .shape = self.shape, .mode = .shared };
        }

        pub fn cast(self: *Self, K: type, _: DeviceReference) !NDArray(K) {
            _ = self;
            @compileError("Not implemented");
            // defer allocator.destroy(self);
            // const result = try allocator.create(NDArray(K));
            // result.* = .{
            //     .data = ,
            //     .shape = self.shape,
            // };
        }

        pub fn fill(self: Self, val: T, device: DeviceReference) void {
            // std.debug.print("FILLING: {}\n", .{val});
            device.mem_fill(T, self.data, val);
        }

        pub fn log_shape(self: Self, comptime msg: ?[]const u8) void {
            log.debug("{s} shape: {}", .{ if (msg) |n| n else "", self.shape.slice() });
        }

        pub fn reshape(self: *const Self, new_shape: []const usize) Shape {
            const tmp = Shape.init(new_shape);
            std.debug.assert(tmp.size() == self.size());
            return tmp;
        }

        pub fn _reshape(self: *Self, new_shape: []const usize) void {
            self.shape = self.reshape(new_shape);
        }

        pub fn size(self: Self) usize {
            return self.data.len;
        }

        pub fn print(self: Self, device: DeviceReference) void {
            self.print_to_writer(std.io.getStdErr().writer(), device) catch @panic("print failure");
        }

        pub fn print_to_writer(self: Self, writer: anytype, device: DeviceReference) !void {
            return print_to_writer_impl(self, self.data, writer, device.is_host(), device);
        }

        fn has_mem_transfer(comptime DT: type) bool {
            return if (@typeInfo(DT) == .@"struct") @hasDecl(DT, "mem_transfer") else false;
        }

        fn print_to_writer_impl(self: Self, _data: []T, writer: anytype, is_host: bool, device: DeviceReference) !void {
            if (comptime has_mem_transfer(@TypeOf(device))) {
                if (!is_host) {
                    device.sync();
                    const _host_data = try device.allocator.alloc(T, _data.len);
                    defer device.allocator.free(_host_data);
                    device.mem_transfer(T, _data, _host_data, .DtoH);
                    device.sync();
                    return self.print_to_writer_impl(_host_data, writer, true, device);
                }
            }

            const alloc = std.heap.page_allocator;
            const shape_str: []u8 = alloc.alloc(u8, @as(usize, self.shape.len) * @sizeOf(usize)) catch @panic("allocation failed in print");
            defer alloc.free(shape_str);
            var j: usize = 0;
            var bytes_written: usize = 0;
            for (self.shape.slice(), 0..) |s, i| {
                const b = std.fmt.formatIntBuf(shape_str[bytes_written..shape_str.len], s, 10, .lower, .{});
                bytes_written += b;
                if (i < self.shape.len - 1 and j + b < shape_str.len - 1) {
                    shape_str[j + b] = 'x';
                    bytes_written += 1;
                } else {
                    break;
                }
                j += 2;
            }
            const preamble = std.fmt.allocPrint(alloc, "NDArray<{any},{s}>", .{ T, shape_str[0..bytes_written] }) catch @panic("allocation failed in print");
            try writer.writeAll(preamble);
            try utils.print_ndslice(T, _data, self.shape.slice(), writer);
        }

        /// View into contiguous slice along a single dim. Shape is allocated COM.
        pub fn slice(self: Self, dim: usize, start: usize, end: usize) !Self {
            var new_shape = self.shape;
            new_shape.set(dim, end - start);
            return .{
                .shape = new_shape,
                .data = try self.slice_raw_no_alloc(dim, start, end),
                .mode = .view,
            };
        }

        /// Completely mutable into a contiguous slice along a single dim, better know what you are doing.
        /// if you touch the shape youre most likely up schitts creek.
        pub fn slice_unsafe_no_alloc(self: Self, dim: usize, start: usize, end: usize) !Self {
            return .{
                .shape = self.shape,
                .data = try self.slice_raw_no_alloc(dim, start, end),
                .mode = .view,
            };
        }

        /// View into a raw contiguous slice along a single dim.
        pub fn slice_raw_no_alloc(self: Self, dim: usize, start: usize, end: usize) ![]T {
            if (start > end or end > self.shape.get(dim)) {
                return error.RangeOutOfBounds;
            }

            const stride = try self.get_stride(dim);
            const start_index = start * stride;
            const slice_size = (end - start) * stride;

            return self.data[start_index .. start_index + slice_size];
        }

        /// Yields an arbitrary view into a an array. May be non-contiguous. Shape is allocated COM.
        pub fn slice_ranges(self: Self, ranges: []const Range) !Self {
            if (ranges.len != self.shape.len) {
                return error.InvalidShape;
            }
            if (ranges.len > self.shape.len) {
                return error.RangeOutOfBounds;
            }

            var buf: [8]usize = undefined; // imposing max ndim here to avoid alloc.
            var new_shape = buf[0..self.shape.len];
            var start_index: usize = 0;
            var total_elements: usize = 1;
            var stride: usize = 1;

            for (ranges, 0..) |range, i| {
                if (range.end > self.shape.get(i)) {
                    return error.IndexOutOfBounds;
                }
                new_shape[i] = range.end - range.start;
                total_elements *= new_shape[i];
                start_index += range.start * stride;
                stride *= self.shape.get(i);
            }

            return .{
                .shape = Shape.init(new_shape),
                .data = self.data[start_index .. start_index + total_elements],
                .mode = .view,
            };
        }

        // this function is somewhat expensive. Try to calculate
        // strides once and use the repeatedly in your work load.
        pub fn get_stride(self: Self, dim: usize) usize {
            // use at your own caution - don't over flow the buffer
            std.debug.assert(dim < self.shape.len);
            return self.shape.strides().get(dim);
        }

        inline fn elwise(x: *const Self, y: *const Self, z: *Self, device: DeviceReference, Op: type) !void {
            std.debug.assert(z.mode != .view);
            if (builtin.mode == .Debug and !x.shape.compatible(y.shape)) {
                log.err("_" ++ Op.__name__ ++ "() self.shape={} other.shape={}", .{ x.shape, y.shape });
                return error.IncompatibleShapes;
            }
            device.dispatch(Op{ .x = x.data, .y = y.data, .z = z.data });
        }

        inline fn elwise_alloc(x: *const Self, y: *const Self, device: DeviceReference, Op: type) !Self {
            const z_shape = try x.shape.broadcast(y.shape);
            var z = try Self.empty(z_shape.slice(), device);
            errdefer z.deinit(device);
            try elwise(x, y, &z, device, Op);
            return z;
        }

        // element wise operations (broadcasting)
        pub fn add(self: *const Self, other: Self, device: DeviceReference) !Self {
            return elwise_alloc(self, &other, device, opspec.add(T));
        }
        pub fn sub(self: *const Self, other: Self, device: DeviceReference) !Self {
            return elwise_alloc(self, &other, device, opspec.sub(T));
        }
        pub fn mul(self: *const Self, other: Self, device: DeviceReference) !Self {
            return elwise_alloc(self, &other, device, opspec.mul(T));
        }
        pub fn div(self: *const Self, other: Self, device: DeviceReference) !Self {
            return elwise_alloc(self, &other, device, opspec.div(T));
        }

        // inplace element wise operations (broadcasting)
        pub fn _add(self: *Self, other: Self, device: DeviceReference) !void {
            return elwise(self, &other, self, device, opspec.add(T));
        }
        pub fn _sub(self: *Self, other: Self, device: DeviceReference) !void {
            return elwise(self, &other, self, device, opspec.sub(T));
        }
        pub fn _mul(self: *Self, other: Self, device: DeviceReference) !void {
            return elwise(self, &other, self, device, opspec.mul(T));
        }
        pub fn _div(self: *Self, other: Self, device: DeviceReference) !void {
            return elwise(self, &other, self, device, opspec.div(T));
        }

        // inplace element wise operations (broadcasting)
        pub fn add_(self: *const Self, other: *Self, device: DeviceReference) !void {
            return elwise(self, other, other, device, opspec.add(T));
        }
        pub fn sub_(self: *const Self, other: *Self, device: DeviceReference) !void {
            return elwise(self, other, other, device, opspec.sub(T));
        }
        pub fn mul_(self: *const Self, other: *Self, device: DeviceReference) !void {
            return elwise(self, other, other, device, opspec.mul(T));
        }
        pub fn div_(self: *const Self, other: *Self, device: DeviceReference) !void {
            return elwise(self, other, other, device, opspec.div(T));
        }

        pub fn sum(self: Self, device: DeviceReference) !Self {
            const sum_arr = try Self.empty(&.{1}, device);
            device.dispatch(opspec.sum(T){ .x = self.data, .y = sum_arr.data });
            return sum_arr;
        }

        pub fn max(self: Self, device: DeviceReference) !Self {
            const max_arr = try Self.empty(&.{1}, device);
            device.dispatch(opspec.max_fwd(T){ .x = self.data, .y = max_arr.data });
            return max_arr;
        }

        pub fn exp(self: Self, device: DeviceReference) !Self {
            const result = try Self.empty(self.shape.slice(), device);
            device.dispatch(opspec.exp_fwd(T){ .x = self.data, .y = result.data });
            return result;
        }

        pub fn _exp(self: *Self, device: DeviceReference) void {
            std.debug.assert(self.mode != .view);
            device.dispatch(opspec.exp_fwd(T){ .x = self.data, .y = self.data });
        }

        pub fn _scale(self: *Self, alpha: T, device: DeviceReference) void {
            std.debug.assert(self.mode != .view);
            device.dispatch(opspec.scale(T){ .x = self.data, .alpha = alpha });
        }

        pub const BmmConfig = struct {
            trans_a: bool = false,
            trans_b: bool = false,
            alpha: T = 1.0,
            beta: T = 0.0,
        };

        pub fn bmm(self: *const Self, other: Self, device: DeviceReference, config: BmmConfig) !Self {
            return bmm_acc_impl(T, self, &other, null, config.alpha, config.beta, config.trans_a, config.trans_b, device);
        }

        /// # Design decisions regarding batch gemm support
        ///
        /// If we broadcasted an operand then we cant just call batch gemm which expects batch_size number of matrices
        /// for all inputs
        /// Note: if we continue with the current behavior, we gain a fast path where if the outer strides match, we can
        ///    directly call batch gemm. This is essentially what we currently do and will not be correct for some cases
        ///    but will be fast.
        ///
        /// There are a few ways to address this, listed below. We should likely have both, but for now we can go with
        ///  option 1 and assume that we will not have to support very large calculations soon.
        ///
        /// ## Option 1
        ///
        ///   - Expand to the target shape, actually performing the broacast.
        ///   - With all shapes matching, call batch gemm
        ///
        /// Discussion:
        ///
        ///   - Requires intermediate allocations.
        ///   - Actually expanding the shape exerts memory pressure, for some problems this wont be feasible.
        ///   - Have to implement expanding and another thing to optimize.
        ///   - Probably fastest
        ///
        /// ## Option 2
        ///
        ///   - Continue to handle the broadcasting manually
        ///   - If the outer strides (stridea and strideb) mismatch, we have to broadcast
        ///   - For all dimensions inside of the first broadcasted dim we call batch gemm.
        ///
        /// Discussion:
        ///
        ///   - Less memory pressure by avoiding expansion
        ///   - Easily parallelized, but need to be careful about what APIs provide what guarantees here
        ///   - Will probably have the same edge case issue
        ///
        pub fn bmm_acc_(self: *const Self, other: Self, out: *Self, device: DeviceReference, config: BmmConfig) !void {
            _ = try bmm_acc_impl(T, self, &other, out, config.alpha, config.beta, config.trans_a, config.trans_b, device);
        }

        pub fn bmm_acc(self: *const Self, other: Self, out_shape: Shape, device: DeviceReference, config: BmmConfig) !Self {
            var out = try Self.empty(out_shape.slice(), device);
            errdefer out.deinit(device);
            try self.bmm_acc_(other, &out, device, config);
            return out;
        }

        pub fn dot(self: Self, other: Self, device: DeviceReference) !Self {
            std.debug.assert(self.shape.len == 1 and other.shape.len == 1);
            std.debug.assert(self.data.len == other.data.len);
            const out_arr = try Self.empty(&.{1}, device);
            device.dispatch(opspec.dot(T){ .x = self.data, .y = other.data, .z = out_arr.data });
            return out_arr;
        }

        pub fn outer(self: Self, other: Self, device: DeviceReference) !Self {
            std.debug.assert(self.shape.realdims() == 1 and other.shape.realdims() == 1);
            var a_shape = self.shape;
            var b_shape = other.shape;

            if (self.shape.len != 1 or other.shape.len != 1) {
                a_shape._squeeze();
                b_shape._squeeze();
            }

            const M: usize = a_shape.get(0);
            const N: usize = b_shape.get(0);
            const output = try Self.empty(&.{ M, N }, device);

            device.dispatch(opspec.outer(T){ .x = self.data, .y = other.data, .A = output.data, .alpha = 0 });

            return output;
        }

        pub const MatvecConfig = struct {
            trans_a: bool = false,
            alpha: T = 1.0,
            beta: T = 0.0,
        };

        pub fn matvec(A: Self, x: Self, device: DeviceReference, config: MatvecConfig) !Self {
            std.debug.assert(A.shape.len == 2);
            std.debug.assert(x.shape.len == 1);
            const M = if (config.trans_a) A.shape.get(1) else A.shape.get(0);
            const N = if (config.trans_a) A.shape.get(0) else A.shape.get(1);
            std.debug.assert(N == x.shape.size());
            const y = try Self.empty(&.{M}, device);
            device.dispatch(opspec.matvec(T){
                .A = A.data,
                .x = x.data,
                .y = y.data,
                .m = M,
                .n = N,
                .trans_a = config.trans_a,
                .alpha = config.alpha,
                .beta = config.beta,
            });
            return y;
        }

        pub fn matvec_(A: Self, x: Self, y: *Self, device: DeviceReference, config: MatvecConfig) void {
            std.debug.assert(A.shape.len == 2);
            std.debug.assert(x.shape.len == 1);
            const M = if (config.trans_a) A.shape.get(1) else A.shape.get(0);
            const N = if (config.trans_a) A.shape.get(0) else A.shape.get(1);
            std.debug.assert(N == x.shape.size());
            std.debug.assert(M == y.shape.size());
            device.dispatch(opspec.matvec(T){
                .A = A.data,
                .x = x.data,
                .y = y.data,
                .m = M,
                .n = N,
                .trans_a = config.trans_a,
                .alpha = config.alpha,
                .beta = config.beta,
            });
        }

        /// Performs `self = alpha*other + self` in place.
        /// Shapes must match (although practically the op is possible under other conditions)
        pub fn _axpy(self: Self, other: Self, alpha: T, device: DeviceReference) void {
            std.debug.assert(self.mode != .view);
            std.debug.assert(self.shape.equal(other.shape));
            device.dispatch(opspec.axpy(T){ .x = other.data, .y = self.data, .alpha = &alpha });
        }

        pub const SumOpts = struct {
            dim: usize,
            keep_dims: bool = false,
        };

        pub fn sum_along(self: Self, device: DeviceReference, opts: SumOpts) !Self {
            std.debug.assert(self.shape.len <= 8);

            if (opts.dim >= self.shape.len)
                return error.ShapeOutOfBounds;

            var output_shape = self.shape;

            if (opts.keep_dims) {
                output_shape.set(opts.dim, 1);
            } else {
                output_shape.remove(opts.dim);
            }

            const output = try Self.empty(output_shape.slice(), device);
            errdefer output.deinit(device);

            device.dispatch(opspec.sum_along(T){
                .x = self.data,
                .x_shape = self.shape.slice(),
                .y = output.data,
                .dim = opts.dim,
            });

            return output;
        }

        pub fn max_along(self: Self, device: DeviceReference, opts: MaxAlongOptions) !Self {
            std.debug.assert(self.shape.len <= 8);

            if (opts.dim >= self.shape.len)
                return error.ShapeOutOfBounds;

            var output_shape = self.shape;

            if (opts.keep_dims) {
                output_shape.set(opts.dim, 1);
            } else {
                output_shape.remove(opts.dim);
            }

            const output = try Self.zeros(output_shape.slice(), device);
            errdefer output.deinit(device);

            device.dispatch(opspec.max_along(T){
                .x = self.data,
                .x_shape = self.shape.slice(),
                .y = output.data,
                .dim = opts.dim,
            });

            return output;
        }

        pub const GatherResult = struct {
            values: Self,
            offsets: ?[]usize, // offsets taken, so they dont have to be recomputed
            device: DeviceReference, // reference for both offsets and values
            pub fn deinit(self: *GatherResult) void {
                self.values.deinit(self.device);
                if (self.offsets) |o|
                    self.device.allocator.free(o); // TODO: cache?
            }
        };

        // TODO: proper gather backend kernel.
        pub fn gather(self: Self, device: DeviceReference, opts: GatherOptions) !GatherResult {
            const indices = opts.indices;
            const dim = opts.dim;

            std.debug.assert(self.shape.len == indices.shape.len);
            for (self.shape.slice(), indices.shape.slice(), 0..) |src_dim, idx_dim, i| {
                if (i != dim and idx_dim > src_dim) return error.InvalidIndexShape;
            }
            // to-owned-slice allows us to properly free regardless of exit, otherwise
            // we could try to free on error and because the user didn't ask for offsets
            var offsets = try device.allocator.alloc(usize, indices.data.len); // TODO: cache?
            defer if (!opts.return_offsets) device.allocator.free(offsets); // TODO: cache?

            const values = try Self.empty(indices.shape.slice(), device);
            const idx_strides = indices.shape.strides();
            const src_strides = self.shape.strides();

            for (0..indices.data.len) |i| {
                const idx_coord = idx_strides.offset_to_pos(i);

                var src_coord = idx_coord;
                src_coord.set(dim, indices.data[i]);

                if (src_coord.get(dim) >= self.shape.get(dim))
                    return error.IndexOutOfBounds;

                offsets[i] = src_strides.pos_to_offset(src_coord);
            }

            device.mem_take(T, self.data, offsets, values.data);

            return .{
                .values = values,
                .offsets = if (opts.return_offsets) offsets else null,
                .device = device,
            };
        }

        /// COM
        pub fn take(self: Self, offsets: []const usize, device: DeviceReference) !Self {
            const result = try Self.empty(&.{offsets.len}, device);
            device.mem_take(T, self.data, offsets, result.data);
            return result;
        }

        pub fn l2_norm(self: Self, device: DeviceReference) !Self {
            const result = try Self.empty(&.{1}, device);
            device.dispatch(opspec.nrm2(T){ .x = self.data, .y = result.data });
            return result;
        }

        pub fn _clip_norm(self: Self, max_norm: T, delta: T, device: DeviceReference) void {
            std.debug.assert(self.mode != .view);
            device.dispatch(opspec.clip_nrm2(T){ .x = self.data, .max_norm = max_norm, .delta = delta });
        }

        pub fn clamp(self: Self, vmin: T, vmax: T, device: DeviceReference) !Self {
            std.debug.assert(vmin <= vmax);
            const out = try Self.empty(self.shape.slice(), device);
            device.dispatch(opspec.clamp_fwd(T){ .x = self.data, .y = out.data, .max = vmax, .min = vmin });
            return out;
        }

        pub fn _clamp(self: Self, vmin: T, vmax: T, device: DeviceReference) void {
            std.debug.assert(self.mode != .view);
            std.debug.assert(vmin <= vmax);
            device.dispatch(opspec.clamp_fwd(T){ .x = self.data, .y = self.data, .max = vmax, .min = vmin });
        }

        // TODO: Maybe we should standardize things in this way:
        //    fn name -> returns value
        //    fn _name -> modifies this value
        //    fn name_ -> modifies output argumet
        pub fn unbroadcast_(
            self: *const Self,
            out: *Self,
            device: DeviceReference,
            config: struct {
                alpha: T = 1.0,
                beta: T = 0.0,
            },
        ) !void {
            std.debug.assert(self.shape.len >= out.shape.len);
            std.debug.assert(self.data.len >= out.data.len);
            std.debug.assert(self.data.len % out.data.len == 0);
            std.debug.assert(self.data.ptr != out.data.ptr);
            std.debug.assert(out.mode != .view);

            const scratch: []T = outer: {
                const delta = self.shape.len - out.shape.len;
                // we need enough scratch memory for at least the first reduce
                // but if that yields the same size as the out.shape, then we know only
                // one reduce is required and we can write directly to the out memory (zero scratch)
                const coef: u64 = inner: {
                    if (0 < delta) // always reduce the outer dimensions
                        break :inner Shape.slice_size(self.shape.head(delta));

                    // there is no difference - unbroadcast dispatches to scaled copy
                    const mm = self.shape.mismatch(out.shape) orelse break :outer &.{};

                    // we only need the size of the first reduced dimension
                    break :inner self.shape.get(mm.a_pos);
                };
                const n = out.data.len * coef;

                break :outer try device.mem_scratch(T, if (n == self.data.len) 0 else n);
            };

            device.dispatch(opspec.unbroadcast(T){
                .x = self.data,
                .x_shape = self.shape.slice(),
                .y = out.data,
                .y_shape = out.shape.slice(),
                .scratch = scratch,
                .alpha = config.alpha,
                .beta = config.beta,
            });
        }

        pub fn unbroadcast(self: *const Self, new_shape: Shape, device: DeviceReference) !Self {
            var out = try Self.empty(new_shape.size(), device);
            self.unbroadcast_(&out, device, .{});
            return out;
        }

        pub fn _unbroadcast(self: *Self, new_shape: Shape, device: DeviceReference) !void {
            std.debug.assert(self.mode != .view);

            if (self.shape.equal(new_shape))
                return;

            var swap_mem = try self.unbroadcast(new_shape, device);
            defer swap_mem.deinit();

            std.mem.swap(Self, self, &swap_mem);
        }

        pub fn broadcast_(
            self: *const Self,
            out: *Self,
            device: DeviceReference,
            config: struct {
                alpha: T = 1.0,
                beta: T = 0.0,
            },
        ) !void {
            std.debug.assert(out.shape.len >= self.shape.len);
            std.debug.assert(self.data.len % self.data.len == 0);
            std.debug.assert(self.data.ptr != out.data.ptr);
            std.debug.assert(out.mode != .view);

            device.dispatch(opspec.broadcast(T){
                .x = self.data,
                .x_shape = self.shape.slice(),
                .y = out.data,
                .y_shape = out.shape.slice(),
                .alpha = config.alpha,
                .beta = config.beta,
            });
        }

        pub fn broadcast(self: *const Self, new_shape: Shape, device: DeviceReference) !Self {
            var out = try Self.empty(new_shape.size(), device);
            self.broadcast_(&out, device, .{});
            return out;
        }

        pub fn _broadcast(self: *Self, new_shape: Shape, device: DeviceReference) !void {
            std.debug.assert(self.mode != .view);

            if (self.shape.equal(new_shape))
                return;

            var swap_mem = try self.broadcast(new_shape, device);
            defer swap_mem.deinit();

            std.mem.swap(Self, self, &swap_mem);
        }

        // TODO: copies. Naive, optimize. Support >2d
        pub fn transpose(self: *const Self, device: DeviceReference) !Self {
            std.debug.assert(self.shape.len < 3);

            if (self.shape.len == 1)
                return try self.copy(device);

            const out_data = try device.mem_alloc(T, self.shape.size());

            device.dispatch(opspec.transpose(T){
                .A = self.data,
                .B = out_data,
                .m = self.shape.get(0),
                .n = self.shape.get(1),
                .alpha = 0.0,
            });

            return .{
                .data = out_data,
                .shape = Shape.init(&.{
                    self.shape.get(1),
                    self.shape.get(0),
                }),
                .vies = false,
            };
        }
    };
}

// Moved out of ndarray so we don't export it as the
// public facing api.
fn bmm_acc_impl(
    T: type,
    A: *const NDArray(T),
    B: *const NDArray(T),
    accumulator: ?*NDArray(T),
    alpha: T,
    beta: T,
    trans_a: bool,
    trans_b: bool,
    device: DeviceReference,
) !NDArray(T) {
    const Self = NDArray(T);
    var a_shape = A.shape;
    var b_shape = B.shape;

    // vectors get unsqueezed to rank-2 tensors
    if (a_shape.len < 2) a_shape._unsqueeze();
    if (b_shape.len < 2) b_shape._unsqueeze();

    const a_batch_dims = Shape.init(if (a_shape.len > 2) a_shape.crop(0, 2) else &.{1});
    const b_batch_dims = Shape.init(if (b_shape.len > 2) b_shape.crop(0, 2) else &.{1});

    // Even if nothing needs broadcasting and we have two 2d inputs, we still get a 3d shape w a batch of 1
    // in that case, we dont have any extra broadcasted batch dims
    const broadcast_batch_dims = blk: {
        if (a_shape.len == 2 and b_shape.len == 2)
            break :blk Shape.empty
        else
            break :blk try a_batch_dims.broadcast(b_batch_dims);
    };

    const a_matrix_dims = a_shape.tail(2);
    const b_matrix_dims = b_shape.tail(2);
    const a_rows = a_matrix_dims[0];
    const a_cols = a_matrix_dims[1];
    const b_rows = b_matrix_dims[0];
    const b_cols = b_matrix_dims[1];

    const M = if (trans_a) a_cols else a_rows;
    const K = if (trans_a) a_rows else a_cols;
    const N = if (trans_b) b_rows else b_cols;

    const C_shape = Shape.merge(&.{ broadcast_batch_dims.slice(), &.{ M, N } });

    var C: Self = if (accumulator) |acc| acc.* else try Self.empty(C_shape.slice(), device);
    errdefer if (accumulator == null) C.deinit(device);

    if (builtin.mode == .Debug) {
        if (accumulator) |_| {
            std.debug.assert(C.mode != .view);
            if (!C.shape.compatible(C_shape)) {
                std.debug.panic("Expected accumulator shape {} but got {}", .{ C_shape, C.shape });
            }
        }

        if ((if (trans_a) a_rows else a_cols) != (if (trans_b) b_cols else b_rows)) {
            std.debug.panic("Incompatible matrix dimensions for matmul: {}x{} and {}x{} bcasted batch dims {} (trans_a={}, trans_b={})", .{
                a_rows, a_cols, b_rows, b_cols, broadcast_batch_dims, trans_a, trans_b,
            });
        }
    }

    const n_batches_a = a_batch_dims.size();
    const n_batches_b = b_batch_dims.size();
    const n_batches_c = broadcast_batch_dims.size();

    device.dispatch(opspec.bmm_acc(T){
        .A = A.data,
        .A_shape = &.{ n_batches_a, M, K },
        .B = B.data,
        .B_shape = &.{ n_batches_b, K, N },
        .C = C.data,
        .C_shape = &.{ n_batches_c, M, N },
        .trans_a = trans_a,
        .trans_b = trans_b,
        .lda = a_cols,
        .ldb = b_cols,
        .ldc = N,
        .alpha = alpha,
        .beta = beta,
    });

    return C;
}

pub const Range = struct {
    start: usize,
    end: usize,
};

test "NDArray._clip_norm,l2_norm" {
    const allocator = std.testing.allocator;
    var device = zg.device.HostDevice.init(allocator);
    defer device.deinit();
    const shape = &[_]usize{ 2, 2 };
    const T = f64;
    const Array = NDArray(T);
    // [2, 2, 1, 4] => 4+4+1+16 = 25 => 5
    var array = try Array.init(&[_]T{ 2, 2, 1, 4 }, shape, device.reference());
    defer array.deinit(device.reference());

    const initial_norm_ndarray = try array.l2_norm(device.reference());
    try std.testing.expectEqual(1, initial_norm_ndarray.size());
    const initial_norm = initial_norm_ndarray.data[0];
    try std.testing.expectEqual(@as(T, 5.0), initial_norm);

    const max_norm: T = 4.0;
    const delta: T = 1e-6;

    array._clip_norm(max_norm, delta, device.reference());

    const clipped_norm_ndarray = try array.l2_norm(device.reference());
    const clipped_norm = clipped_norm_ndarray.data[0];
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

test "NDArray._clamp" {
    const allocator = std.testing.allocator;
    var device = zg.device.HostDevice.init(allocator);
    defer device.deinit();
    const shape = &[_]usize{5};
    const T = f64;
    const Array = NDArray(T);
    // [-2, -1, 0, 1, 2] -> [-1, -1, 0, 1, 1]
    var array = try Array.init(&[_]T{ -2, -1, 0, 1, 2 }, shape, device.reference());
    defer array.deinit(device.reference());

    var expected = try Array.init(&[_]T{ -1, -1, 0, 1, 1 }, shape, device.reference());
    defer expected.deinit(device.reference());
    const vmin = -1;
    const vmax = 1;
    array._clamp(vmin, vmax, device.reference());
    try std.testing.expectEqualSlices(T, expected.data, array.data);
}

// We need to support cross device scalar transfer for this
// first. This is doable, but needs to be discussed.

//test "NDArray.reshape" {
//    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//    defer arena.deinit();
//    const allocator = arena.allocator();
//    var cpu = zg.device.HostDevice.init(allocator);
//    defer cpu.deinit();
//    const device = cpu.reference();
//    const T = f64;
//    const Array = NDArray(T);
//
//    var A = try Array.init(&[_]T{ 1, 2, 3, 4, 5, 6 }, null, device);
//    A._reshape(&.{ 2, 3 });
//    try std.testing.expectError(error.ShapeOutOfBounds, A._reshape(&[_]usize{9}));
//    const v1 = A.get(&[_]usize{ 0, 0 });
//    try A._reshape(&[_]usize{ 2, 3 });
//    const v2 = A.get(&[_]usize{ 0, 0 });
//    try std.testing.expectEqual(v1, v2);
//}

test "NDArray.add" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();
    const T = f64;
    const Array = NDArray(T);
    const a = try Array.init(&[_]T{1}, null, device);
    const b = try Array.init(&[_]T{ 1, 1, 1 }, null, device);
    const sum = try a.add(b, device);
    try std.testing.expectEqualSlices(T, &[_]T{ 2, 2, 2 }, sum.data);
    const b2 = try Array.init(&[_]T{3}, null, device);
    const sum2 = try a.add(b2, device);
    try std.testing.expectEqualSlices(T, &[_]T{4}, sum2.data);

    const b3 = try Array.init(&[_]T{ 3, 2, 1 }, null, device);
    const diff = try a.sub(b3, device);
    try std.testing.expectEqualSlices(T, &[_]T{ -2, -1, 0 }, diff.data);
}

test "NDArray.dot" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();
    const T = f64;
    const Array = NDArray(T);

    const a = try Array.init(&.{ 1, 2, 3 }, null, device);
    const b = try Array.init(&.{ 1, 1, 1 }, null, device);
    const result = try a.dot(b, device);
    const expected = Array.init(&.{6}, null, device);
    try std.testing.expectEqualDeep(expected, result);
}

test "NDArray.matmul" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();

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
        const A1 = try Array.init(&.{ 1, 2, 0, 1 }, &[_]usize{ 2, 2 }, device);
        const B1 = try Array.init(&.{ 1, 1, 1, 1 }, &[_]usize{ 2, 2 }, device);
        const C1 = try A1.bmm(B1, device, .{});
        const expected1 = try Array.init(&.{ 3, 3, 1, 1 }, &[_]usize{ 2, 2 }, device);
        try std.testing.expectEqualDeep(expected1.data, C1.data);
        try std.testing.expectEqualDeep(expected1.shape.slice(), C1.shape.slice());
        try std.testing.expectEqualDeep(expected1, C1);
    }

    // Test 2: 2x3 * 3x2 = 2x2
    {

        // multiply a 2x3 and a 3x2 to get a 2x2
        // [[1, 2, 3]  * [[1, 0]  = [[1, 2]
        //  [4, 5, 6]]    [0, 1]     [4, 5]]
        //                [0, 0]]
        const A1 = try Array.init(&.{ 1, 2, 3, 4, 5, 6 }, &[_]usize{ 2, 3 }, device);
        const B1 = try Array.init(&.{ 1, 0, 0, 1, 0, 0 }, &[_]usize{ 3, 2 }, device);
        const C1 = try A1.bmm(B1, device, .{});
        const expected2 = try Array.init(&.{ 1, 2, 4, 5 }, &[_]usize{ 2, 2 }, device);
        try std.testing.expectEqualDeep(expected2.data, C1.data);
        try std.testing.expectEqualDeep(expected2.shape.slice(), C1.shape.slice());
        try std.testing.expectEqualDeep(expected2, C1);
    }

    // Test 3: 1x2x3 * 1x3x2 = 1x2x2
    {
        const A1 = try Array.init(&.{ 1, 2, 3, 4, 5, 6 }, &[_]usize{ 1, 2, 3 }, device);
        const B1 = try Array.init(&.{ 1, 2, 3, 4, 5, 6 }, &[_]usize{ 1, 3, 2 }, device);
        const C1 = try A1.bmm(B1, device, .{});
        const expected3 = try Array.init(&.{ 22, 28, 49, 64 }, &[_]usize{ 1, 2, 2 }, device);
        try std.testing.expectEqualDeep(expected3.data, C1.data);
        try std.testing.expectEqualDeep(expected3.shape.slice(), C1.shape.slice());
        try std.testing.expectEqualDeep(expected3, C1);
    }

    // Test 4: 2x2x3 * 2x3x2 = 2x2x2
    {
        const A1 = try Array.init(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, &[_]usize{ 2, 2, 3 }, device);
        const B1 = try Array.init(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, &[_]usize{ 2, 3, 2 }, device);
        const C1 = try A1.bmm(B1, device, .{});
        const expected4 = try Array.init(&.{ 22, 28, 49, 64, 220, 244, 301, 334 }, &[_]usize{ 2, 2, 2 }, device);
        try std.testing.expectEqualDeep(expected4.data, C1.data);
        try std.testing.expectEqualDeep(expected4.shape.slice(), C1.shape.slice());
        try std.testing.expectEqualDeep(expected4, C1);
    }

    // Test 5: 2x2x3 * 3x2 = 2x2x2 (broadcasting)
    {
        const A1 = try Array.init(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, &[_]usize{ 2, 2, 3 }, device);
        const B1 = try Array.init(&.{ 1, 2, 3, 4, 5, 6 }, &[_]usize{ 3, 2 }, device);
        const C1 = try A1.bmm(B1, device, .{});
        const expected5 = try Array.init(&.{ 22, 28, 49, 64, 76, 100, 103, 136 }, &[_]usize{ 2, 2, 2 }, device);
        try std.testing.expectEqualDeep(expected5.data, C1.data);
        try std.testing.expectEqualDeep(expected5.shape.slice(), C1.shape.slice());
        try std.testing.expectEqualDeep(expected5, C1);
    }

    // Test 6: 2x2x3 * 1x3x2 = 2x2x2 (broadcasting)
    {
        const A1 = try Array.init(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, &[_]usize{ 2, 2, 3 }, device);
        const B1 = try Array.init(&.{ 1, 2, 3, 4, 5, 6 }, &[_]usize{ 1, 3, 2 }, device);
        const C1 = try A1.bmm(B1, device, .{});
        const expected6 = try Array.init(&.{ 22, 28, 49, 64, 76, 100, 103, 136 }, &[_]usize{ 2, 2, 2 }, device);
        try std.testing.expectEqualDeep(expected6.data, C1.data);
        try std.testing.expectEqualDeep(expected6.shape.slice(), C1.shape.slice());
        try std.testing.expectEqualDeep(expected6, C1);
    }

    // Test 7: 1x2x2x3 * 3x2 = 1x2x2x2 (4D broadcasting)
    {
        var A1 = try Array.init(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, &[_]usize{ 1, 2, 2, 3 }, device);
        const B1 = try Array.init(&.{ 1, 2, 3, 4, 5, 6 }, &[_]usize{ 3, 2 }, device);
        const C1 = try A1.bmm(B1, device, .{});
        const expected7 = try Array.init(&.{ 22, 28, 49, 64, 76, 100, 103, 136 }, &[_]usize{ 1, 2, 2, 2 }, device);
        try std.testing.expectEqualDeep(expected7.data, C1.data);
        try std.testing.expectEqualDeep(expected7.shape.slice(), C1.shape.slice());
        try std.testing.expectEqualDeep(expected7, C1);

        // NOTE: This is different behavior than torch which treats this as a different case in broadcasting
        const expected7b = try Array.init(&.{ 22, 28, 49, 64, 76, 100, 103, 136 }, &[_]usize{ 2, 1, 2, 2 }, device);
        A1._reshape(&.{ 2, 1, 2, 3 });
        const C1b = try A1.bmm(B1, device, .{});
        try std.testing.expectEqualDeep(expected7b.data, C1b.data);
        try std.testing.expectEqualDeep(expected7b.shape.slice(), C1b.shape.slice());
        try std.testing.expectEqualDeep(expected7b, C1b);
    }

    // Test 8: 2x2x2x3 * 2x3x2 = 2x2x2x2 (4D broadcasting)
    {
        const A1 = try Array.init(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 }, &[_]usize{ 2, 2, 2, 3 }, device);
        const B1 = try Array.init(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, &[_]usize{ 2, 3, 2 }, device);
        const C1 = try A1.bmm(B1, device, .{});
        const expected8 = try Array.init(&.{ 22, 28, 49, 64, 220, 244, 301, 334, 130, 172, 157, 208, 544, 604, 625, 694 }, &[_]usize{ 2, 2, 2, 2 }, device);
        try std.testing.expectEqualDeep(expected8.data, C1.data);
        try std.testing.expectEqualDeep(expected8.shape.slice(), C1.shape.slice());
        try std.testing.expectEqualDeep(expected8, C1);
    }

    // Test 9: 2x2x2x3 * 2x1x3x2 = 2x2x2x2 (4D broadcasting)
    // FIXME: mm edge case pytorch parity, when this works test 7b should break
    // {
    //     const A1 = try Array.init(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 }, &[_]usize{ 2, 2, 2, 3 }, alloc);
    //     const B1 = try Array.init(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, &[_]usize{ 2, 1, 3, 2 }, alloc);
    //     const C1 = try A1.matmul(B1, false, false, alloc);
    //     const expected9 = try Array.init(&.{ 22, 28, 49, 64, 76, 100, 103, 136, 382, 424, 463, 514, 544, 604, 625, 694 }, &[_]usize{ 2, 2, 2, 2 }, alloc);
    //     expected9.print();
    //     C1.print();
    //     try std.testing.expectEqualDeep(expected9.data, C1.data);
    //     try std.testing.expectEqualDeep(expected9.shape.shape, C1.shape.shape);
    //     try std.testing.expectEqualDeep(expected9, C1);
    // }
}

test "NDArray.matmul with accumulation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cpu = zg.device.HostDevice.init(arena.allocator());

    const T = f64;
    const Array = NDArray(T);

    // Test: 2x2 * 2x2 with accumulation
    {
        // A = [[1, 2], [3, 4]]
        const A = try Array.init(&.{ 1, 2, 3, 4 }, &[_]usize{ 2, 2 }, cpu.reference());

        // B = [[5, 6], [7, 8]]
        const B = try Array.init(&.{ 5, 6, 7, 8 }, &[_]usize{ 2, 2 }, cpu.reference());

        // C (accum) = [[1, 1], [1, 1]]
        var C = try Array.init(&.{ 1, 1, 1, 1 }, &[_]usize{ 2, 2 }, cpu.reference());

        // Perform C = 2 * A * B + 3 * C
        try A.bmm_acc_(B, &C, cpu.reference(), .{ .alpha = 2.0, .beta = 3.0 });

        // Expected result:
        // 2 * [[1, 2], [3, 4]] * [[5, 6], [7, 8]] + 3 * [[1, 1], [1, 1]]
        // = 2 * [[19, 22], [43, 50]] + [[3, 3], [3, 3]]
        // = [[41, 47], [89, 103]]
        const expected = try Array.init(&.{ 41, 47, 89, 103 }, &[_]usize{ 2, 2 }, cpu.reference());

        try std.testing.expectEqualDeep(expected.data, C.data);
        try std.testing.expectEqualDeep(expected.shape.slice(), C.shape.slice());
    }
}

test "NDArray.matvec" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cpu = zg.device.HostDevice.init(arena.allocator());

    const T = f64;
    const Array = NDArray(T);

    // multiply a 2x2 and a 2x to get a 2x
    // [[1, 2]  * [[1]  = [[13]
    //  [0, 1]]    [6]]    [6]]
    const A1 = try Array.init(&.{ 1, 2, 0, 1 }, &.{ 2, 2 }, cpu.reference());
    const X1 = try Array.init(&.{ 1, 6 }, null, cpu.reference());
    const Y1 = try A1.matvec(X1, cpu.reference(), .{});

    const expected1 = try Array.init(&.{ 13, 6 }, null, cpu.reference());
    try std.testing.expectEqualDeep(expected1, Y1);
}

test "NDArray.sum_along" {
    const allocator = std.testing.allocator;
    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();

    const T = f64;
    const Array = NDArray(T);

    // sum a 2x2 along dim=0
    // [[3, 2],    = [7, 3]
    //  [4, 1]]
    var a1 = try Array.init(@constCast(&[_]T{ 3, 2, 4, 1 }), &[_]usize{ 2, 2 }, device);
    defer a1.deinit(device);

    var a2 = try a1.sum_along(device, .{ .dim = 0 });
    defer a2.deinit(device);

    var expected1 = try Array.init(&[_]T{ 7, 3 }, null, device);
    defer expected1.deinit(device);

    try std.testing.expectEqualDeep(expected1, a2);
    try std.testing.expect(a2.shape.equal(expected1.shape));
}

test "NDArray.max_along" {
    const allocator = std.testing.allocator;
    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();

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
    }), shape, device);
    defer array.deinit(device);

    // max over dim 0
    var result1 = try array.max_along(device, .{ .dim = 0, .keep_dims = true });
    defer result1.deinit(device);

    var expected1 = try Array.init(&[_]T{
        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24,
    }, &[_]usize{ 1, 3, 4 }, device);

    defer expected1.deinit(device);
    try std.testing.expectEqualDeep(expected1, result1);

    // max over dim 1
    var result2 = try array.max_along(device, .{ .dim = 1, .keep_dims = false });
    defer result2.deinit(device);

    var expected2 = try Array.init(&[_]T{
        9,  10, 11, 12,
        21, 22, 23, 24,
    }, &[_]usize{ 2, 4 }, device);
    defer expected2.deinit(device);

    try std.testing.expectEqualDeep(expected2, result2);

    // max over dim 2
    var result3 = try array.max_along(device, .{ .dim = 2, .keep_dims = true });
    defer result3.deinit(device);

    var expected3 = try Array.init(&[_]T{
        4,  8,  12,
        16, 20, 24,
    }, &[_]usize{ 2, 3, 1 }, device);
    defer expected3.deinit(device);
    try std.testing.expectEqualDeep(expected3, result3);
}

test "NDArray.gather" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cpu = zg.device.HostDevice.init(arena.allocator());

    const T = f32;

    // zig fmt: off
    const input_data = [_]T{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };
    const input_shape = [_]usize{ 3, 3 };
    // zig fmt: on

    var input = try NDArray(T).init(&input_data, &input_shape, cpu.reference());
    defer input.deinit(cpu.reference());

    // zig fmt: off
    const index_data = [_]usize{
        0, 1,
        1, 2,
        0, 2,
    };
    const index_shape = [_]usize{ 3, 2 };
    // zig fmt: on

    var index = try NDArray(usize).init(&index_data, &index_shape, cpu.reference());
    defer index.deinit(cpu.reference());

    var output = try input.gather(cpu.reference(), .{ .indices = index, .dim = 1, .return_offsets = true });
    defer output.deinit();

    try std.testing.expectEqualSlices(T, &[_]T{ 1, 2, 5, 6, 7, 9 }, output.values.data);
    try std.testing.expectEqualSlices(usize, &index_shape, output.values.shape.slice());

    // Case 2: out of bounds
    //try index.set(&.{ 0, 0 }, 3);
    //try std.testing.expectError(error.IndexOutOfBounds, input.gather(cpu.reference(), .{ .indices = index, .dim = 1 }));

    // // Case 3: wrong shape
    // const invalid_index_shape = [_]usize{4};
    // var invalid_index = try NDArray(usize).init(&index_data, &invalid_index_shape, alloc);
    // defer invalid_index.deinit(alloc);
    //
    // try std.testing.expectError(error.InvalidIndexShape, input.gather(invalid_index, 1, alloc));
}

//test "NDArray.slice" {
//    var cpu = zg.device.HostDevice.init(std.testing.allocator);
//    var arr = try NDArray(f32).init(&[_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, &[_]usize{ 3, 3 }, cpu.reference());
//    defer arr.deinit(cpu.reference());
//
//    // get 2x2 slice from the top-left corner
//    var slice = try arr.slice_ranges(&[_]Range{ Range{ .start = 0, .end = 2 }, Range{ .start = 0, .end = 2 } });
//
//    // mutate
//    const new_values = try NDArray(f32).init(&[_]f32{ 10, 20, 30, 40 }, &[_]usize{ 2, 2 }, cpu.reference());
//    defer new_values.deinit(cpu.reference());
//
//    try slice.set_slice_ranges(&[_]Range{ Range{ .start = 0, .end = 2 }, Range{ .start = 0, .end = 2 } }, new_values.*);
//
//    const exp = try NDArray(f32).init(&[_]f32{ 10, 20, 30, 40, 5, 6, 7, 8, 9 }, &[_]usize{ 3, 3 }, cpu.reference());
//    defer exp.deinit(cpu.reference());
//
//    try std.testing.expectEqualDeep(exp, arr);
//}

// Explicitly expands a tensor to match the shape of another.
// This operation can only expand dimensions, never contract them.
// If any dimension of self is larger than the corresponding dimension in other,
// an error.InvalidExpansion is returned.
//pub fn expand_as(self: Self, other: Self, device: DeviceReference) !Self {
//    // if shapes are identical, just return a copy.
//    if (self.shape.eq(other.shape.*, .{ .strict = true })) {
//        return self.copy(device);
//    }

//    const bshape = try self.shape.broadcast(other.shape.*);
//    defer bshape.deinit();

//    for (0..@min(self.shape.len, other.shape.len)) |i| {
//        if (self.shape.shape[i] > other.shape.shape[i]) return error.InvalidExpansion;
//    }

//    const bvalues = try Self.empty(bshape.shape, device);

//    // allocate index mappings
//    var self_indices: std.BoundedArray(usize, 8) = .{};

//    var indices: std.BoundedArray(usize, 8) = .{};
//    @memset(indices.buffer, 0);

//    // precompute which dimensions are broadcasted (i.e., dimensions of size 1).
//    var broadcasted = std.StaticBitSet(8).initEmpty();

//    for (self.shape.shape, 0..) |dim, i| {
//        broadcasted.setValue(i, dim == 1);
//    }

//    // iterate over broadcasted shape and copy values
//    while (true) {
//        // map indices to self.shape, using precomputed broadcast information
//        for (self.shape.shape, 0..) |_, i| {
//            const bshape_index = bshape.shape.len - self.shape.shape.len + i;
//            // if the dimension is broadcasted, always use 0, otherwise use the index
//            self_indices.buffer[i] = if (broadcasted.isSet(i)) 0 else indices.buffer[bshape_index];
//        }

//        const self_offset = self.pos_to_offset(self_indices.slice());
//        const bvalues_offset = bvalues.pos_to_offset(indices.slice());
//        bvalues.data[bvalues_offset] = self.data[self_offset];

//        var dim = indices.len;
//        while (dim > 0) {
//            dim -= 1;
//            indices.buffer[dim] += 1;
//            if (indices.buffer[dim] < bshape.shape[dim]) break;
//            indices.buffer[dim] = 0;
//        }
//        if (dim == 0) break;
//    }

//    return bvalues;
//}

//test expand_as {
//    const content_check = struct {
//        /// Make sure the expansion was correct
//        fn content_check(original: *NDArray(f32), expanded: *NDArray(f32)) !void {
//            if (original.shape.eq(expanded.shape.*, .{ .strict = true })) {
//                try std.testing.expectEqualSlices(f32, original.data, expanded.data);
//            }

//            const orig_shape = original.shape.shape;
//            const exp_shape = expanded.shape.shape;

//            var indices = try std.testing.allocator.alloc(usize, exp_shape.len);
//            defer std.testing.allocator.free(indices);
//            @memset(indices, 0);

//            while (true) {
//                // calculate index for expanded array
//                const exp_index = expanded.pos_to_offset(indices);

//                // calculate corresponding index for original array
//                var orig_indices = try std.testing.allocator.alloc(usize, orig_shape.len);
//                defer std.testing.allocator.free(orig_indices);
//                for (orig_shape, 0..) |dim, i| {
//                    orig_indices[i] = if (dim == 1) 0 else indices[i];
//                }
//                const orig_index = original.pos_to_offset(orig_indices);

//                try std.testing.expectApproxEqAbs(expanded.data[exp_index], original.data[orig_index], std.math.floatEps(f32));

//                // increment indices
//                var dim = exp_shape.len;
//                while (dim > 0) {
//                    dim -= 1;
//                    indices[dim] += 1;
//                    if (indices[dim] < exp_shape[dim]) break;
//                    indices[dim] = 0;
//                }
//                if (dim == 0) break;
//            }
//        }
//    }.content_check;
//    const allocator = std.testing.allocator;
//    var device = zg.device.HostDevice.init(allocator);
//    defer device.deinit();
//    const Array = NDArray(f32);

//    const test_cases = [_][3][3]usize{
//        .{ .{ 1, 3, 2 }, .{ 2, 3, 2 }, .{ 2, 3, 2 } },
//        .{ .{ 1, 3, 2 }, .{ 2, 3, 2 }, .{ 2, 3, 2 } },
//        .{ .{ 1, 3, 1 }, .{ 7, 3, 1 }, .{ 7, 3, 1 } },
//        .{ .{ 2, 3, 4 }, .{ 2, 3, 4 }, .{ 2, 3, 4 } },
//        .{ .{ 7, 7, 7 }, .{ 3, 3, 3 }, .{ 0, 0, 0 } }, // failing case
//        .{ .{ 1, 3, 3 }, .{ 3, 1, 3 }, .{ 1, 1, 1 } }, // failing case
//    };
//    var rng = std.Random.DefaultPrng.init(99);
//    var nums: [512]f32 = undefined;
//    for (&nums) |*b| b.* = rng.random().float(f32);

//    inline for (test_cases) |case| {
//        const sa = utils.prod(&case[0]);
//        const A = try Array.init(nums[0..sa], &case[0], device.reference());
//        defer A.deinit(device.reference());
//        const B = try Array.zeros(&case[1], device.reference());
//        defer B.deinit(device.reference());
//        const expt = case[2];
//        if (std.mem.allEqual(usize, &expt, 0)) {
//            try std.testing.expectError(error.Unbroadcastable, A.expand_as(B, device.reference()));
//        } else if (std.mem.allEqual(usize, &expt, @intCast(1))) {
//            try std.testing.expectError(error.InvalidExpansion, A.expand_as(B, device.reference()));
//        } else {
//            const C = try A.expand_as(B, device.reference());
//            defer C.deinit(device.reference());
//            try std.testing.expectEqualSlices(usize, &expt, C.shape.shape);
//            try content_check(A, C);
//        }
//    }

//    // const A = try Array.init(&.{1, 2, 3, 4}, &.{2, 2}, alloc);
//    // const B = try Array.init(&.{1, 2, 3, 4}, &.{2, 2}, alloc);
//    // const C = try A.expand_as(B, alloc);
//}
//

// TODO: This cannot directly touch device memory
//pub fn get(self: Self, indices: []const usize) T {
//    return self.data[self.shape.strides().pos_to_offset(indices)];
//}

// TODO: This cannot directly touch device memory
//pub fn set(self: Self, indices: []const usize, value: T) !void {
//    if (indices.len != self.shape.len) return error.InvalidShape;
//    std.debug.assert(indices.len == self.shape.len);
//    self.data[self.shape.strides().pos_to_offset(indices)] = value;
//}

// USE: Shape.strides() -> creates offset helper
//pub fn pos_to_offset(self: Self, indices: []const usize) usize {
//    if (indices.len < self.shape.len) log.warn("Hope you know what you are doing.", .{});
//    std.debug.assert(indices.len >= self.shape.len);
//    var index: usize = 0;
//    var stride: usize = 1;
//    for (0..self.shape.len) |i| {
//        const dim = self.shape.len - i - 1;
//        const dimSize = self.shape.get(dim) catch unreachable;
//        const idx = indices[dim];
//        std.debug.assert(idx < dimSize);

//        index += idx * stride;
//        stride *= dimSize;
//    }
//    return index;
//}

//pub fn offset_to_pos(self: Self, offset: usize, device: DeviceReference) ![]usize {
//    const n = self.shape.len;
//    const pos = try device.allocator.alloc(usize, n);
//    errdefer device.allocator.free(pos);

//    var remaining_index = offset;
//    for (0..n) |i| {
//        pos[i] = remaining_index / self.shape.strides[i];
//        remaining_index %= self.shape.strides[i];
//    }

//    return pos;
//}

// TODO: Check device support
// Should be analogous to: https://pytorch.org/docs/stable/generated/torch.slice_scatter.html#torch.slice_scatter
//pub fn set_slice_ranges(self: Self, ranges: []const Range, values: Self) !void {
//    const slice_ = try self.slice_ranges(ranges);
//    if (!slice_.shape.equal(values.shape)) {
//        return error.IncompatibleShapes;
//    }
//    var i: usize = 0;
//    while (i < slice_.data.len) : (i += 1) {
//        slice_.data[i] = values.data[i];
//    }
//}
