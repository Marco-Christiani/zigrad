const std = @import("std");
const HostDevice = @import("host_device.zig").HostDevice;
const ReduceType = @import("device_common.zig").ReduceType;
const SmaxType = @import("device_common.zig").SmaxType;
const RandType = @import("device_common.zig").RandType;
const DimensionMap = @import("dimension_map.zig");

pub fn Blas(comptime Parent: type) type {
    return struct {
        const Self = @This();

        /// Computes dot product assuming a stride of 1 and row-major. (N,) x (N,) = (1,)
        pub fn dot(
            self: *const Self,
            T: type,
            x: []const T,
            y: []const T,
            z: []T,
        ) void {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.dot(T, x, y, z),
            };
        }

        pub fn add(
            self: *const Self,
            T: type,
            x: []const T,
            y: []const T,
            z: []T,
        ) void {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.add(T, x, y, z),
            };
        }

        pub fn sub(
            self: *const Self,
            T: type,
            x: []const T,
            y: []const T,
            z: []T,
        ) void {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.sub(T, x, y, z),
            };
        }

        pub fn mul(
            self: *const Self,
            T: type,
            x: []const T,
            y: []const T,
            z: []T,
        ) void {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.mul(T, x, y, z),
            };
        }

        pub fn div(
            self: *const Self,
            T: type,
            x: []const T,
            y: []const T,
            z: []T,
        ) void {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.div(T, x, y, z),
            };
        }

        /// Computes mat-vec assuming a stride of 1 for the vec and row-major.
        /// a * (M, N) x (N,) + b * (N,) = (M,)
        /// Y = aAX + bY
        pub fn matvec(
            self: *const Self,
            T: type,
            A: []const T,
            x: []const T,
            y: []T,
            M: usize,
            N: usize,
            trans_a: bool,
            alpha: T,
            beta: T,
        ) void {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.matvec(T, A, x, y, M, N, trans_a, alpha, beta),
            };
        }

        ///  Assumes row-major.
        ///  (M, K) x (K, N) = (M, N)
        /// C := alpha*op(A)*op(B) + beta*C
        pub fn matmul(
            self: *const Self,
            T: type,
            A: []const T,
            B: []const T,
            C: []T,
            M: usize,
            N: usize,
            K: usize,
            trans_a: bool,
            trans_b: bool,
            lda: usize,
            ldb: usize,
            ldc: usize,
            alpha: T,
            beta: T,
        ) void {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.matmul(T, A, B, C, M, N, K, trans_a, trans_b, lda, ldb, ldc, alpha, beta),
            };
        }

        pub fn bmm_acc(
            self: *const Self,
            T: type,
            A: []const T,
            A_sizes: []const usize,
            B: []const T,
            B_sizes: []const usize,
            C: []T,
            C_sizes: []const usize,
            trans_a: bool,
            trans_b: bool,
            lda: usize,
            ldb: usize,
            ldc: usize,
            alpha: T,
            beta: T,
        ) void {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.bmm_acc(T, A, A_sizes, B, B_sizes, C, C_sizes, trans_a, trans_b, lda, ldb, ldc, alpha, beta),
            };
        }
        /// Outer product: A = alpha(xy') + A
        /// A: (M, N)
        pub fn outer(
            self: *const Self,
            T: type,
            x: []const T,
            y: []const T,
            A: []T,
            alpha: T,
        ) void {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.outer(T, x, y, A, alpha),
            };
        }

        pub fn nrm2(
            self: *const Self,
            T: type,
            x: []const T,
            y: []T,
        ) void {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.nrm2(T, x, y),
            };
        }

        pub fn max_forward(
            self: *const Self,
            T: type,
            src: []const T,
            dst: []T,
            idx: *i32,
        ) void {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.max_forward(T, self.cublas(), src.ptr, dst.ptr, idx),
            };
        }

        pub fn max_reverse(
            self: *const Self,
            T: type,
            y_grd: []const T,
            x_grd: []T,
            idx: *i32,
        ) void {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.max_reverse(T, self.cublas(), y_grd.ptr, x_grd.ptr, idx),
            };
        }

        pub fn sum(
            self: *const Self,
            T: type,
            x: []const T,
            y: []T,
        ) void {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.sum(T, x, y),
            };
        }

        pub fn sum_along(
            self: *const Self,
            T: type,
            src_vals: []const T,
            src_sizes: []const usize,
            dst_vals: []T,
            rdx_idx: usize,
        ) void {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.sum_along(T, src_vals, src_sizes, dst_vals, rdx_idx),
            };
        }

        pub fn max_along(
            self: *const Self,
            T: type,
            src_vals: []const T,
            src_sizes: []const usize,
            dst_vals: []T,
            rdx_idx: usize,
        ) void {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.max_along(T, src_vals, src_sizes, dst_vals, rdx_idx),
            };
        }

        pub fn scale(
            self: *const Self,
            T: type,
            x: []T,
            alpha: T,
        ) void {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.scale(T, x, alpha),
            };
        }

        pub fn axpy(
            self: *const Self,
            comptime T: type,
            x: []const T,
            y: []T,
            alpha: *const T,
        ) void {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.axpy(T, x, y, alpha),
            };
        }

        pub fn clip_norm(
            self: *const Self,
            comptime T: type,
            x_val: []T,
            max_norm: T,
            delta: T,
        ) void {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.clip_norm(T, x_val, max_norm, delta),
            };
        }

        fn parent(self: *const Self) Parent.DevicePtrs {
            return @as(*const Parent, @alignCast(@fieldParentPtr("blas", self))).ptrs;
        }
    };
}

pub fn NN(comptime Parent: type) type {
    return struct {
        const Self = @This();

        pub fn relu_forward(self: *const Self, comptime T: type, x: []const T, y: []T) void {
            return switch (self.parent()) {
                inline else => |dev| dev.nn.relu_forward(T, x, y),
            };
        }

        pub fn relu_backward(self: *const Self, comptime T: type, x: []const T, y_grd: []const T, x_grd: []T) void {
            return switch (self.parent()) {
                inline else => |dev| dev.nn.relu_backward(T, x, y_grd, x_grd),
            };
        }

        pub fn smax_vec_forward(self: *const Self, comptime T: type, x: []const T, y: []T, op: SmaxType) void {
            return switch (self.parent()) {
                inline else => |dev| dev.nn.smax_vec_forward(T, x, y, op),
            };
        }

        pub fn smax_vec_backward(self: *const Self, comptime T: type, y_val: []const T, y_grd: []const T, x_grd: []T, op: SmaxType) void {
            return switch (self.parent()) {
                inline else => |dev| dev.nn.smax_vec_backward(T, y_val, y_grd, x_grd, op),
            };
        }

        pub fn smax_row_forward(self: *const Self, comptime T: type, X: []const T, Y: []T, m: usize, n: usize, op: SmaxType) void {
            return switch (self.parent()) {
                inline else => |dev| dev.nn.smax_row_forward(T, X, Y, m, n, op),
            };
        }

        pub fn smax_row_backward(self: *const Self, comptime T: type, Y_val: []const T, Y_grd: []const T, X_grd: []T, m: usize, n: usize, op: SmaxType) void {
            return switch (self.parent()) {
                inline else => |dev| dev.nn.smax_row_forward(T, Y_val, Y_grd, X_grd, m, n, op),
            };
        }

        pub fn nll_loss_1d_index_forward(self: *const Self, comptime T: type, src: []T, trg: usize, dst: []T, input_logits: bool, reduce: bool, reduce_type: ReduceType) f64 {
            return switch (self.parent()) {
                inline else => |dev| dev.nn.nll_loss_1d_index_forward(T, src, trg, dst.ptr, input_logits, reduce, reduce_type),
            };
        }

        pub fn nll_loss_1d_index_backward(self: *const Self, comptime T: type, src_val: []const T, src_grd: []T, trg: usize, reduce_type: ReduceType) f64 {
            return switch (self.parent()) {
                inline else => |dev| dev.nn.nll_loss_1d_index_forward(T, src_val, src_grd, trg, reduce_type),
            };
        }

        fn parent(self: *const Self) Parent.DevicePtrs {
            return @as(*const Parent, @alignCast(@fieldParentPtr("nn", self))).ptrs;
        }
    };
}

pub fn DeviceReference(comptime AuxDevice: type) type {
    return struct {
        const Self = @This();

        pub const HostType = HostDevice;
        pub const AuxType = AuxDevice;

        pub const Error = std.mem.Allocator.Error;

        pub const DevicePtrs = union(enum) {
            host: *HostDevice,
            aux: *AuxDevice,
        };

        ptrs: DevicePtrs,
        nn: NN(Self) = .{},
        blas: Blas(Self) = .{},
        cache: *DimensionMap,
        allocator: std.mem.Allocator,

        pub fn mem_alloc(self: Self, comptime T: type, n: usize) Error![]T {
            return switch (self.ptrs) {
                inline else => |dev| dev.mem_alloc(T, n),
            };
        }

        pub fn mem_free(self: Self, slice: anytype) void {
            return switch (self.ptrs) {
                inline else => |dev| dev.mem_free(slice),
            };
        }

        pub fn mem_create(self: Self, comptime T: type) Error!*T {
            return switch (self.ptrs) {
                inline else => |dev| dev.mem_create(T),
            };
        }

        pub fn mem_destroy(self: Self, ptr: anytype) void {
            return switch (self.ptrs) {
                inline else => |dev| dev.mem_destroy(ptr),
            };
        }

        pub fn mem_dupe(self: Self, T: type, slice: anytype) Error![]T {
            return switch (self.ptrs) {
                inline else => |dev| dev.mem_dupe(T, slice),
            };
        }

        pub fn mem_fill(self: Self, comptime T: type, slice: []T, value: T) void {
            return switch (self.ptrs) {
                inline else => |dev| dev.mem_fill(T, slice, value),
            };
        }

        pub fn mem_random(self: Self, comptime T: type, slice: []T, op: RandType, seed: u64) void {
            return switch (self.ptrs) {
                inline else => |dev| dev.mem_random(T, slice, op, seed),
            };
        }

        pub fn mem_copy(self: Self, comptime T: type, src: []const T, dst: []T) void {
            return switch (self.ptrs) {
                inline else => |dev| dev.mem_copy(T, src, dst),
            };
        }

        pub fn mem_transfer(
            self: Self,
            comptime T: type,
            src: []const T,
            dst: []T,
            direction: AuxDevice.Direction,
        ) void {
            return switch (self.ptrs) {
                .aux => |dev| dev.mem_transfer(T, src, dst, direction),
                .host => @memcpy(dst, src),
            };
        }

        pub fn sync(self: Self) void {
            return switch (self.ptrs) {
                .aux => |dev| dev.sync(),
                .host => {},
            };
        }

        pub fn is_compatible(self: Self, other: Self) bool {
            if (std.meta.activeTag(self.ptrs) != std.meta.activeTag(other.ptrs)) {
                return false;
            }
            return switch (self.ptrs) {
                .aux => self.ptrs.aux.is_compatible(other.ptrs.aux),
                .host => true,
            };
        }

        pub fn is_host(self: Self) bool {
            return self.ptrs == .host;
        }
    };
}
