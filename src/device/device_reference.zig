const std = @import("std");
const HostDevice = @import("host_device.zig").HostDevice;
const ReduceType = @import("device_common.zig").ReduceType;
const SmaxType = @import("device_common.zig").SmaxType;

pub fn Blas(comptime Parent: type) type {
    return struct {
        const Self = @This();

        /// Computes dot product assuming a stride of 1 and row-major. (N,) x (N,) = (1,)
        pub fn dot(
            self: *const Blas,
            T: type,
            x: []const T,
            y: []const T,
            z: []T,
        ) void {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.dot(T, x, y, z),
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

        pub fn maxForward(
            self: *const Self,
            T: type,
            src: []const T,
            dst: []T,
            idx: *i32,
        ) void {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.maxForward(T, self.cublas(), src.ptr, dst.ptr, idx),
            };
        }

        pub fn maxReverse(
            self: *const Blas,
            T: type,
            y_grd: []const T,
            x_grd: []T,
            idx: *i32,
        ) void {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.maxReverse(T, self.cublas(), y_grd.ptr, x_grd.ptr, idx),
            };
        }

        pub fn sum(
            self: *const Self,
            T: type,
            x: []const T,
        ) T {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.sum(T, x),
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
            alpha: T,
        ) void {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.axpy(T, x, y, alpha),
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

        pub fn reluForward(self: *const Self, comptime T: type, x: []const T, y: []T) void {
            return switch (self.parent()) {
                inline else => |dev| dev.nn.reluForward(T, x, y),
            };
        }

        pub fn reluBackward(self: *const Self, comptime T: type, x: []const T, y_grd: []const T, x_grd: []T) void {
            return switch (self.parent()) {
                inline else => |dev| dev.nn.reluBackward(T, x, y_grd, x_grd),
            };
        }

        pub fn smaxVecForward(self: *const Self, comptime T: type, x: []const T, y: []T, op: SmaxType) void {
            return switch (self.parent()) {
                inline else => |dev| dev.nn.smaxVecForward(T, x, y, op),
            };
        }

        pub fn smaxVecBackward(self: *const Self, comptime T: type, y_val: []const T, y_grd: []const T, x_grd: []T, op: SmaxType) void {
            return switch (self.parent()) {
                inline else => |dev| dev.nn.smaxVecBackward(T, y_val, y_grd, x_grd, op),
            };
        }

        pub fn smaxRowForward(self: *const Self, comptime T: type, X: []const T, Y: []T, m: usize, n: usize, op: SmaxType) void {
            return switch (self.parent()) {
                inline else => |dev| dev.nn.smaxRowForward(T, X, Y, m, n, op),
            };
        }

        pub fn smaxRowBackward(self: *const Self, comptime T: type, Y_val: []const T, Y_grd: []const T, X_grd: []T, m: usize, n: usize, op: SmaxType) void {
            return switch (self.parent()) {
                inline else => |dev| dev.nn.smaxRowForward(T, Y_val, Y_grd, X_grd, m, n, op),
            };
        }

        pub fn nllLoss1DIndexForward(self: *const Self, comptime T: type, src: []T, trg: usize, dst: []T, input_logits: bool, reduce: bool, reduce_type: ReduceType) f64 {
            return switch (self.parent()) {
                inline else => |dev| dev.nn.nllLoss1DIndexForward(T, src, trg, dst.ptr, input_logits, reduce, reduce_type),
            };
        }

        pub fn nllLoss1DIndexBackward(self: *const Self, comptime T: type, src_val: []const T, src_grd: []T, trg: usize, reduce_type: ReduceType) f64 {
            return switch (self.parent()) {
                inline else => |dev| dev.nn.nllLoss1DIndexForward(T, src_val, src_grd, trg, reduce_type),
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
        allocator: std.mem.Allocator,

        pub fn memAlloc(self: Self, comptime T: type, n: usize) Error![]T {
            return switch (self.ptrs) {
                inline else => |dev| dev.memAlloc(T, n),
            };
        }

        pub fn memFree(self: Self, slice: anytype) void {
            return switch (self.ptrs) {
                inline else => |dev| dev.memFree(slice),
            };
        }

        pub fn memCreate(self: Self, comptime T: type) Error!*T {
            return switch (self.ptrs) {
                inline else => |dev| dev.memCreate(T),
            };
        }

        pub fn memDestroy(self: Self, ptr: anytype) void {
            return switch (self.ptrs) {
                inline else => |dev| dev.memFree(ptr),
            };
        }

        pub fn memDupe(self: Self, T: type, slice: anytype) Error![]T {
            return switch (self.ptrs) {
                inline else => |dev| dev.memDupe(T, slice),
            };
        }

        pub fn memFill(self: Self, comptime T: type, slice: []T, value: T) void {
            return switch (self.ptrs) {
                inline else => |dev| dev.memFill(T, slice, value),
            };
        }

        pub fn memTransfer(
            self: Self,
            comptime T: type,
            src: []const T,
            dst: []T,
            direction: AuxDevice.Direction,
        ) void {
            return switch (self.ptrs) {
                .aux => |dev| dev.memTransfer(T, src, dst, direction),
                .host => @memcpy(dst, src),
            };
        }

        pub fn sync(self: Self) void {
            return switch (self.ptrs) {
                .aux => |dev| dev.sync(),
                .host => {},
            };
        }

        pub fn isCompatible(self: DeviceReference, other: DeviceReference) bool {
            if (std.meta.activeTag(self.ptrs) != std.meta.activeTag(other.ptrs)) {
                return false;
            }
            return switch (self.ptrs) {
                .aux => self.ptrs.aux.isCompatible(other.ptrs.aux),
                .host => true,
            };
        }

        pub fn isHost(self: DeviceReference) bool {
            return self.ptrs == .HOST;
        }
    };
}
