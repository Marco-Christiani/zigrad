const std = @import("std");
const HostDevice = @import("host_device.zig").HostDevice;

pub fn Blas(comptime Parent: type) type {
    return struct {
        const Self = @This();

        /// Computes dot product assuming a stride of 1 and row-major. (N,) x (N,) = (1,)
        pub fn dot(self: *const Self, T: type, x: []const T, y: []const T) T {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.dot(T, x, y),
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
            x: []T,
        ) T {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.nrm2(T, x),
            };
        }

        pub fn max(
            self: *const Self,
            T: type,
            x: []const T,
        ) T {
            return switch (self.parent()) {
                inline else => |dev| dev.blas.max(T, x),
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

pub fn DeviceReference(comptime AuxDevice: type) type {
    return struct {
        const Self = @This();

        pub const Error = std.mem.Allocator.Error;

        pub const DevicePtrs = union(enum) {
            host: *HostDevice,
            aux: *AuxDevice,
        };

        ptrs: DevicePtrs,
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
    };
}
