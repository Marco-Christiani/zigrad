const std = @import("std");
const HostDevice = @import("host_device.zig");
const ReduceType = @import("device_common.zig").ReduceType;
const SmaxType = @import("device_common.zig").SmaxType;
const RandType = @import("device_common.zig").RandType;
const TransferDirection = @import("device_common.zig").TransferDirection;

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
        allocator: std.mem.Allocator,

        pub fn dispatch(self: Self, params: anytype) void {
            switch (self.ptrs) {
                inline else => |d| {
                    const D = std.meta.Child(@TypeOf(d));
                    const P = @TypeOf(params);
                    if (comptime !@hasDecl(D, P.__name__)) {
                        @panic("Unimplemented: " ++ @typeName(D) ++ ", " ++ P.__name__);
                    } else {
                        @field(D, P.__name__)(d, P.__type__, params);
                    }
                },
            }
        }

        pub fn mem_alloc(self: Self, T: type, n: usize) ![]T {
            return switch (self.ptrs) {
                inline else => |dev| dev.mem_alloc(T, n),
            };
        }

        pub fn mem_alloc_byte_mask(self: Self, n: usize) ![]u8 {
            return switch (self.ptrs) {
                inline else => |dev| dev.mem_alloc_byte_mask(n),
            };
        }

        pub fn mem_free(self: Self, slice: anytype) void {
            return switch (self.ptrs) {
                inline else => |dev| dev.mem_free(slice),
            };
        }

        pub fn mem_dupe(self: Self, T: type, slice: anytype) ![]T {
            return switch (self.ptrs) {
                inline else => |dev| dev.mem_dupe(T, slice),
            };
        }

        pub fn mem_scratch(self: Self, T: type, n: usize) ![]T {
            return switch (self.ptrs) {
                inline else => |dev| dev.mem_scratch(T, n),
            };
        }

        pub fn mem_fill(self: Self, T: type, slice: []T, value: T) void {
            return switch (self.ptrs) {
                inline else => |dev| dev.mem_fill(T, slice, value),
            };
        }

        pub fn mem_random(self: Self, T: type, slice: []T, op: RandType, seed: u64) void {
            return switch (self.ptrs) {
                inline else => |dev| dev.mem_random(T, slice, op, seed),
            };
        }

        pub fn mem_copy(self: Self, T: type, src: []const T, dst: []T) void {
            return switch (self.ptrs) {
                inline else => |dev| dev.mem_copy(T, src, dst),
            };
        }

        pub fn mem_sequence(self: Self, T: type, dst: []T, initial: T, step: T) void {
            return switch (self.ptrs) {
                inline else => |dev| dev.mem_sequence(T, dst, initial, step),
            };
        }

        pub fn mem_transfer(self: Self, T: type, src: []const T, dst: []T, direction: TransferDirection) void {
            return switch (self.ptrs) {
                inline else => |dev| dev.mem_transfer(T, src, dst, direction),
            };
        }

        pub fn mem_take(self: Self, T: type, src: []const T, idxs: []const usize, dst: []T) void {
            return switch (self.ptrs) {
                inline else => |dev| dev.mem_take(T, src, idxs, dst),
            };
        }

        pub fn clear_cache(self: Self) void {
            return switch (self.ptrs) {
                inline else => |dev| dev.clear_cache(),
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
