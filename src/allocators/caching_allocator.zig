const std = @import("std");
const Allocator = std.mem.Allocator;
const constants = @import("constants.zig");
const DeviceData = @import("device_data.zig").DeviceData;
const Error = @import("device_data.zig").Error;
const zg = @import("../zigrad.zig");

const logger = @import("../logging.zig").scoped(.caching_allocator);

// I understand the philosophy of passing in allocators, but this data structure
// itself doesn't benefit from having its internal state managed by yet another
// allocator. This is similar to the original "GPA" allocator in this sense.
const allocator = std.heap.smp_allocator;

pub fn CachingAllocator(DataHandler: type) type {
    return struct {
        const Self = @This();
        const BlockPool = @import("block_pool.zig").BockPool;

        const pool_threshold = zg.constants.@"1MiB";

        const SmallPool = BlockPool(DataHandler, .{
            .min_order = std.math.log2_int(usize, DataHandler.min_split_size),
            .max_order = 20, // largest split is 500kB
        });

        const LargePool = BlockPool(DataHandler, .{
            .min_order = 20, // smallest split is 1MiB
            .max_order = 32, // largest split is arbitrary
        });

        data_handler: DataHandler,
        small_pool: SmallPool,
        large_pool: LargePool,

        scratch: struct {
            ptr: [*]u8 = undefined,
            total: usize = 0,
        } = .{},

        pub const Options = struct {
            large_pool_size: ?usize = null,
            small_pool_size: ?usize = null,
        };

        // we could make this an "empty" constant, but this structure needs to be
        // deinitialized, so I'll match it with an init.
        pub fn init(data_handler: DataHandler, opts: Options) Self {
            return .{
                .data_handler = data_handler,
                .small_pool = SmallPool.init(data_handler, allocator, opts.small_pool_size),
                .large_pool = LargePool.init(data_handler, allocator, opts.large_pool_size),
            };
        }

        pub fn deinit(self: *Self) void {
            self.small_pool.deinit(self.data_handler, allocator);
            self.large_pool.deinit(self.data_handler, allocator);

            if (self.scratch.total != 0)
                self.data_handler.free(self.scratch.ptr[0..self.scratch.total]);

            self.data_handler.deinit();

            self.* = undefined;
        }

        pub fn reset(self: *Self) void {
            self.small_pool.reset();
            self.large_pool.reset();

            if (self.scratch.total != 0)
                self.data_handler.free(self.scratch.ptr[0..self.scratch.total]);

            self.scratch.total = 0;

            self.data_handler.reset();
        }

        pub fn alloc(self: *Self, T: type, len: usize) Error!DeviceData(T) {
            if (len == 0) {
                // "Nothing! It does nothing. But it does it in style!"
                //    ~Andrei Alexandrescu
                return .{ .raw = &.{}, .ctx = 0 };
            }

            const total_bytes = @sizeOf(T) * len;
            const is_small = (total_bytes <= pool_threshold);

            logger.debug("alloc - type: {s}, len: {} bytes: {}, pool: {s}", .{
                @typeName(T), len, total_bytes, if (is_small) "small" else "large",
            });

            const result = if (is_small)
                self.small_pool.alloc(T, self.data_handler, allocator, len)
            else
                self.large_pool.alloc(T, self.data_handler, allocator, len);

            return if (result) |r| r else |err| switch (err) {
                Allocator.Error.OutOfMemory => @panic("Page allocator ran out of memory"),
                else => blk: {
                    logger.err("alloc - pool: {s}, {s}", .{
                        if (is_small) "small" else "large", @errorName(err),
                    });
                    break :blk error.DeviceOOM;
                },
            };
        }

        pub fn free(self: *Self, data: anytype) void {
            if (data.raw.len == 0)
                return;

            const T = std.meta.Child(@TypeOf(data.raw));
            const total_bytes = data.raw.len * @sizeOf(T);
            const is_small = (total_bytes < pool_threshold);

            logger.debug("free - type: {s}, len: {} bytes: {}, pool: {s}", .{
                @typeName(T), data.raw.len, total_bytes, if (is_small) "small" else "large",
            });

            if (is_small)
                self.small_pool.free(data)
            else
                self.large_pool.free(data);
        }

        /// Scratch memory does not have to be freed after calling this this function,
        /// scratch is freed upon calling deinit.
        ///
        /// ## ADR
        ///  - Using anytype because this is a sub-allocator.
        pub fn alloc_scratch(self: *Self, T: type, n: usize) []T {
            if (n == 0) return &.{};

            const total: usize = @sizeOf(T) * n;
            // check if we have enough scratch to provide a payload
            if (self.scratch.total < total) {
                if (self.scratch.total != 0) {
                    logger.debug("alloc_scratch - freeing: {}", .{self.scratch.total});
                    self.data_handler.free(self.scratch.ptr[0..self.scratch.total]);
                }
                // Hard error - we cannot fail to allocate scratch memory.
                // After warmup, you'll likely have sufficient scratch.
                self.scratch.ptr = self.data_handler.alloc(total) orelse
                    @panic("Cannot allocate scratch memory");

                self.scratch.total = total;
            }

            logger.debug("alloc_scratch - type: {s}, len: {} bytes: {}", .{
                @typeName(T), n, total,
            });

            return cast_to_slice(T, self.scratch.ptr, n);
        }
    };
}

fn get_stack_slot(byte_size: usize) u6 {
    std.debug.assert(byte_size <= 128);
    const slot: u6 = @truncate(std.math.log2_int(usize, byte_size));
    return slot + @intFromBool(@popCount(byte_size) != 1);
}

fn get_slot_size(slot: u6) usize {
    return @as(usize, 1) << slot;
}

fn cast_to_slice(T: type, ptr: [*]u8, len: usize) []T {
    const tptr: [*]T = @ptrCast(@alignCast(ptr));
    return tptr[0..len];
}
