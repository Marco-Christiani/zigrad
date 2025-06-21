const std = @import("std");
const BlockPool = @import("block_pool.zig");
const Allocator = std.mem.Allocator;
const constants = @import("constants.zig");
const DeviceData = @import("device_data.zig").DeviceData;
const Error = @import("device_data.zig").Error;
const zg = @import("../zigrad.zig");

// I understand the philosophy of passing in allocators, but
// this data structure itself doesn't benefit from having its
// internal state managed by yet another allocator. This is
// similar to the original "GPA" allocator in this sense.
//
// I considered using the smp_allocator here isntead, but we
// really don't need any of its functionality. Internally,
// the Self heavily uses free lists and arenas,
// thus there's very little benefit in using something that
// ultimately uses a page allocator behind the scenes.
const allocator = std.heap.smp_allocator;

pub fn CachingAllocator(DataHandler: type) type {
    return struct {
        const Self = @This();
        const ScalarSize = @sizeOf(f64);

        data_handler: DataHandler,
        scalar_stack: std.ArrayListUnmanaged([*]u8) = .empty,
        block_pool: BlockPool,
        scratch: struct {
            ptr: [*]u8 = undefined,
            total: usize = 0,
        } = .{},

        pub const Options = struct {
            // TODO: Come up with a better default value.
            max_cache_size: ?usize = null,
            scalar_limit: usize = 256,
        };

        // we could make this an "empty" constant, but this structure
        // needs to be deinitialized, so I'll match it with an init.
        pub fn init(data_handler: DataHandler, opts: Options) Self {
            const max_cache_size = opts.max_cache_size orelse zg.runtime.max_cache_size;

            return .{
                .data_handler = data_handler,
                .block_pool = BlockPool.init(data_handler, allocator, max_cache_size),
            };
        }

        pub fn deinit(self: *Self) void {
            self.block_pool.deinit(self.data_handler, allocator);

            while (self.scalar_stack.pop()) |ptr| {
                self.data_handler.free(ptr[0..ScalarSize]);
            }
            self.scalar_stack.deinit(allocator);

            if (self.scratch.total != 0) {
                self.data_handler.free(self.scratch.ptr[0..self.scratch.total]);
            }
            self.* = undefined;
        }

        pub fn reset(self: *Self) void {
            self.block_pool.reset();

            while (self.scalar_stack.pop()) |ptr|
                self.data_handler.free(ptr[0..ScalarSize]);

            if (self.scratch.total != 0)
                self.data_handler.free(self.scratch.ptr[0..self.scratch.total]);

            self.scratch.total = 0;
        }

        pub fn alloc(self: *Self, T: type, size: usize) Error!DeviceData(T) {
            return switch (size) {
                // "Nothing! It does nothing. But it does it in style!"
                //    ~Andrei Alexandrescu
                0 => .{ .raw = &.{}, .ctx = 0 },
                1 => {
                    const ptr = self.scalar_stack.pop() orelse
                        self.data_handler.alloc(ScalarSize) orelse
                        return Error.DeviceOOM;

                    const tptr: [*]T = @ptrCast(@alignCast(ptr));
                    return .{ .raw = tptr[0..1], .ctx = 0 };
                },
                else => self.block_pool.alloc(T, allocator, size) catch |err| switch (err) {
                    Allocator.Error.OutOfMemory => @panic("Page allocator ran out of memory."),
                    else => Error.DeviceOOM,
                },
            };
        }

        pub fn free(self: *Self, data: anytype) void {
            switch (data.raw.len) {
                0 => return,
                1 => {
                    const ptr: [*]u8 = @ptrCast(@alignCast(data.raw.ptr));
                    self.scalar_stack.append(allocator, ptr) catch
                        self.data_handler.free(ptr[0..ScalarSize]);
                },
                else => self.block_pool.free(data),
            }
        }

        /// Scratch memory does not have to be freed after calling this
        /// this function Instead, scratch is freed upon calling deinit.
        /// Also, using anytype because this is a sub-allocator.
        pub fn alloc_scratch(self: *Self, T: type, n: usize) []T {
            if (n == 0) return &.{};

            const total: usize = @sizeOf(T) * n;
            // check if we have enough scratch to provide a payload
            if (self.scratch.total < total) {
                if (self.scratch.total != 0)
                    self.data_handler.free(self.scratch.ptr[0..self.scratch.total]);

                // Hard error - we cannot fail to allocate scratch memory.
                // After warmup, you'll likely have sufficient scratch.
                self.scratch.ptr = self.data_handler.alloc(total) orelse
                    @panic("Cannot allocate scratch memory.");

                self.scratch.total = total;
            }
            const tptr: [*]T = @ptrCast(@alignCast(self.scratch.ptr));
            return tptr[0..n];
        }
    };
}
