const std = @import("std");
const BlockAllocator = @import("block_allocator.zig");
const Allocator = std.mem.Allocator;
const constants = @import("constants.zig");
const DeviceData = @import("device_data.zig").DeviceData;
const Error = @import("device_data.zig").Error;
const ScalarAllocator = @import("scalar_allocator.zig");
const ScratchAllocator = @import("scratch_allocator.zig");
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
const allocator = std.heap.page_allocator;

pub fn CachingAllocator(DataHandler: type) type {
    return struct {
        const Self = @This();

        data_handler: DataHandler,
        scalar_allocator: ScalarAllocator,
        scratch_allocator: ScratchAllocator,
        block_allocator: BlockAllocator,

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
                .scalar_allocator = ScalarAllocator.init(opts.scalar_limit),
                .scratch_allocator = .{},
                .block_allocator = BlockAllocator.init(data_handler, allocator, max_cache_size),
            };
        }

        pub fn deinit(self: *Self) void {
            self.scalar_allocator.deinit(self.data_handler);
            self.scratch_allocator.deinit(self.data_handler);
            self.block_allocator.deinit(self.data_handler, allocator);
        }

        pub fn reset(self: *Self) void {
            self.scalar_allocator.reset(self.data_handler);
            self.scratch_allocator.reset(self.data_handler);
            self.block_allocator.reset();
        }

        pub fn alloc(self: *Self, T: type, size: usize) Error!DeviceData(T) {
            return switch (size) {
                // "Nothing! It does nothing. But it does it in style!"
                //    ~Andrei Alexandrescu
                0 => .{ .raw = &.{}, .ctx = 0 },
                1 => self.scalar_allocator.alloc(self.data_handler, T),
                else => self.block_allocator.alloc(T, allocator, size) catch |err| switch (err) {
                    Allocator.Error.OutOfMemory => @panic("Page allocator ran out of memory."),
                    else => Error.DeviceOOM,
                },
            };
        }

        pub fn free(self: *Self, data: anytype) void {
            switch (data.raw.len) {
                0 => return,
                1 => self.scalar_allocator.free(self.data_handler, data),
                else => self.block_allocator.free(data),
            }
        }

        pub fn scratch(self: *Self, T: type, n: usize) []T {
            return self.scratch_allocator.alloc(self.data_handler, T, n);
        }
    };
}
