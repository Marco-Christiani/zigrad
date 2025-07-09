const std = @import("std");
const Allocator = std.mem.Allocator;
const constants = @import("constants.zig");
const DeviceData = @import("device_data.zig").DeviceData;
const Error = @import("device_data.zig").Error;
const zg = @import("../zigrad.zig");

// I understand the philosophy of passing in allocators, but this data structure
// itself doesn't benefit from having its internal state managed by yet another
// allocator. This is similar to the original "GPA" allocator in this sense.
const allocator = std.heap.smp_allocator;

pub fn CachingAllocator(DataHandler: type) type {
    return struct {
        const Self = @This();
        const BlockPool = @import("block_pool.zig").BlockPool(DataHandler);

        // support for small tensor sizes that do not work with general block
        // alignment requirements.
        // TODO: Pool this?
        const Stack = std.ArrayListUnmanaged([*]u8);

        data_handler: DataHandler,
        stacks: [8]Stack = @splat(Stack.empty),
        block_pool: BlockPool,
        scratch: struct {
            ptr: [*]u8 = undefined,
            total: usize = 0,
        } = .{},

        pub const Options = struct {
            // TODO: Come up with a better default value.
            max_cache_size: ?usize = null,
        };

        // we could make this an "empty" constant, but this structure needs to be
        // deinitialized, so I'll match it with an init.
        pub fn init(data_handler: DataHandler, opts: Options) Self {
            const max_cache_size = opts.max_cache_size orelse zg.runtime.max_cache_size;

            return .{
                .data_handler = data_handler,
                .block_pool = BlockPool.init(data_handler, allocator, max_cache_size),
            };
        }

        pub fn deinit(self: *Self) void {
            self.block_pool.deinit(self.data_handler, allocator);

            for (0..self.stacks.len) |slot| {
                const slot_size = get_slot_size(@truncate(slot));
                while (self.stacks[slot].pop()) |ptr|
                    self.data_handler.free(ptr[0..slot_size]);

                self.stacks[slot].deinit(allocator);
            }

            if (self.scratch.total != 0)
                self.data_handler.free(self.scratch.ptr[0..self.scratch.total]);

            self.data_handler.deinit();

            self.* = undefined;
        }

        pub fn reset(self: *Self) void {
            self.block_pool.reset();

            for (0..self.stacks.len) |slot| {
                const slot_size = get_slot_size(@truncate(slot));
                while (self.stacks[slot].pop()) |ptr|
                    self.data_handler.free(ptr[0..slot_size]);
            }

            if (self.scratch.total != 0)
                self.data_handler.free(self.scratch.ptr[0..self.scratch.total]);

            self.scratch.total = 0;

            self.data_handler.reset();
        }

        pub fn alloc(self: *Self, T: type, size: usize) Error!DeviceData(T) {
            if (size == 0) {
                // "Nothing! It does nothing. But it does it in style!"
                //    ~Andrei Alexandrescu
                return .{ .raw = &.{}, .ctx = 0 };
            }

            const byte_size = size * @sizeOf(T);

            if (byte_size <= 128) {
                const slot = get_stack_slot(byte_size);

                const new_size = get_slot_size(slot);

                const ptr = self.stacks[slot].pop() orelse
                    self.data_handler.alloc(new_size) orelse return Error.DeviceOOM;

                return .{ .raw = cast_to_slice(T, ptr, size), .ctx = 0 };
            }

            return self.block_pool.alloc(T, self.data_handler, allocator, size) catch |err| switch (err) {
                Allocator.Error.OutOfMemory => @panic("Page allocator ran out of memory."),
                else => Error.DeviceOOM,
            };
        }

        pub fn free(self: *Self, data: anytype) void {
            if (data.raw.len == 0)
                return;

            const byte_size = data.raw.len * @sizeOf(std.meta.Child(@TypeOf(data.raw)));

            if (byte_size <= 128) {
                const slot = get_stack_slot(byte_size);

                const ptr: [*]u8 = @ptrCast(@alignCast(data.raw.ptr));

                self.stacks[slot].append(allocator, ptr) catch {
                    const new_size = get_slot_size(slot);
                    self.data_handler.free(ptr[0..new_size]);
                };
            } else {
                self.block_pool.free(data);
            }
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
                if (self.scratch.total != 0)
                    self.data_handler.free(self.scratch.ptr[0..self.scratch.total]);

                // Hard error - we cannot fail to allocate scratch memory.
                // After warmup, you'll likely have sufficient scratch.
                self.scratch.ptr = self.data_handler.alloc(total) orelse
                    @panic("Cannot allocate scratch memory.");

                self.scratch.total = total;
            }
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
