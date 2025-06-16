const std = @import("std");
const DeviceData = @import("device_data.zig").DeviceData;
const Error = @import("device_data.zig").Error;

// TODO: Optimize this.

// Note: This is a sub-allocator - it does not
// guard against 0-length slices. That is up
// to the parent allocator.

// Very simple stack-based scalar allocator. This
// is primarily here to prevent scalars from
// fragmenting caching_allocator pools.
const ScalarAllocator = @This();

const allocator = std.heap.smp_allocator;

// I'm going with 64-bit size because scalars
// are cheap and for making caching easier.
const scalar_size: usize = @sizeOf(f64);
const Stack = std.ArrayListUnmanaged([*]u8);

mem_buf: []u8,
stack: Stack,
capacity: usize,

pub fn init(capacity: usize) ScalarAllocator {
    return .{ .capacity = capacity, .stack = Stack.empty, .mem_buf = &.{} };
}

pub fn deinit(self: *ScalarAllocator, data_handler: anytype) void {
    if (self.mem_buf.len > 0)
        data_handler.unmap(self.mem_buf);

    self.stack.deinit(allocator);
    self.* = undefined;
}

// using any type because this is a sub-allocator.
pub fn alloc(self: *ScalarAllocator, data_handler: anytype, T: type) Error!DeviceData(T) {
    comptime std.debug.assert(@sizeOf(T) <= scalar_size);

    if (self.mem_buf.len > 0) {
        @branchHint(.likely);
        const ptr = (self.stack.pop() orelse data_handler.alloc(scalar_size)) orelse
            return Error.DeviceOOM;

        const tptr: [*]T = @ptrCast(@alignCast(ptr));

        // non-contextual DeviceMem allocation
        return .{ .raw = tptr[0..1], .ctx = 0 };
    } else {
        @branchHint(.cold);
        self.stack = Stack.initCapacity(allocator, self.capacity) catch
            return Error.DeviceOOM;

        self.mem_buf = data_handler.map(self.capacity * scalar_size) catch {
            self.stack.deinit(allocator);
            return Error.DeviceOOM;
        };

        var index: usize = 0;
        while (index < self.mem_buf.len) : (index += scalar_size)
            self.stack.appendAssumeCapacity(self.mem_buf.ptr + index);

        std.debug.assert(self.mem_buf.len > 0);

        return self.alloc(data_handler, T);
    }
}

pub fn free(self: *ScalarAllocator, data_handler: anytype, data: anytype) void {
    std.debug.assert(data.raw.len == 1);

    const ptr: [*]u8 = @ptrCast(@alignCast(data.raw.ptr));

    if (self.stack.items.len == self.stack.capacity)
        return data_handler.free(ptr[0..scalar_size]);

    self.stack.appendAssumeCapacity(ptr);
}

pub fn clear(self: *ScalarAllocator, data_handler: anytype) void {
    data_handler.unmap(self.mem_buf);
    const capacity = self.stack.capacity;
    self.stack.deinit(allocator);
    self.stack = .{ .capacity = capacity, .items = &.{} };
}
