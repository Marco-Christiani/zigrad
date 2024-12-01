const std = @import("std");
const debug = @import("builtin").mode == .Debug;
const backend = @import("root.zig").backend;

// This data structure assumes that all allocations are
// alsigned the same. This cannot be used without either
// calling aligned alloc or using something like malloc.
const Self = @This();
const Error = std.mem.Allocator.Error;

// map components //
const MapKey = usize; // use slot length as key

const PtrStack = struct {
    ptrs: []*anyopaque = &.{},
    index: usize = 0,

    pub fn push(self: *PtrStack, ptr: *anyopaque) bool {
         if (self.index < self.ptrs.len) {
             self.ptrs[self.index] = ptr;
             self.index += 1;
             return true;
         }
         return false;
    }
    pub fn pop(self: *PtrStack) ?*anyopaque {
         if (self.index > 0) {
             self.index -= 1;
             return self.ptrs[self.index];
         }
         return null;
    }

    pub fn deinit(self: *PtrStack, allocator: std.mem.Allocator) void {
        allocator.free(self.ptrs);
    }
    
    // after setup, this function should never be called again.
    pub fn growCapacity(self: *PtrStack, allocator: std.mem.Allocator, n: usize) !void {
        const cap = self.ptrs.len;
        if (allocator.resize(self.ptrs, cap + n)) {
            self.ptrs.len += n;
            return;
        }
        const ptrs = try allocator.alloc(*anyopaque, cap + n);
        @memcpy(ptrs[0..cap], self.ptrs);
        allocator.free(self.ptrs);
        self.ptrs = ptrs;
    }
};

pub const MapType = std.AutoHashMapUnmanaged(MapKey, PtrStack);

const PoppingIterator = struct {
    map_iter: MapType.ValueIterator,    
    ptr_stack: ?*PtrStack,

    pub fn init(map: *MapType) PoppingIterator {
        var map_iter = map.valueIterator();
        const ptr_stack = map_iter.next();
        return .{
            .map_iter = map_iter,
            .ptr_stack = ptr_stack,
        };
    }

    pub fn next(self: *PoppingIterator) ?*anyopaque {
        var ptr_stack = self.ptr_stack orelse return null;
        return ptr_stack.pop() orelse {
            self.ptr_stack = self.map_iter.next();
            return self.next();
        };
    }
};

/// keeps track of freed allocations
map: MapType = .{},
// This could be moved to unmanged but that starts to tangle it
// up with the porent device. We can cross that bridge later.
allocator: std.mem.Allocator,
capacity: usize = 0, 

pub fn deinit(self: *Self) void {
    var iter = self.map.valueIterator();

    while (iter.next()) |stack| {
        if (debug and stack.index != 0) {
            @panic("Unfreed cached pointers - try poppingIterator() to access and free all memory first.");
        }
        stack.deinit(self.allocator);
    }
    self.map.deinit(self.allocator);
}

pub fn reserve(self: *Self, comptime T: type, len: usize, n: usize) Error!void {
    const entry = try self.map.getOrPut(self.allocator, len * @sizeOf(T));
    if (!entry.found_existing) entry.value_ptr.* = .{};
    try entry.value_ptr.growCapacity(self.allocator, n);
    self.capacity += 1;
}

pub fn get(self: *Self, comptime T: type, len: usize) ?[]T {
    const stack = self.map.getPtr(len * @sizeOf(T)) orelse return null;
    const raw = stack.pop() orelse return null;
    const ptr: [*]T = @alignCast(@ptrCast(raw));
    return ptr[0..len];
}

pub fn put(self: *Self, comptime T: type, data: []T) bool {
    const stack = self.map.getPtr(data.len * @sizeOf(T)) orelse return false;
    return stack.push(@constCast(data.ptr));
}

pub fn poppingIterator(self: *Self) PoppingIterator {
    return PoppingIterator.init(&self.map);
}
    
