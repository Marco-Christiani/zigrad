const std = @import("std");

// This data structure only works with managed memory objects.
// In otherwords, they have their own "deinit" method that takes
// only a non-const pointer to self as an argument.
const Self = @This();

// ensure that we always return the correct opaque pointer stack
pub fn type_id(comptime T: type) usize {
    const _Impl = struct {
        const _T = T;
        var id: u8 = 0;
    };
    return @intFromPtr(&_Impl.id);
}

// need to be able to call non-const deinit on the type
pub fn is_non_const_pointer(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Pointer => |ptr| return !ptr.is_const,
        else => false,
    };
}

const Callback = *const fn (*anyopaque) void;
const Key = struct { id: usize, dims: []const usize };
const Value = struct { ptrs: std.ArrayListUnmanaged(*anyopaque), free: Callback };

const MapContext = struct {
    pub fn hash(_: MapContext, key: Key) u64 {
        return std.hash.Wyhash.hash(0, std.mem.asBytes(key.dims));
    }
    pub fn eql(_: MapContext, a: Key, b: Key) bool {
        return a.id == b.id and std.mem.eql(usize, a.dims, b.dims);
    }
};

const MapType = std.HashMapUnmanaged(Key, Value, MapContext, std.hash_map.default_max_load_percentage);

map: MapType = .{},
allocator: std.mem.Allocator,

pub fn deinit(self: *Self) void {
    var iter = self.map.iterator();
    while (iter.next()) |entry| {
        const free = entry.value_ptr.free;
        while (entry.value_ptr.ptrs.popOrNull()) |ptr| free(ptr);
        entry.value_ptr.ptrs.deinit(self.allocator);
        self.allocator.free(entry.key_ptr.dims);
    }
    self.map.deinit(self.allocator);
}

// clear the cached items but keep the keys and ptr stacks
pub fn clear(self: *Self) void {
    var iter = self.map.iterator();
    while (iter.next()) |entry| {
        const free = entry.value_ptr.free;
        while (entry.value_ptr.ptrs.popOrNull()) |ptr| free(ptr);
        entry.value_ptr.ptrs.clearRetainingCapacity();
    }
}

pub fn get(self: *Self, comptime T: type, dims: []const usize) ?*T {
    const result = self.map.getPtr(.{ .id = type_id(*T), .dims = dims }) orelse return null;
    const ptr = result.ptrs.popOrNull() orelse return null;
    return @ptrCast(@alignCast(ptr));
}

pub fn put(self: *Self, ptr: anytype, dims: []const usize) void {
    const P = @TypeOf(ptr);

    if (comptime !is_non_const_pointer(P)) {
        @compileError("Dimension map requires non-const pointer types");
    }

    const result = self.map.getOrPut(self.allocator, .{ .id = type_id(P), .dims = dims }) catch @panic("Failed to make hash entry.");

    if (!result.found_existing) {
        const callback = struct {
            pub fn free(obj: *anyopaque) void {
                const _obj: P = @ptrCast(@alignCast(obj));
                _obj.deinit();
            }
        }.free;

        result.key_ptr.dims = self.allocator.dupe(usize, dims) catch @panic("Failed to duplicate dims");

        result.value_ptr.* = .{
            .ptrs = .{},
            .free = callback,
        };
    }

    result.value_ptr.ptrs.append(self.allocator, ptr) catch @panic("Failed to append pointer.");
}

test {
    const Foo = struct {
        data: u8,
        dims: []const usize,
        pub fn deinit(_: @This()) void {}
    };
    const Bar = struct {
        data: u8,
        dims: []const usize,
        pub fn deinit(_: @This()) void {}
    };
    const Baz = struct {
        data: u8,
        dims: []const usize,
        allocator: std.mem.Allocator,
        pub fn deinit(self: *@This()) void {
            self.allocator.free(self.dims);
        }
    };
    var foo1: Foo = .{ .data = 1, .dims = &.{ 1, 2, 3 } };
    var foo2: Foo = .{ .data = 3, .dims = &.{ 5, 6 } };
    var bar1: Bar = .{ .data = 8, .dims = &.{ 1, 2, 3 } };
    var bar2: Bar = .{ .data = 9, .dims = &.{ 5, 6 } };

    var baz: Baz = .{
        .data = 0,
        .dims = try std.testing.allocator.dupe(usize, &.{ 8, 9 }),
        .allocator = std.testing.allocator,
    };

    var self: Self = .{ .allocator = std.testing.allocator };
    defer self.deinit();

    self.put(&foo1, foo1.dims);
    self.put(&foo2, foo2.dims);
    self.put(&bar1, bar1.dims);
    self.put(&bar2, bar2.dims);
    self.put(&baz, baz.dims);

    {
        const ptr = self.get(Foo, &.{ 1, 2, 3 }) orelse unreachable;
        try std.testing.expectEqual(ptr.data, foo1.data);
        ptr.data = 2;
        try std.testing.expectEqual(2, foo1.data);
    }
    {
        const ptr = self.get(Foo, &.{ 1, 2, 4 });
        try std.testing.expectEqual(null, ptr);
    }
}
