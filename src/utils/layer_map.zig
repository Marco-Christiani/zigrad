const std = @import("std");
const Allocator = std.mem.Allocator;
const NDTensor = @import("../ndtensor.zig").NDTensor;
const rtti = @import("../utils/rtti.zig");
const ClosurePointer = rtti.ClosurePointer;

const Self = @This();
const ParamMap = std.StrinngArrayHashMapUnmanaged(ClosurePointer);

const PopulateOptions = struct {
    shared: bool = true,
};

allocator: Allocator,
map: ParamMap,

pub fn init(allocator: Allocator) Self {
    return .{ .allocator = allocator, .map = .empty };
}

pub fn deinit(self: *Self) void {
    self.map.deinit(self.allocator);
    self.* = undefined;
}

pub fn put(self: *Self, key: []const u8, ptr: anytype) !void {
    try self.map.put(self.allocator, key, ClosurePointer.init(ptr));
}

pub fn extract(
    self: *Self,
    ParamType: type,
    root: []const u8,
    opts: PopulateOptions,
) ParamType {
    var tmp: ParamType = undefined;
    self.populate(&tmp, root, opts);
    return tmp;
}

pub fn extract_map(
    self: *Self,
    ParamType: type,
    prefix: []const u8,
    opts: PopulateOptions,
) !std.StringArrayHashMapUnmanaged(ParamType) {
    var strings: std.StringArrayHashMapUnmanaged(void) = .empty;
    defer strings.deinit(self.allocator);

    for (self.map.keys()) |key| {
        if (!std.mem.startsWith(u8, key, prefix)) continue;
        const end = std.mem.indexOfScalarPos(u8, key, prefix.len, '.') orelse key.len;
        try strings.put(self.allocator, key[0..end], {});
    }

    var output: std.StringArrayHashMapUnmanaged(ParamType) = .empty;
    errdefer output.deinit(self.allocator);

    for (strings.keys()) |key| {
        const param = extract(ParamType, self.map, key, opts);
        try output.put(self.allocator, key, param);
    }
    return output;
}

const PathBuffer = std.BoundedArray(u8, 1024);

pub fn populate(
    self: *Self,
    ptr: anytype,
    root: []const u8,
    opts: PopulateOptions,
) void {
    var buf = PathBuffer.fromSlice(root) catch unreachable;
    recursive_populate(ptr, &self.map, &buf, opts.shared);
}

fn recursive_populate(
    ptr: anytype,
    map: *ParamMap,
    buf: *PathBuffer,
    shared: bool,
) void {
    const T = @TypeOf(ptr.*);

    switch (@typeInfo(T)) {
        .@"struct" => |s| inline for (s.fields) |field| {
            const ext = if (buf.len > 0)
                "." ++ field.name
            else
                field.name;

            buf.appendSlice(ext) catch unreachable;
            recursive_populate(&@field(ptr.*, field.name), map, buf);
            buf.len -= ext.len;
        },
        else => {
            const entry = map.get(buf.slice()) orelse unreachable;
            ptr.* = entry.cast(T);
            if (!shared) {
                _ = map.orderedRemove(buf.slice());
            }
        },
    }
}
