//! A SIMD accelerated shape abstraction.
const std = @import("std");
const zg = @import("../zigrad.zig");

const Label = @This();
const SizeType = std.math.IntFittingRange(0, capacity);

pub const capacity: u64 = zg.settings.label_capacity;
// shapes are value 1 by default
pub const empty: Label = .{ .buffer = undefined };

buffer: [capacity]u8,
len: SizeType = 0,

pub fn init(str: []const u8) Label {
    std.debug.assert(str.len <= capacity);
    var self: Label = .empty;
    @memcpy(self.buffer[0..str.len], str);
    self.len = @intCast(str.len);
    return self;
}

pub fn slice(self: anytype) MatchedSlice(@TypeOf(&self.buffer)) {
    return self.buffer[0..self.len];
}

pub fn print(self: *Label, comptime fstr: []const u8, args: anytype) void {
    std.fmt.bufPrint(&self.buffer, fstr, args) catch @panic("Overflow for format: " ++ fstr);
}

pub fn format(
    shape: Label,
    comptime fstr: []const u8,
    _: std.fmt.FormatOptions,
    writer: anytype,
) !void {
    try writer.print(fstr, .{shape.slice()});
}

pub fn MatchedSlice(T: type) type {
    return switch (T) {
        *[capacity]u8 => []u8,
        *const [capacity]u8 => []const u8,
        else => unreachable,
    };
}
