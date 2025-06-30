//! A SIMD accelerated shape abstraction.
const std = @import("std");

const Shape = @This();
const SimdType = @Vector(capacity, u64);
const SizeType = std.math.IntFittingRange(0, capacity);
const ReduceOp = std.builtin.ReduceOp;
const Slice = []const u64;

pub const capacity: u64 = 8;
// shapes are value 1 by default
pub const empty: Shape = .{ .buffer = .{1} ** capacity };

// These classes are here to ensure that the correct
// values are initialized througout. Strides and Shape
// are 1 initialized and Indices are 0 initialized.
// This ensures that their linear decompositions
// and combinations create the correct outputs for
// calculating index offsets.

pub const Indices = struct {
    pub const empty: Indices = .{ .buffer = .{0} ** capacity };
    buffer: [capacity]u64,
    len: SizeType = 0,
    pub fn set(self: *Indices, i: u64, value: u64) void {
        self.buffer[i] = value;
    }
    pub fn get(self: Indices, i: u64) u64 {
        return self.buffer[i];
    }
    pub fn slice(self: anytype) MatchedSlice(@TypeOf(&self.buffer)) {
        return self.buffer[0..self.len];
    }
    pub inline fn simd(self: Indices) SimdType {
        return self.buffer;
    }
};

fn decay(T: type) type {
    return switch (@typeInfo(T)) {
        .pointer => |ptr| ptr.child,
        else => T,
    };
}

fn is_tuple(T: type) bool {
    return switch (@typeInfo(T)) {
        .@"struct" => |s| s.is_tuple,
        else => false,
    };
}

fn get_len(T: type) u64 {
    return switch (@typeInfo(T)) {
        .@"struct" => |s| {
            if (T == Indices or T == Shape) {
                return capacity;
            }
            return s.fields.len;
        },
        .array => |arr| arr.len,
        .pointer => |ptr| switch (ptr.size) {
            .slice => {
                // For slices, we can't determine length at compile time
                // This should be handled differently in pos_to_offset
                @compileError("get_len called with slice type - use runtime length instead");
            },
            else => @compileError("Unsupported pointer type for get_len"),
        },
        else => @compileError("Unsupported type for get_len: " ++ @typeName(T)),
    };
}

pub const Strides = struct {
    buffer: [capacity]u64,
    len: SizeType,
    // calculate inner-product between indices and strides to create offset
    pub inline fn pos_to_offset(self: Strides, indices: anytype) u64 {
        const T = decay(@TypeOf(indices));
        const N: u64 = comptime get_len(T);
        const V = @Vector(N, u64);
        std.debug.print("N: {d}\n", .{N});

        if (comptime @typeInfo(decay(T)) == .@"struct") {
            if (@hasField(T, "buffer")) {
                return @reduce(ReduceOp.Add, @as(V, indices.buffer[0..N].*) * self.simd(N));
            }
            if (comptime is_tuple(T)) {
                return @reduce(ReduceOp.Add, @as(V, indices) * self.simd(N));
            }
        }
        return @reduce(ReduceOp.Add, @as(V, indices[0..N].*) * self.simd(N));
    }
    pub inline fn offset_to_pos(self: Strides, offset: u64) Indices {
        var pos = Indices.empty;
        pos.len = self.len;
        var remaining_index = offset;
        for (0..self.len) |i| {
            pos.set(i, remaining_index / self.buffer[i]);
            remaining_index %= self.buffer[i];
        }
        return pos;
    }

    pub fn slice(self: anytype) MatchedSlice(@TypeOf(&self.buffer)) {
        return self.buffer[0..self.len];
    }
    pub inline fn simd(self: Strides, comptime N: usize) @Vector(N, u64) {
        return self.buffer[0..N].*;
    }
    pub fn get(self: Strides, i: u64) u64 {
        return self.buffer[i];
    }
};

// fn offset_with_coord_replaced(self: Shape, val: u64, dim: u64) u64 {
//     std.debug.print("strides: {d}\n", .{self.strides().slice()});
//     // const s = self.strides();
//     var tmp = self;
//     std.debug.assert(dim < tmp.len);
//     tmp.set(dim, val);
//     std.debug.print("strides: {d}\n", .{self.strides().slice()});
//     std.debug.print("tmp: {d}\n", .{tmp});
//     const result = self.strides().pos_to_offset(tmp);
//     std.debug.assert(result <= self.size());
//     return result;
// }
// pub fn main() !void {
//     const shape = Shape.init(&.{ 5, 4, 3, 2 });
//     const strides_ = shape.strides();
//     const offset = strides_.pos_to_offset(.{ 4, 3, 2, 1 });
//     const pos = strides_.offset_to_pos(offset);

//     std.debug.print("shape: {d}\n", .{shape.slice()});
//     std.debug.print("strides: {d}\n", .{strides_.slice()});
//     std.debug.print("offset: {d}\n", .{offset});
//     std.debug.print("pos: {d}\n", .{pos.slice()});
//     std.debug.assert(pos.get(0) == 4);
//     std.debug.assert(pos.get(1) == 3);
//     std.debug.assert(pos.get(2) == 2);
//     std.debug.assert(pos.get(3) == 1);

//     // Test offset with coordinate replaced
//     // const new_offset = shape.offset_with_coord_replaced(0, 0);
//     // const tgt_coord = [_]usize{ 0, 4, 3, 2 };
//     // std.debug.print("new_offset: {d}\n", .{new_offset});
//     // std.debug.print("tgt_coord: {d}\n", .{tgt_coord});
//     // std.debug.print("{d}\n", .{shape.strides().offset_to_pos(new_offset).slice()});
// }

buffer: [8]u64,
len: SizeType = 0,

pub fn init(values: Slice) Shape {
    return Shape.empty.overlay(values);
}

pub fn merge(slices: []const Slice) Shape {
    var tmp = Shape.empty;
    for (slices) |s| {
        @memcpy(tmp.buffer[tmp.len..][0..s.len], s);
        tmp.len += @intCast(s.len);
    }
    return tmp;
}

pub fn overlay(shape: Shape, values: Slice) Shape {
    std.debug.assert(values.len <= capacity);
    var tmp = shape;
    tmp.len = @intCast(values.len);
    @memcpy(tmp.slice(), values);
    return tmp;
}

// TODO: pass by value? check assembly
pub fn simd(self: Shape) SimdType {
    return self.buffer[0..capacity].*;
}

pub fn set(self: *Shape, i: u64, value: u64) void {
    self.buffer[i] = value;
}

pub fn get(self: Shape, i: u64) u64 {
    return self.buffer[i];
}

pub fn last(self: Shape) u64 {
    std.debug.assert(self.len > 1);
    return self.buffer[self.len - 1];
}

pub fn remove(self: *Shape, i: u64) void {
    std.debug.assert(i < self.len);
    std.mem.copyForwards(u64, self.crop(i, 1), self.crop(i + 1, 0));
    self.set(self.len - 1, 1);
    self.len -= 1;
}

pub fn insert(self: *Shape, i: u64, item: u64) error{Overflow}!void {
    std.debug.assert(i > self.len);
    std.debug.assert(self.len < self.capacity);
    self.len += 1;
    var s = self.slice();
    std.mem.copyBackwards(u64, s[i + 1 .. s.len], s[i .. s.len - 1]);
    self.buffer[i] = item;
}

pub fn append(self: *Shape, value: u64) void {
    std.debug.assert(self.len < capacity);
    self.set(self.len, value);
    self.len += 1;
}

pub fn count(self: Shape, value: u64) u64 {
    return std.simd.countElementsWithValue(self.simd(), value);
}

pub fn equal(a: Shape, b: Shape) bool {
    return @reduce(ReduceOp.And, a.simd() == b.simd());
}

// I'd like a better name for this. Compatible for what, exactly?
// It's definitely not "equal", but we're seeing if we disagree
// at very specific moments
pub fn compatible(a: Shape, b: Shape) bool {
    const dims = @max(a.len, b.len);
    for (0..dims) |i| {
        const dim_a = if (i < a.len) a.get(a.len - 1 - i) else 1;
        const dim_b = if (i < b.len) b.get(b.len - 1 - i) else 1;
        if (dim_a != dim_b and dim_a != 1 and dim_b != 1) return false;
    }
    return true;
}

pub fn unsqueeze(self: Shape) Shape {
    std.debug.assert(self.len < capacity);
    var tmp = Shape.empty;
    tmp.len = self.len + 1;
    @memcpy(tmp.crop(1, 0), self.slice());
    return tmp;
}

pub fn _unsqueeze(self: *Shape) void {
    self.* = self.unsqueeze();
}

pub fn broadcast(self: Shape, other: Shape) error{Unbroadcastable}!Shape {
    if (self.equal(other)) {
        return self;
    }
    const dims = @max(self.len, other.len);
    var result = Shape.empty;
    result.len = dims;
    for (0..dims) |i| {
        const dim_a = if (i < self.len) self.get(self.len - 1 - i) else 1;
        const dim_b = if (i < other.len) other.get(other.len - 1 - i) else 1;

        if (dim_a != dim_b and dim_a != 1 and dim_b != 1) {
            return error.Unbroadcastable;
        }
        result.set(dims - 1 - i, @max(dim_a, dim_b));
    }
    return result;
}

pub fn realdims(self: Shape) u64 {
    return std.simd.countTrues(self.simd() != @as(SimdType, @splat(1)));
}

pub fn size(self: Shape) u64 {
    return @reduce(ReduceOp.Mul, self.simd());
}

pub fn squeeze(self: Shape) Shape {
    var tmp = self;
    // preserve the case of a single scalar
    if (self.len > 0 and self.size() == 1) {
        tmp.len = 1;
        return self;
    }
    var j: usize = 0;
    for (0..self.len) |i| {
        if (self.get(i) != 1) {
            tmp.set(j, self.get(i));
            j += 1;
        }
    }
    tmp.len = @intCast(j);
    return tmp;
}

pub fn _squeeze(self: *Shape) void {
    self.* = self.squeeze();
}

pub fn slice(self: anytype) MatchedSlice(@TypeOf(&self.buffer)) {
    return self.buffer[0..self.len];
}

pub fn head(self: anytype, amount: u64) MatchedSlice(@TypeOf(&self.buffer)) {
    return self.buffer[0..amount];
}

pub fn tail(self: anytype, amount: u64) MatchedSlice(@TypeOf(&self.buffer)) {
    return self.buffer[self.len - amount .. self.len];
}

/// Crop `shape[lhs:-rhs]`
pub fn crop(self: anytype, lhs: u64, rhs: u64) MatchedSlice(@TypeOf(&self.buffer)) {
    return if (lhs < self.len and rhs < self.len)
        self.buffer[lhs..@max(lhs, self.len - rhs)]
    else
        &.{};
}

/// This is an aligned mismatch - useful for comparing likewise
/// modes. Consider the following cases:
///
/// These two shapes are equivalent for expressing the same tensor, but naive
/// mismatch will say that index 0 is different.
///
///      s1: (5,5,5), s2: (1,5,5,5) -> null
///
/// These two shapes do not express the same tensor, but naive mismatch would
/// assume they the same because every element of s1 matches indicially to s2.
///
///      s1: (5,5,5,6), s2: (5,5,5) -> { a_pos = 3, b_pos = 2 };
///
/// To access the Shape value, use the following:
///
///    const mm = Shape.modal_mismatch(a,b) orelse .. // same as a.modal_mismatch(b)
///
///    _ = a.get(mm.a_pos);
///    _ = b.get(mm.b_pos);
///
///    if (mm.a_pos != mm.b_pos) ... // check if a realignment occured
///
pub fn mismatch(a: Shape, b: Shape) ?struct {
    a_pos: SizeType,
    b_pos: SizeType,
} {
    const a_shift: SizeType = if (a.len > b.len) a.len - b.len else 0;
    const b_shift: SizeType = if (b.len > a.len) b.len - a.len else 0;

    const index: SizeType = for (a.crop(a_shift, 0), b.crop(b_shift, 0), 0..) |m, n, i| {
        if (m != n) break @intCast(i);
    } else return null;

    return .{
        .a_pos = a_shift + index,
        .b_pos = b_shift + index,
    };
}

// this is fairly helpful for shape slices so I'm putting it here
pub fn slice_size(sizes: []const usize) usize {
    var n: usize = 1;
    return for (sizes) |m| {
        n *= m;
    } else n;
}

fn _suffix_scan(self: Shape) SimdType {
    // zig fmt: off
    return std.simd.reverseOrder(
        std.simd.prefixScan(ReduceOp.Mul, 1,
            std.simd.shiftElementsRight(
                std.simd.reverseOrder(self.simd()), 1, 1
        )));
    // zig fmt: on
}

pub fn strides(self: Shape) Strides {
    return Strides{
        .buffer = switch (self.len) {
            0 => @panic("Cannot calculate strides on 0 length shape"),
            1 => empty.overlay(&.{1}).simd(),
            2 => empty.overlay(&.{ self.last(), 1 }).simd(),
            3...8 => _suffix_scan(self),
            else => @panic("Unsupported length for tensor shape"),
        },
        .len = self.len,
    };
}

pub fn format(
    shape: Shape,
    comptime _: []const u8,
    _: std.fmt.FormatOptions,
    writer: anytype,
) !void {
    try writer.print("{d}", .{shape.slice()});
}

pub fn MatchedSlice(T: type) type {
    return switch (T) {
        *[capacity]u64 => []u64,
        *const [capacity]u64 => []const u64,
        else => unreachable,
    };
}

test "remove" {
    var shape = Shape.init(&.{ 5, 4, 3, 2 });
    shape.remove(3);
    try std.testing.expect(shape.size() == 60);
    shape.remove(0);
    try std.testing.expect(shape.size() == 12);
    try std.testing.expect(shape.get(0) == 4);
    shape.remove(1);
    try std.testing.expect(shape.size() == 4);
    try std.testing.expect(shape.get(0) == 4);
    shape.remove(0);
    try std.testing.expect(shape.size() == 1);
    try std.testing.expect(shape.len == 0);
}

test "broadcast" {
    const shape1 = Shape.init(&.{ 5, 3, 4, 2 });
    const shape2 = Shape.init(&.{ 4, 2 });

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var result_shape = try shape1.broadcast(shape2);
    try std.testing.expectEqualSlices(u64, shape1.slice(), result_shape.slice());

    result_shape = try shape2.broadcast(shape1);
    try std.testing.expectEqualSlices(u64, shape1.slice(), result_shape.slice());

    result_shape = try shape1.broadcast(shape1);
    try std.testing.expectEqualSlices(u64, shape1.slice(), result_shape.slice());

    result_shape = try shape2.broadcast(shape2);
    try std.testing.expectEqualSlices(u64, shape2.slice(), result_shape.slice());

    try std.testing.expectError(error.Unbroadcastable, shape1.broadcast(Shape.init(&.{ 4, 3 })));
    try std.testing.expectError(error.Unbroadcastable, shape1.broadcast(Shape.init(&.{ 3, 2 })));
    try std.testing.expectError(error.Unbroadcastable, shape1.broadcast(Shape.init(&.{ 3, 3 })));
}
