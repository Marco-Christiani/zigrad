const std = @import("std");
const utils = @import("utils.zig");

const log = std.log.scoped(.zg_shape);

const Self = @This();
shape: []usize,
strides: []usize, // TODO: make use in more places
alloc: std.mem.Allocator,

pub fn init(shape: []const usize, allocator: std.mem.Allocator) !*Self {
    if (shape.len == 0) return error.InvalidShape;
    const self = try allocator.create(Self);
    self.* = Self{
        .alloc = allocator,
        .shape = try allocator.dupe(usize, shape),
        .strides = try calculateStrides(shape, allocator),
    };
    return self;
}

pub fn deinit(self: *Self) void {
    self.alloc.free(self.shape);
    self.alloc.free(self.strides);
    self.alloc.destroy(self);
}

pub fn copy(self: Self, allocator: std.mem.Allocator) !*Self {
    return try Self.init(self.shape, allocator);
}

pub fn broadcast(self: Self, other: Self) !*Self {
    const dims = @max(self.len(), other.len());
    const result_shape = try self.alloc.alloc(usize, dims);
    errdefer self.alloc.free(result_shape);
    var i: usize = 0;
    while (i < dims) : (i += 1) {
        const dim_a = if (i < self.shape.len) self.shape[self.shape.len - 1 - i] else 1;
        const dim_b = if (i < other.shape.len) other.shape[other.shape.len - 1 - i] else 1;
        if (dim_a != dim_b and dim_a != 1 and dim_b != 1) {
            log.err("Cannot broadcast {d} and {d}. (i={d}) dim a_{d}={d} b_{d}={d}", .{
                self.shape,
                other.shape,
                i,
                self.shape.len - 1 - i,
                dim_a,
                other.shape.len - 1 - i,
                dim_b,
            });
            return error.Unbroadcastable;
        }
        result_shape[dims - 1 - i] = @max(dim_a, dim_b);
    }
    const result = try self.alloc.create(Self);
    result.* = Self{
        .shape = result_shape,
        .alloc = self.alloc,
        .strides = try calculateStrides(result_shape, self.alloc),
    };
    return result;
}

// TODO: usage should be minimized (or size stored but rarely needed this was)
pub fn size(self: Self) usize {
    return utils.prod(self.shape);
}

pub fn len(self: Self) usize {
    return self.shape.len;
}

/// Functional number of dims if the shape was squeezed
pub fn realdims(self: Self) !usize {
    if (self.shape.len == 0) return error.EmptyShape;
    if (self.size() == 1) return 1; // scalar
    var ndim: usize = 0;
    for (self.shape) |e| {
        if (e != 1) ndim += 1;
    }
    return ndim;
}

pub fn _squeeze(self: *Self) !void {
    // TODO: realloc is a thing
    var newshape = try self.alloc.alloc(usize, try self.realdims());
    // scalar case
    if (self.size() == 1) {
        newshape[0] = 1;
    } else {
        var j: usize = 0;
        for (0..self.shape.len) |i| {
            if (self.shape[i] != 1) {
                newshape[j] = self.shape[i];
                j += 1;
            }
        }
    }
    self.alloc.free(self.shape);
    self.shape = newshape;
    self.strides = try calculateStrides(newshape, self.alloc);
}

pub fn _unsqueeze(self: *Self) !void {
    // TODO: realloc is a thing
    var newshape = try self.alloc.alloc(usize, self.shape.len + 1);
    @memcpy(newshape[1..], self.shape);
    newshape[0] = 1;
    self.alloc.free(self.shape);
    self.shape = newshape;
    self.strides = try calculateStrides(newshape, self.alloc);
}

pub fn _reshape(self: *Self, shape: []const usize) !void {
    // TODO: realloc is a thing
    const requested_size = utils.prod(shape);
    if (requested_size != self.size()) {
        log.info("ShapeOutOfBounds requested_size={d} self.size={d} self.shape={d} requested_shape={d}", .{ requested_size, self.size(), self.shape, shape });
        return error.ShapeOutOfBounds;
    }
    self.alloc.free(self.shape);
    self.shape = try self.alloc.dupe(usize, shape);
    self.strides = try calculateStrides(self.shape, self.alloc);
}

pub fn get(self: Self, dim: usize) !usize {
    if (dim >= self.len()) return error.DimOutOfBounds;
    return self.shape[dim];
}

pub fn rget(self: Self, dim: i32) !usize {
    if (@abs(dim) > self.len()) return error.DimOutOfBounds;
    if (dim >= 0) return error.InvalidDim; // must be a negative number
    return self.shape[self.shape.len - @abs(dim)];
}

const EqualOptions = struct {
    /// require exactly equal
    strict: bool = false,
};

pub fn eq(a: Self, b: Self, options: EqualOptions) bool {
    return eqRaw(a.shape, b.shape, options);
}

pub fn eqRaw(a: []usize, b: []usize, options: EqualOptions) bool {
    if (options.strict) return std.mem.eql(usize, a, b);
    const dims = @max(a.len, b.len);
    var i: usize = 0;
    while (i < dims) : (i += 1) {
        const dim_a = if (i < a.len) a[a.len - 1 - i] else 1;
        const dim_b = if (i < b.len) b[b.len - 1 - i] else 1;
        if (dim_a != dim_b and dim_a != 1 and dim_b != 1) return false;
    }
    return true;
}

fn calculateStrides(shape: []const usize, allocator: std.mem.Allocator) ![]usize {
    var strides = try allocator.alloc(usize, shape.len);
    errdefer allocator.free(strides);
    const n = shape.len;
    var stride: usize = 1;
    for (0..n) |i| {
        strides[n - i - 1] = stride;
        stride *= shape[n - i - 1];
    }

    return strides;
}
