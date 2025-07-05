// const std = @import("std");
//
// pub fn main() !void {
//     var stdout = std.io.getStdOut().writer();
//     var sum: u64 = 0;
//     var i: usize = 0;
//     while (i < 1_000_000) : (i += 1) {
//         sum += i * 3;
//     }
//     try stdout.print("Sum: {}\n", .{sum});
// }
const std = @import("std");
const Timer = std.time.Timer;

/// Stateless: recompute digits from integer index
fn BaseCounter_DivMod(comptime N: usize, comptime B: usize) type {
    return struct {
        const Self = @This();
        i: usize = 0,

        pub fn next(self: *Self) ?[N]usize {
            const max = std.math.pow(usize, B, N);
            if (self.i >= max) return null;

            var x = self.i;
            var result: [N]usize = undefined;
            for (0..N) |j| {
                result[N - 1 - j] = x % B;
                x /= B;
            }
            self.i += 1;
            return result;
        }
    };
}

/// Stateful: increment digits directly with carry
fn BaseCounter_Increment(comptime N: usize, comptime B: usize) type {
    return struct {
        const Self = @This();
        done: bool = false,
        digits: [N]usize = .{0} ** N,

        fn increment(self: *Self) void {
            var i: usize = N;
            while (i > 0) {
                i -= 1;
                self.digits[i] += 1;
                if (self.digits[i] < B) return;
                self.digits[i] = 0;
            }
            self.done = true;
        }

        pub fn next(self: *Self) ?[N]usize {
            if (self.done) return null;
            const current = self.digits;
            self.increment();
            return current;
        }
    };
}

fn CoordinateIterator(comptime N: usize) type {
    return struct {
        index: [N]usize = .{0} ** N,
        shape: []const usize,
        rank: usize,
        done: bool = false,

        const Self = @This();

        pub fn init(shape: []const usize) Self {
            std.debug.assert(shape.len <= N);
            return Self{
                .shape = shape,
                .rank = shape.len,
            };
        }

        fn increment(self: *Self) void {
            var i: usize = self.rank;
            while (i > 0) {
                i -= 1;
                self.index[i] += 1;
                if (self.index[i] < self.shape[i]) return;
                self.index[i] = 0;
            }
            self.done = true;
        }

        pub fn next(self: *Self) ?[]const usize {
            if (self.done) return null;
            const current = self.index[0..self.rank];
            self.increment();
            return current;
        }

        pub fn reset(self: *Self) void {
            self.index = .{0} ** N;
            self.done = false;
        }
    };
}

fn CoordinateIterator_Branchless(comptime N: usize) type {
    return struct {
        index: [N]usize = .{0} ** N,
        shape: []const usize,
        rank: usize,
        done: bool = false,

        const Self = @This();

        pub fn init(shape: []const usize) Self {
            std.debug.assert(shape.len <= N);
            return .{ .shape = shape, .rank = shape.len };
        }

        inline fn increment(self: *Self) void {
            var carry: usize = 1;
            var i: usize = self.rank;
            while (i > 0) {
                i -= 1;
                const val = self.index[i] + carry;
                const limit = self.shape[i];

                if (val < limit) {
                    self.index[i] = val;
                    carry = 0;
                } else {
                    self.index[i] = val - limit;
                    carry = 1;
                }
            }
            if (carry != 0) self.done = true;
        }

        pub inline fn next(self: *Self) ?[]const usize {
            if (self.done) return null;
            const result = self.index[0..self.rank];
            self.increment();
            return result;
        }
    };
}

fn UnrolledCoordinateIterator(comptime N: usize) type {
    return struct {
        index: [N]usize = .{0} ** N,
        shape: [N]usize,
        done: bool = false,

        const Self = @This();

        pub fn init(shape: []const usize) Self {
            std.debug.assert(shape.len == N);
            var fixed: [N]usize = undefined;
            @memcpy(&fixed, shape.ptr);
            return .{ .shape = fixed };
        }

        inline fn increment(self: *Self) void {
            comptime var i = N;
            var index = &self.index;
            const shape = &self.shape;
            inline while (i > 0) {
                i -= 1;
                index[i] += 1;
                if (index[i] < shape[i]) return;
                index[i] = 0;
            }
            self.done = true;
        }

        pub inline fn next(self: *Self) ?[]const usize {
            if (self.done) return null;
            const result = self.index[0..N];
            self.increment();
            return result;
        }
    };
}

fn UnrolledConstShapeIterator(comptime Shape: []const usize) type {
    const N = Shape.len;

    return struct {
        index: [N]usize = .{0} ** N,
        done: bool = false,

        const Self = @This();

        pub inline fn next(self: *Self) ?[]const usize {
            if (self.done) return null;
            const result = self.index[0..];

            comptime var i = N;
            var index = &self.index;
            inline while (i > 0) {
                i -= 1;
                index[i] += 1;
                if (index[i] < Shape[i]) break;
                index[i] = 0;
                if (i == 0) self.done = true;
            }

            return result;
        }
    };
}

fn CoordinateIterator_Pointer(comptime N: usize) type {
    return struct {
        index: [N]usize = .{0} ** N,
        shape: [N]usize,
        done: bool = false,

        const Self = @This();

        pub fn init(shape: []const usize) Self {
            std.debug.assert(shape.len == N);
            var fixed: [N]usize = undefined;
            @memcpy(&fixed, shape.ptr);
            return .{ .shape = fixed };
        }

        pub inline fn next(self: *Self) ?[]const usize {
            if (self.done) return null;
            const result = self.index[0..N];

            const base_idx = @as([*]usize, @ptrCast(&self.index));
            const base_shp = @as([*]usize, @ptrCast(&self.shape));
            var iptr = base_idx + (N - 1);
            var sptr = base_shp + (N - 1);

            while (true) {
                const v = iptr[0] + 1;
                iptr[0] = v;
                if (v < sptr[0]) break;
                iptr[0] = 0;
                if (@intFromPtr(iptr) == @intFromPtr(base_idx)) {
                    self.done = true;
                    break;
                }
                iptr -= 1;
                sptr -= 1;
            }

            return result;
        }
    };
}

fn benchmark_coords(comptime Label: []const u8, comptime N: usize, shape: []const usize, stdout: anytype) !f64 {
    var it = CoordinateIterator(N).init(shape);
    var timer = try Timer.start();
    var sum: usize = 0;
    while (it.next()) |idx| {
        sum += idx[0];
    }
    const elapsed = @as(f64, @floatFromInt(timer.read())) / @as(f64, @floatFromInt(std.time.ns_per_s));
    try stdout.print("{s}: {} s (checksum = {})\n", .{ Label, elapsed, sum });
    return elapsed;
}

fn benchmark_coords_static(comptime Label: []const u8, it: anytype, stdout: anytype) !f64 {
    var counter = it;
    var timer = try Timer.start();
    var sum: usize = 0;
    while (counter.next()) |idx| {
        sum += idx[0];
    }
    const elapsed = @as(f64, @floatFromInt(timer.read())) / @as(f64, @floatFromInt(std.time.ns_per_s));
    try stdout.print("{s}: {} s (checksum = {})\n", .{ Label, elapsed, sum });
    return elapsed;
}

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();

    inline for (.{ 4, 8 }) |N| {
        inline for (.{ 4, 8, 10 }) |B| {
            const shape = [_]usize{B} ** N;
            try stdout.print("Running coordinate iterator variants: N={d} B={d}\n", .{ N, B });

            _ = try benchmark_coords_static("UnrolledConstShapeIterator", UnrolledConstShapeIterator(&shape){}, stdout);
            _ = try benchmark_coords_static("CoordinateUnrolledCT      ", UnrolledCoordinateIterator(N).init(&shape), stdout);
            _ = try benchmark_coords_static("CoordinatePointer         ", CoordinateIterator_Pointer(N).init(&shape), stdout);
            _ = try benchmark_coords_static("CoordinateBranchless      ", CoordinateIterator_Branchless(N).init(&shape), stdout);
            _ = try benchmark_coords("CoordinateLoop            ", N, &shape, stdout);
            try stdout.print("\n", .{});
        }
    }
}
