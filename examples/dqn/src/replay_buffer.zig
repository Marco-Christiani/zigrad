const std = @import("std");

/// TODO: Currently a rough prototype this is not what the final version will be
pub fn ReplayBuffer(T: type, capacity: usize) type {
    return struct {
        const Self = @This();
        const Sample = std.MultiArrayList(T);
        allocator: std.mem.Allocator,
        data: [capacity]T,
        size: usize,
        idx: usize, // make sure oldest item gets replaced

        pub fn init(allocator: std.mem.Allocator) Self {
            const buf: [capacity]T = undefined;
            return Self{ .allocator = allocator, .data = buf, .size = 0, .idx = 0 };
        }

        pub fn add(self: *Self, elem: T) void {
            self.data[self.idx] = elem;
            self.size = @min(self.size + 1, capacity);
            self.idx += 1;
            self.idx %= capacity;
        }

        /// COM
        pub fn sample(self: Self, n: usize) !Sample {
            var soa = Sample{};
            for (0..n) |_| {
                const i = std.crypto.random.uintLessThan(usize, self.size);
                try soa.append(self.allocator, self.data[i]);
            }
            return soa;
        }

        /// Sample without replacement. COM.
        pub fn sample2(self: Self, n: usize) !Sample {
            std.debug.assert(n <= capacity);
            var soa = Sample{};
            var zeros: [capacity]usize = undefined;
            for (&zeros) |*z| z.* = 0;
            const taken = zeros[0..n];
            for (0..n) |i| {
                var idx = std.crypto.random.uintLessThan(usize, self.size);
                var retries: usize = 0;
                while (std.mem.containsAtLeast(usize, taken, 1, &.{idx})) {
                    idx = std.crypto.random.uintLessThan(usize, self.size);
                    retries += 1;
                    if (retries >= n - n / 2) std.debug.print("retry:{d} idx:{d} n:{d} size:{d} capacity:{d}\n", .{ retries, idx, n, self.size, capacity });
                }
                taken[i] = idx;
                try soa.append(self.allocator, self.data[idx]);
            }
            return soa;
        }

        pub fn isFull(self: Self) bool {
            return capacity == self.size;
        }
    };
}

test ReplayBuffer {
    const Transition = struct {
        idx: usize,
        action: enum { right, noop, left },
        reward: f32,
        state: [4]f32,
    };

    const allocator = std.testing.allocator;
    var rb = ReplayBuffer(Transition, 4).init(allocator);

    rb.add(.{
        .idx = 0,
        .action = .right,
        .reward = 0.11,
        .state = .{ 1.0, 2.0, 3.0, 4.0 },
    });

    rb.add(.{
        .idx = 1,
        .action = .left,
        .reward = -0.11,
        .state = .{ 1.1, 2.1, 3.1, 4.1 },
    });

    rb.add(.{
        .idx = 2,
        .action = .right,
        .reward = 0.3,
        .state = .{ 1.2, 2.2, 3.2, 4.2 },
    });

    try std.testing.expectEqual(3, rb.size);

    var s = try rb.sample2(2);
    try std.testing.expectEqual(2, s.len);
    s.deinit(allocator);
    // larger than current size, smaller than capacity
    s = try rb.sample(4);
    try std.testing.expectEqual(4, s.len);
    std.debug.print("{any} {any}\n", .{ s.items(.idx), s.items(.state) });
    s.deinit(allocator);

    // larger than capacity
    s = try rb.sample(5);
    try std.testing.expectEqual(5, s.len);
    std.debug.print("{any} {any}\n", .{ s.items(.idx), s.items(.state) });
    s.deinit(allocator);

    // add beyond capacity
    rb.add(.{
        .idx = 3,
        .action = .noop,
        .reward = 0.1,
        .state = .{ 1.3, 2.3, 3.3, 4.3 },
    });

    rb.add(.{
        .idx = 4,
        .action = .noop,
        .reward = -0.4,
        .state = .{ 1.4, 2.4, 3.4, 4.4 },
    });
    s = try rb.sample(3);
    try std.testing.expectEqual(3, s.len);
    s.deinit(allocator);
    std.debug.print("repeating\n", .{});
    const bs = 3;
    s = try rb.sample2(bs);
    for (0..2 * bs) |_| {
        std.debug.print("{any} {any}\n", .{ s.items(.idx), s.items(.state) });
        var states_flat = try allocator.alloc(f32, bs * 4);
        defer allocator.free(states_flat);
        for (s.items(.state), 0..) |state, i| {
            @memcpy(states_flat[i * 4 .. (i + 1) * 4], &state);
        }
        for (0..bs) |i| {
            std.debug.print("[", .{});
            for (0..4) |j| {
                std.debug.print(" {d}", .{states_flat[i * 4 + j]});
            }
            std.debug.print("]\n", .{});
        }
        s.deinit(allocator);
        s = try rb.sample2(bs);
    }
    s.deinit(allocator);
}
