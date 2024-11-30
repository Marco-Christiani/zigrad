const std = @import("std");
const maskedCrc32c = @import("crc.zig").maskedCrc32c;
pub const TensorBoardLogger = @import("TensorboardLogger.zig");

test TensorBoardLogger {
    const allocator = std.testing.allocator;
    var logger = try TensorBoardLogger.init("/tmp/", allocator);
    defer logger.deinit();

    var prng = std.rand.DefaultPrng.init(0);
    var random = prng.random();

    for (0..10) |i| {
        const step: i64 = @intCast(i);

        // Add scalar
        const scalar_value: f32 = @as(f32, @floatFromInt(i)) * 2.0;
        try logger.addScalar("example/scalar", scalar_value, step);

        // Add histogram
        var histogram_values: [1000]f32 = undefined;
        for (&histogram_values) |*v| {
            v.* = random.floatNorm(f32) + @as(f32, @floatFromInt(i));
        }
        try logger.addHistogram("example/histogram", &histogram_values, step);
    }
}

test {
    std.testing.refAllDecls(@This());
}
