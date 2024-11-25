const std = @import("std");
const testing = std.testing;
const pb = @import("protobuf");
const tb = @import("gen_proto/tensorboard.pb.zig");
const maskedCrc32c = @import("crc.zig").maskedCrc32c;
const TensorBoardLogger = @import("tensorboard_logger.zig").TensorBoardLogger;

test "tensorboard logger" {
    const allocator = std.testing.allocator;
    var logger = try TensorBoardLogger.init("/tmp/tensorboard_logs", allocator);
    defer logger.deinit();

    var prng = std.rand.DefaultPrng.init(0);
    var random = prng.random();

    for (0..10) |i| {
        const step: i64 = @intCast(i);

        // Add scalar
        const scalar_value: f32 = @as(f32, @floatFromInt(i)) * 2.0;
        try logger.addScalar("example/scalar", scalar_value, step);

        // Add histogram
        var histogram_values: [1000]f64 = undefined;
        for (&histogram_values) |*v| {
            v.* = random.floatNorm(f64) + @as(f64, @floatFromInt(i));
        }
        try logger.addHistogram("example/histogram", &histogram_values, step);
    }
}
