const std = @import("std");
pub const zg = @import("zigrad");
const mnist = zg.mnist;
const layer = zg.layer;
pub const std_options = zg.std_options;

pub const zigrad_settings: zg.Settings = .{
    .caching_policy = .{},
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    var cpu = zg.device.HostDevice.init(gpa.allocator());
    defer cpu.deinit();

    const reference = cpu.reference();

    var total: f64 = 0.0;

    for (0..10_000) |i| {
        var start = try std.time.Timer.start();
        const x = try zg.NDTensor(f32).empty(&.{ 64, 1024 }, true, reference);
        x.deinit();

        const y = try zg.NDTensor(f32).empty(&.{ 64, 1024 }, true, reference);
        y.deinit();

        const z = try zg.NDTensor(f32).empty(&.{ 64, 1024 }, true, reference);
        z.deinit();

        const elapsed = start.lap();

        if (i != 0) {
            // skip the warmup round to make things fair
            total += @floatFromInt(elapsed);
        }

        _ = arena.reset(.retain_capacity);
    }

    std.debug.print("Elapsed: {d:.10}\n", .{total / 10_000.0});
}
