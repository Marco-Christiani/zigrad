const std = @import("std");
fn print(comptime fmt: []const u8, args: anytype) void {
    if (@inComptime()) {
        @compileError(std.fmt.comptimePrint(fmt, args));
    } else if (std.testing.backend_can_print) {
        std.debug.print(fmt, args);
    }
}

pub fn expectApproxEqRelSlices(
    comptime T: type,
    expected: []const T,
    actual: []const T,
    tolerance: T,
) !void {
    if (@typeInfo(T) != .float) {
        @compileError("expectApproxEqRelSlices only works with float types");
    }

    if (expected.ptr == actual.ptr and expected.len == actual.len) return;

    const min_len = @min(expected.len, actual.len);
    var index: usize = 0;
    while (index < min_len) : (index += 1) {
        if (!std.math.approxEqRel(T, expected[index], actual[index], tolerance)) {
            break;
        }
    }

    if (index == expected.len and expected.len == actual.len) return;

    const stderr = std.io.getStdErr();
    const ttyconf = std.io.tty.detectConfig(stderr);

    print("slices differ (relative tolerance = {}). First mismatch at index {d} (0x{X})\n", .{
        tolerance, index, index,
    });

    const max_lines: usize = 16;
    const window_start: usize = @max(@min(index, expected.len), 1) - 1;
    const window_end: usize = @min(expected.len, window_start + max_lines);

    const expected_window = expected[window_start..window_end];
    const actual_window = actual[window_start..@min(actual.len, window_start + expected_window.len)];

    print("\n============ expected ============\n", .{});
    try printFloatSliceWindow(T, expected_window, actual_window, window_start, ttyconf, false, tolerance);
    print("============  actual  ============\n", .{});
    try printFloatSliceWindow(T, actual_window, expected_window, window_start, ttyconf, true, tolerance);
    print("==================================\n", .{});

    return error.TestExpectedApproxEqRel;
}

fn printFloatSliceWindow(
    comptime T: type,
    primary: []const T,
    other: []const T,
    start_index: usize,
    ttyconf: std.io.tty.Config,
    color_diff: bool,
    tolerance: T,
) !void {
    const writer = std.io.getStdErr().writer();
    for (primary, 0..) |val, i| {
        const idx = start_index + i;
        const mismatch = i >= other.len or !std.math.approxEqRel(T, val, other[i], tolerance);

        if (color_diff and mismatch) try ttyconf.setColor(writer, .red);
        try writer.print("[{d}]: {e}\n", .{ idx, val });
        if (color_diff and mismatch) try ttyconf.setColor(writer, .reset);
    }
}

test "expectApproxEqRelSlices" {
    const T = f32;
    const eps = std.math.floatEps(T);

    const a = [_]T{ 1.0, 2.0, 3.0000005 };
    const b = [_]T{ 1.0, 2.0, 3.0 };
    try expectApproxEqRelSlices(T, &a, &b, 10 * eps); // Should pass
    const result = expectApproxEqRelSlices(T, &a, &b, eps); // Should fail
    try std.testing.expectError(error.TestExpectedApproxEqRel, result);
}
