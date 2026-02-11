/// Statistical utilities for benchmark result analysis.
const std = @import("std");

/// Computes the median of a slice of f64 values.
/// The input slice is sorted in-place.
pub fn median(values: []f64) f64 {
    if (values.len == 0) return 0.0;

    std.mem.sort(f64, values, {}, std.sort.asc(f64));

    const mid = values.len / 2;
    if (values.len % 2 == 0) {
        return (values[mid - 1] + values[mid]) / 2.0;
    } else {
        return values[mid];
    }
}

/// Computes the mean (average) of a slice of f64 values.
pub fn mean(values: []const f64) f64 {
    if (values.len == 0) return 0.0;

    var sum: f64 = 0.0;
    for (values) |v| {
        sum += v;
    }
    return sum / @as(f64, @floatFromInt(values.len));
}

/// Computes the standard deviation of a slice of f64 values.
pub fn stddev(values: []const f64) f64 {
    if (values.len <= 1) return 0.0;

    const m = mean(values);
    var sum_sq: f64 = 0.0;
    for (values) |v| {
        const diff = v - m;
        sum_sq += diff * diff;
    }
    return @sqrt(sum_sq / @as(f64, @floatFromInt(values.len)));
}

test "stats: median odd length" {
    var values = [_]f64{ 3.0, 1.0, 2.0 };
    try std.testing.expectApproxEqAbs(2.0, median(&values), 1e-9);
}

test "stats: median even length" {
    var values = [_]f64{ 4.0, 1.0, 3.0, 2.0 };
    try std.testing.expectApproxEqAbs(2.5, median(&values), 1e-9);
}

test "stats: mean" {
    const values = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    try std.testing.expectApproxEqAbs(3.0, mean(&values), 1e-9);
}

test "stats: stddev" {
    const values = [_]f64{ 2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0 };
    const expected_stddev = 2.0;
    try std.testing.expectApproxEqAbs(expected_stddev, stddev(&values), 1e-9);
}
