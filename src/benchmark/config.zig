//! Configuration types for the matmul benchmark harness.
const std = @import("std");
const utils = @import("../utils/root.zig");

const syms = utils.Symbols.unicode;

/// Matmul shape specification (MxK @ KxN = MxN).
pub const Shape = struct {
    m: i64,
    n: i64,
    k: i64,

    pub fn format(
        self: Shape,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("{d}{s}{d}{s}{d}", .{ self.m, syms.mul, self.n, syms.mul, self.k });
    }
};

/// Matmul implementation variant.
pub const Implementation = enum {
    blas,
    zig_naive,
    tvm_cpu,
    tvm_gpu,
    xla_cpu,
    xla_gpu,

    pub fn display_name(self: Implementation) []const u8 {
        return switch (self) {
            inline else => |x| @tagName(x),
        };
    }
};

/// Benchmark configuration.
pub const BenchmarkConfig = struct {
    /// Matrix shapes to benchmark.
    shapes: []const Shape,

    /// Implementations to test.
    implementations: []const Implementation,

    /// Number of warmup iterations (cache warming, not measured).
    warmup_iters: usize = 10,

    /// Number of benchmark iterations (timed).
    bench_iters: usize = 100,

    /// Random seed for input data generation.
    seed: u64 = 0x123456789abcdef0,

    /// Verify correctness against reference implementation.
    verify_correctness: bool = true,
};

/// Result of a single benchmark run (one implementation, one shape).
pub const BenchmarkResult = struct {
    /// Implementation tested.
    impl: Implementation,

    /// Shape tested.
    shape: Shape,

    /// Median execution time (microseconds).
    median_us: f64,

    /// Mean execution time (microseconds).
    mean_us: f64,

    /// Standard deviation of execution time (microseconds).
    stddev_us: f64,

    /// Throughput in GFLOP/s (2*M*N*K / median_us / 1000.0).
    gflops: f64,

    /// Maximum absolute error vs reference (if verified).
    max_abs_error: f64,

    /// Correctness verification passed.
    passed: bool,
};
