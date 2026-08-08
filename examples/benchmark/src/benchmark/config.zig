//! Configuration types for the matmul benchmark harness.
const std = @import("std");
const zg = @import("zigrad");

const syms = zg.utils.Symbols.unicode;

pub const DeviceKind = enum { cpu, gpu };

/// Element type for benchmark buffers.
///
/// Maps to Zig numeric types at comptime.
/// DLPack's dtype_of does not appear to support bf16 yet and our naive
///  baseline is zig, which has no native bf16 type.
/// TODO: need to look into bf16, may just be incomplete bindings on our part.
pub const DType = enum {
    f16,
    f32,

    /// Returns the Zig numeric type corresponding to this dtype.
    pub fn ZigType(comptime self: DType) type {
        return switch (self) {
            .f16 => f16,
            .f32 => f32,
        };
    }

    /// Maps to the PR-level dtype for compiled implementations.
    pub fn to_pr_dtype(self: DType) zg.DType {
        return switch (self) {
            .f16 => .f16,
            .f32 => .f32,
        };
    }

    /// Maps a Zig numeric type back to the DType enum (comptime inverse of ZigType).
    pub fn from_zig_type(comptime T: type) DType {
        return switch (T) {
            f16 => .f16,
            f32 => .f32,
            else => @compileError("unsupported type for DType: " ++ @typeName(T)),
        };
    }

    /// Tolerance for correctness verification (wider for lower precision).
    pub fn tolerance(self: DType) f64 {
        return switch (self) {
            .f16 => 1e-2,
            .f32 => 1e-4,
        };
    }
};

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
    iree,
    xla_cpu,
    xla_gpu,

    pub fn display_name(self: Implementation) []const u8 {
        return switch (self) {
            inline else => |x| @tagName(x),
        };
    }

    pub fn is_gpu(self: Implementation) bool {
        return std.mem.endsWith(u8, @tagName(self), "gpu");
    }
};

/// Benchmark configuration.
pub const BenchmarkConfig = struct {
    /// Matrix shapes to benchmark.
    shapes: []const Shape,

    /// Implementations to test.
    implementations: []const Implementation,

    /// Element type for all buffers and kernels.
    dtype: DType = .f32,

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

    /// Element type used.
    dtype: DType,

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
