//! Main benchmark orchestration for matmul performance testing.
const std = @import("std");
const zg = @import("../root.zig");
const gemm = zg.kernels.gemm;
const tvm_module = zg.tvm.module;
const config = @import("config.zig");
const stats = @import("stats.zig");
const correctness = @import("correctness.zig");
const tvm_adapter = @import("tvm_adapter.zig");
const xla_adapter = @import("xla_adapter.zig");

const BenchmarkConfig = config.BenchmarkConfig;
const Shape = config.Shape;
const Implementation = config.Implementation;
const BenchmarkResult = config.BenchmarkResult;

const log = std.log.scoped(.@"zg/benchmark");

/// Tolerance for correctness verification (matches TVM convention).
/// TODO: centralize
const TOLERANCE: f64 = 1e-4;
const syms = zg.utils.Symbols.unicode;

/// Benchmark harness for matmul implementations.
pub const Harness = struct {
    allocator: std.mem.Allocator,
    config: BenchmarkConfig,
    results: std.ArrayList(BenchmarkResult),
    rng: std.Random.DefaultPrng,
    tvm_cache_dir: []const u8,

    // XLA/PJRT contexts (lazy-initialized on first use)
    xla_cpu_ctx: ?xla_adapter.XlaContext = null,
    xla_gpu_ctx: ?xla_adapter.XlaContext = null,

    // TVM module caches (lazy-initialized per shape, key = "MxNxK")
    tvm_cpu_cache: std.StringHashMap(*tvm_module.TunedModule),
    tvm_gpu_cache: std.StringHashMap(*tvm_module.TunedModule),

    /// Initialize the harness with the given configuration.
    pub fn init(allocator: std.mem.Allocator, cfg: BenchmarkConfig, tvm_cache_dir: []const u8) !Harness {
        const results = try std.ArrayList(BenchmarkResult).initCapacity(allocator, 16);
        return Harness{
            .allocator = allocator,
            .config = cfg,
            .results = results,
            .rng = std.Random.DefaultPrng.init(cfg.seed),
            .tvm_cache_dir = tvm_cache_dir,
            .tvm_cpu_cache = std.StringHashMap(*tvm_module.TunedModule).init(allocator),
            .tvm_gpu_cache = std.StringHashMap(*tvm_module.TunedModule).init(allocator),
        };
    }

    /// Release resources.
    pub fn deinit(self: *Harness) void {
        self.results.deinit(self.allocator);

        if (self.xla_cpu_ctx) |*ctx| ctx.deinit();
        if (self.xla_gpu_ctx) |*ctx| ctx.deinit();

        self.deinit_tvm_cache(&self.tvm_cpu_cache);
        self.deinit_tvm_cache(&self.tvm_gpu_cache);
    }

    fn deinit_tvm_cache(self: *Harness, cache: *std.StringHashMap(*tvm_module.TunedModule)) void {
        var iter = cache.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        cache.deinit();
    }

    /// Run all benchmarks according to the configuration.
    pub fn run(self: *Harness) !void {
        log.info("starting benchmark suite: {d} shapes x {d} implementations", .{
            self.config.shapes.len,
            self.config.implementations.len,
        });

        for (self.config.shapes) |shape| {
            try self.run_shape(shape);
        }

        log.info("benchmark suite complete: {d} results collected", .{self.results.items.len});
    }

    /// Run benchmarks for a single shape across all implementations.
    fn run_shape(self: *Harness, shape: Shape) !void {
        log.info("benchmarking shape: {any}", .{shape});

        // Allocate input/output matrices (row-major layout)
        const a = try self.allocator.alloc(f32, @intCast(shape.m * shape.k));
        defer self.allocator.free(a);
        const b = try self.allocator.alloc(f32, @intCast(shape.k * shape.n));
        defer self.allocator.free(b);
        const c = try self.allocator.alloc(f32, @intCast(shape.m * shape.n));
        defer self.allocator.free(c);

        // Fill inputs with random data
        self.fill_random(a);
        self.fill_random(b);

        // Compute reference result for correctness verification
        var reference: ?[]f32 = null;
        defer if (reference) |ref| self.allocator.free(ref);

        if (self.config.verify_correctness) {
            reference = try self.allocator.alloc(f32, @intCast(shape.m * shape.n));
            correctness.reference_matmul_f32(
                shape.m,
                shape.n,
                shape.k,
                a,
                @intCast(shape.k),
                b,
                @intCast(shape.n),
                reference.?,
                @intCast(shape.n),
            );
        }

        // Benchmark each implementation
        for (self.config.implementations) |impl| {
            const result = try self.benchmark_impl(impl, shape, a, b, c, reference);
            try self.results.append(self.allocator, result);

            if (result.passed) {
                log.info("  {s}: {d:.2} GFLOP/s (median {d:.2} us, error {any})", .{
                    impl.display_name(),
                    result.gflops,
                    result.median_us,
                    result.max_abs_error,
                });
            } else {
                log.err("  {s}: FAILED correctness check (error {any} > tolerance {any})", .{
                    impl.display_name(),
                    result.max_abs_error,
                    TOLERANCE,
                });
            }
        }
    }

    /// Benchmark a single implementation on the given shape.
    fn benchmark_impl(
        self: *Harness,
        impl: Implementation,
        shape: Shape,
        a: []const f32,
        b: []const f32,
        c: []f32,
        reference: ?[]const f32,
    ) !BenchmarkResult {
        // Warmup
        for (0..self.config.warmup_iters) |_| {
            @memset(c, 0.0);
            try self.run_kernel(impl, shape, a, b, c);
        }

        // Measurement
        var times = try self.allocator.alloc(f64, self.config.bench_iters);
        defer self.allocator.free(times);

        for (0..self.config.bench_iters) |i| {
            @memset(c, 0.0); // Zero before each iteration

            const start = std.time.nanoTimestamp();
            try self.run_kernel(impl, shape, a, b, c);
            const end = std.time.nanoTimestamp();

            times[i] = @as(f64, @floatFromInt(end - start)) / 1000.0; // Convert to microseconds
        }

        // Compute statistics
        const median_us = stats.median(times);
        const mean_us = stats.mean(times);
        const stddev_us = stats.stddev(times);

        // Compute GFLOP/s: 2*M*N*K flops / (time_us * 1000.0)
        const flops = 2.0 * @as(f64, @floatFromInt(shape.m)) *
            @as(f64, @floatFromInt(shape.n)) *
            @as(f64, @floatFromInt(shape.k));
        const gflops = flops / (median_us * 1000.0);

        // Verify correctness
        var max_err: f64 = 0.0;
        var passed = true;
        if (reference) |ref| {
            max_err = correctness.max_abs_error(ref, c);
            passed = (max_err <= TOLERANCE);
        }

        return BenchmarkResult{
            .impl = impl,
            .shape = shape,
            .median_us = median_us,
            .mean_us = mean_us,
            .stddev_us = stddev_us,
            .gflops = gflops,
            .max_abs_error = max_err,
            .passed = passed,
        };
    }

    /// Execute a single kernel invocation.
    fn run_kernel(
        self: *Harness,
        impl: Implementation,
        shape: Shape,
        a: []const f32,
        b: []const f32,
        c: []f32,
    ) !void {
        const lda: usize = @intCast(shape.k);
        const ldb: usize = @intCast(shape.n);
        const ldc: usize = @intCast(shape.n);
        switch (impl) {
            .blas => {
                gemm.blas_gemm_f32(shape.m, shape.n, shape.k, a, lda, b, ldb, c, ldc);
            },
            .zig_naive => {
                gemm.gemm_f32(shape.m, shape.n, shape.k, a, lda, b, ldb, c, ldc);
            },
            .tvm_cpu => try self.run_tvm(shape, a, b, c, .cpu),
            .tvm_gpu => try self.run_tvm(shape, a, b, c, .gpu),
            .xla_cpu => if (comptime zg.build_options.has_mlir)
                try self.run_xla(shape, a, b, c, .cpu)
            else
                return error.MlirDisabled,
            .xla_gpu => if (comptime zg.build_options.has_mlir)
                try self.run_xla(shape, a, b, c, .gpu)
            else
                return error.MlirDisabled,
        }
    }

    const DeviceKind = enum { cpu, gpu };

    /// Execute TVM matmul (loads pre-tuned module from artifacts/tvm_cache/).
    fn run_tvm(
        self: *Harness,
        shape: Shape,
        a: []const f32,
        b: []const f32,
        c: []f32,
        device: DeviceKind,
    ) !void {
        try zg.tvm.ffi.ensure_loaded(self.allocator, .{});
        const cache_key = try std.fmt.allocPrint(self.allocator, "{d}x{d}x{d}", .{ shape.m, shape.n, shape.k });
        defer self.allocator.free(cache_key);

        const cache = switch (device) {
            .cpu => &self.tvm_cpu_cache,
            .gpu => &self.tvm_gpu_cache,
        };

        const tuned = if (cache.get(cache_key)) |module|
            module
        else blk: {
            const module = try self.allocator.create(tvm_module.TunedModule);
            errdefer self.allocator.destroy(module);

            // const work_dir = try std.fmt.allocPrint(
            //     self.allocator,
            //     "{s}/{s}",
            //     .{ self.tvm_cache_dir, if (device == .cpu) "cpu" else "cuda" },
            // );
            // defer self.allocator.free(work_dir);
            // TODO: we are migrating to the hash based system, device is included in the hash,
            //  a valid path is like `artifacts/tvm_cache/cpu/820e34c955246375` and would have
            //  to be passed explicitly unless we put the hashing in the load path, which it
            //  probably should be. Also, we have to look into evidence of the above convention
            //  lingering around. Once we remove references we can clean this up.
            const work_dir = self.tvm_cache_dir;

            module.* = try tvm_module.load(self.allocator, .{ .work_dir = work_dir });

            const owned_key = try self.allocator.dupe(u8, cache_key);
            try cache.put(owned_key, module);
            break :blk module;
        };

        switch (device) {
            .cpu => try tvm_adapter.execute_with_module(self.allocator, tuned, shape.m, shape.n, shape.k, a, b, c),
            .gpu => try tvm_adapter.execute_gpu_with_module(self.allocator, tuned, shape.m, shape.n, shape.k, a, b, c),
        }
    }

    /// Execute XLA/PJRT matmul (compiles on-the-fly, caches per shape).
    fn run_xla(
        self: *Harness,
        shape: Shape,
        a: []const f32,
        b: []const f32,
        c: []f32,
        device: DeviceKind,
    ) !void {
        const ctx = switch (device) {
            .cpu => &self.xla_cpu_ctx,
            .gpu => &self.xla_gpu_ctx,
        };

        if (ctx.* == null) {
            ctx.* = switch (device) {
                .cpu => try xla_adapter.XlaContext.init(self.allocator),
                .gpu => try xla_adapter.XlaContext.init_gpu(self.allocator),
            };
        }

        try ctx.*.?.execute(shape.m, shape.n, shape.k, a, b, c);
    }

    /// Fill a slice with random f32 values in [-1.0, 1.0].
    fn fill_random(self: *Harness, slice: []f32) void {
        for (slice) |*val| {
            val.* = self.rng.random().float(f32) * 2.0 - 1.0;
        }
    }

    /// Print benchmark results in a human-readable table format.
    pub fn print_results(self: *const Harness) !void {
        var buffer: [8192]u8 = undefined;
        var stdout_writer = std.fs.File.stdout().writer(&buffer);
        const writer = &stdout_writer.interface;

        const header_sep = "\n" ++ "=" ** 60 ++ "\n";
        try writer.writeAll(header_sep ++ "Matmul Benchmark Results (CPU)" ++ header_sep);

        // Group results by shape
        var seen_shapes = std.AutoHashMap(Shape, void).init(self.allocator);
        defer seen_shapes.deinit();

        for (self.config.shapes) |shape| {
            if (seen_shapes.contains(shape)) continue;
            try seen_shapes.put(shape, {});

            try writer.print("\nShape: {any}\n\n", .{shape});
            try writer.writeAll("Implementation            Median (us)    GFLOP/s    vs Naive\n");
            try writer.writeAll("---------------------------------------------------------\n");

            // Find naive baseline for this shape
            var naive_gflops: f64 = 1.0;
            for (self.results.items) |result| {
                if (result.shape.m == shape.m and result.shape.n == shape.n and result.shape.k == shape.k and
                    result.impl == .zig_naive)
                {
                    naive_gflops = result.gflops;
                    break;
                }
            }

            // Print results for this shape
            for (self.results.items) |result| {
                if (result.shape.m != shape.m or result.shape.n != shape.n or result.shape.k != shape.k) continue;

                const speedup = result.gflops / naive_gflops;
                const status = if (result.passed) syms.check else syms.x;

                try writer.print("{s} {s:<23} {d:>10.2} {d:>10.2} {d:>8.2}{s}\n", .{
                    status,
                    result.impl.display_name(),
                    result.median_us,
                    result.gflops,
                    speedup,
                    syms.mul,
                });
            }
        }

        // Summary statistics
        try writer.writeAll(header_sep ++ "Summary" ++ header_sep);

        if (self.results.items.len > 0) {
            // Find best implementations
            var best_overall: ?BenchmarkResult = null;
            var best_hand_rolled: ?BenchmarkResult = null;

            for (self.results.items) |result| {
                if (!result.passed) continue;

                if (best_overall == null or result.gflops > best_overall.?.gflops) {
                    best_overall = result;
                }

                const is_hand_rolled = switch (result.impl) {
                    .zig_naive => true,
                    else => false,
                };

                if (is_hand_rolled and (best_hand_rolled == null or result.gflops > best_hand_rolled.?.gflops)) {
                    best_hand_rolled = result;
                }
            }

            if (best_overall) |best| {
                try writer.print("Best overall: {s} ({d:.2} GFLOP/s)\n", .{
                    best.impl.display_name(),
                    best.gflops,
                });
            }

            if (best_hand_rolled) |best| {
                try writer.print("Best hand-rolled: {s} ({d:.2} GFLOP/s)\n", .{
                    best.impl.display_name(),
                    best.gflops,
                });
            }
        }

        try writer.writeAll("\n");
        try writer.flush();
    }
};

test "harness: init and deinit" {
    const allocator = std.testing.allocator;

    const cfg = BenchmarkConfig{
        .shapes = &[_]Shape{.{ .m = 4, .n = 4, .k = 4 }},
        .implementations = &[_]Implementation{.zig_naive},
        .bench_iters = 5,
    };

    var harness = try Harness.init(allocator, cfg, "artifacts/tvm_cache");
    defer harness.deinit();

    try std.testing.expect(harness.results.items.len == 0);
}
