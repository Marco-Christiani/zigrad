//! Main benchmark orchestration for matmul performance testing.
const std = @import("std");
const zg = @import("zigrad");
const gemm = @import("../gemm.zig");
const tvm_module = zg.tvm.module;
const tvm_tir = zg.tvm.tir;
const tvm_tune = zg.tvm.tune;
const tvm_ffi = zg.tvm.ffi;
const config = @import("config.zig");
const stats = @import("stats.zig");
const tvm_adapter = @import("tvm_adapter.zig");
const xla_adapter = @import("xla_adapter.zig");
const Cache = zg.Cache;

const BenchmarkConfig = config.BenchmarkConfig;
const DType = config.DType;
const Shape = config.Shape;
const Implementation = config.Implementation;
const BenchmarkResult = config.BenchmarkResult;

const log = std.log.scoped(.benchmark);

pub const DeviceKind = enum { cpu, gpu };

const syms = zg.utils.Symbols.unicode;

/// Benchmark harness for matmul implementations.
pub const Harness = struct {
    io: std.Io,
    allocator: std.mem.Allocator,
    config: BenchmarkConfig,
    results: std.ArrayList(BenchmarkResult),
    rng: std.Random.DefaultPrng,
    cache: Cache,

    // XLA/PJRT contexts (lazy-initialized on first use)
    xla_cpu_ctx: ?xla_adapter.XlaContext = null,
    xla_gpu_ctx: ?xla_adapter.XlaContext = null,

    // TVM module caches (lazy-initialized per shape/dtype/etc)
    // TODO: need to work on the hashing logic, dtype+target+etc should all be accounted for
    tvm_cpu_cache: std.StringHashMap(*tvm_module.TunedModule),
    tvm_gpu_cache: std.StringHashMap(*tvm_module.TunedModule),

    pub fn init(io: std.Io, environ: *const std.process.Environ.Map, allocator: std.mem.Allocator, cfg: BenchmarkConfig) !Harness {
        const results = try std.ArrayList(BenchmarkResult).initCapacity(allocator, 16);
        return Harness{
            .io = io,
            .allocator = allocator,
            .config = cfg,
            .results = results,
            .rng = std.Random.DefaultPrng.init(cfg.seed),
            .cache = try Cache.init(io, environ, .{}),
            .tvm_cpu_cache = std.StringHashMap(*tvm_module.TunedModule).init(allocator),
            .tvm_gpu_cache = std.StringHashMap(*tvm_module.TunedModule).init(allocator),
        };
    }

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
        log.info("starting benchmark suite: {d} shapes x {d} implementations ({s})", .{
            self.config.shapes.len,
            self.config.implementations.len,
            @tagName(self.config.dtype),
        });

        for (self.config.shapes) |shape| {
            try self.run_shape(shape);
        }

        log.info("benchmark suite complete: {d} results collected", .{self.results.items.len});
    }

    pub const TuneOpts = struct {
        max_trials: u32 = 64,
        trials_per_iter: u32 = 16,
        /// Delete existing artifacts and tune from scratch.
        retune: bool = false,
    };

    /// Tune search-based kernels (TVM) for all configured shapes and implementations.
    pub fn tune(self: *Harness, opts: TuneOpts) !void {
        var need_tvm_cpu = false;
        var need_tvm_gpu = false;
        for (self.config.implementations) |impl| switch (impl) {
            .tvm_cpu => {
                need_tvm_cpu = true;
            },
            .tvm_gpu => {
                need_tvm_gpu = true;
            },
            else => {},
        };

        if (!need_tvm_cpu and !need_tvm_gpu) {
            log.info("no tunable implementations requested, nothing to tune", .{});
            return;
        }

        try tvm_ffi.ensure_loaded(self.allocator, .{});

        if (need_tvm_cpu) {
            for (self.config.shapes) |shape| {
                try self.tune_tvm_shape(shape, .cpu, opts);
            }
        }
        if (need_tvm_gpu) {
            for (self.config.shapes) |shape| {
                try self.tune_tvm_shape(shape, .cuda, opts);
            }
        }
    }

    fn tune_tvm_shape(self: *Harness, shape: Shape, target_kind: tvm_tir.TargetKind, opts: TuneOpts) !void {
        const tvm_cache = try self.cache.subdir(self.io, "tvm", .{});

        const key = tvm_module.matmul_cache_key(target_kind, shape.m, shape.n, shape.k);

        if (opts.retune) {
            const work = try tvm_cache.subdir(self.io, key.slice(), .{ .create = false });
            std.Io.Dir.cwd().deleteTree(self.io, work.path()) catch {};
            log.info("retune: cleared {s}", .{work.path()});
        }

        log.info("tuning: {any} ({s}), {d} trials", .{ shape, @tagName(target_kind), opts.max_trials });

        var ir_mod = try tvm_tir.build_matmul_tir(self.allocator, shape.m, shape.n, shape.k);
        defer ir_mod.deinit();
        var target = try tvm_tir.Target.create(self.allocator, target_kind);
        defer target.deinit();

        const shape_a = try self.allocator.dupe(i64, &[_]i64{ shape.m, shape.k });
        defer self.allocator.free(shape_a);
        const shape_b = try self.allocator.dupe(i64, &[_]i64{ shape.k, shape.n });
        defer self.allocator.free(shape_b);
        const shape_c = try self.allocator.dupe(i64, &[_]i64{ shape.m, shape.n });
        defer self.allocator.free(shape_c);
        const tensor_shapes = try self.allocator.dupe([]const i64, &[_][]const i64{ shape_a, shape_b, shape_c });
        defer self.allocator.free(tensor_shapes);

        const work_cache = try tvm_cache.subdir(self.io, key.slice(), .{});

        try tvm_tune.tune(self.io, self.allocator, ir_mod, target, target_kind, tensor_shapes, .{
            .work_cache = work_cache,
            .max_trials = opts.max_trials,
            .trials_per_iter = opts.trials_per_iter,
        });

        const update = try tvm_module.update_cache_from_work_dir(self.io, self.allocator, tvm_cache, work_cache, key.slice(), target_kind);

        log.info("tuned: {any} ({s}), best candidate {d} ({d:.2} us)", .{
            shape, @tagName(target_kind), update.best_candidate, update.best_time_us,
        });
    }

    /// Dispatch to comptime-typed inner function based on configured dtype.
    fn run_shape(self: *Harness, shape: Shape) !void {
        return switch (self.config.dtype) {
            inline else => |dtype| self.run_shape_typed(dtype.ZigType(), shape),
        };
    }

    fn run_shape_typed(self: *Harness, comptime T: type, shape: Shape) !void {
        log.info("benchmarking shape: {any} ({s})", .{ shape, @typeName(T) });

        const elems_a: usize = @intCast(shape.m * shape.k);
        const elems_b: usize = @intCast(shape.k * shape.n);
        const elems_c: usize = @intCast(shape.m * shape.n);

        const a = try self.allocator.alloc(T, elems_a);
        defer self.allocator.free(a);
        const b = try self.allocator.alloc(T, elems_b);
        defer self.allocator.free(b);
        const c = try self.allocator.alloc(T, elems_c);
        defer self.allocator.free(c);

        fill_random(T, &self.rng, a);
        fill_random(T, &self.rng, b);

        // Reference result via naive gemm for correctness verification
        var reference: ?[]T = null;
        defer if (reference) |ref| self.allocator.free(ref);

        if (self.config.verify_correctness) {
            const ref = try self.allocator.alloc(T, elems_c);
            @memset(ref, @as(T, 0));
            gemm.gemm(T, shape.m, shape.n, shape.k, a, @intCast(shape.k), b, @intCast(shape.n), ref, @intCast(shape.n));
            reference = ref;
        }

        for (self.config.implementations) |impl| {
            const result = try self.benchmark_impl_typed(T, impl, shape, a, b, c, reference);
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
                    self.config.dtype.tolerance(),
                });
            }
        }
    }

    fn benchmark_impl_typed(
        self: *Harness,
        comptime T: type,
        impl: Implementation,
        shape: Shape,
        a: []const T,
        b: []const T,
        c: []T,
        reference: ?[]const T,
    ) !BenchmarkResult {
        // Warmup
        for (0..self.config.warmup_iters) |_| {
            @memset(c, @as(T, 0));
            try self.run_kernel(T, impl, shape, a, b, c);
        }

        // Measurement
        var times = try self.allocator.alloc(f64, self.config.bench_iters);
        defer self.allocator.free(times);

        for (0..self.config.bench_iters) |i| {
            @memset(c, @as(T, 0));

            const start = std.Io.Timestamp.now(self.io, .awake);
            try self.run_kernel(T, impl, shape, a, b, c);
            const elapsed = start.untilNow(self.io, .awake);

            times[i] = @as(f64, @floatFromInt(elapsed.toNanoseconds())) / 1000.0;
        }

        const median_us = stats.median(times);
        const mean_us = stats.mean(times);
        const stddev_us = stats.stddev(times);

        const flops = 2.0 * @as(f64, @floatFromInt(shape.m)) *
            @as(f64, @floatFromInt(shape.n)) *
            @as(f64, @floatFromInt(shape.k));
        const gflops = flops / (median_us * 1000.0);

        var max_err: f64 = 0.0;
        var passed = true;
        if (reference) |ref| {
            max_err = max_abs_error(T, ref, c);
            passed = (max_err <= self.config.dtype.tolerance());
        }

        return BenchmarkResult{
            .impl = impl,
            .shape = shape,
            .dtype = self.config.dtype,
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
        comptime T: type,
        impl: Implementation,
        shape: Shape,
        a: []const T,
        b: []const T,
        c: []T,
    ) !void {
        const lda: usize = @intCast(shape.k);
        const ldb: usize = @intCast(shape.n);
        const ldc: usize = @intCast(shape.n);
        switch (impl) {
            .blas => try gemm.blas_gemm(T, shape.m, shape.n, shape.k, a, lda, b, ldb, c, ldc),
            .zig_naive => gemm.gemm(T, shape.m, shape.n, shape.k, a, lda, b, ldb, c, ldc),
            .tvm_cpu => try self.run_tvm(T, shape, a, b, c, .cpu),
            .tvm_gpu => try self.run_tvm(T, shape, a, b, c, .gpu),
            .xla_cpu => if (comptime zg.build_options.has_mlir)
                try self.run_xla(T, shape, a, b, c, .cpu)
            else
                return error.MlirDisabled,
            .xla_gpu => if (comptime zg.build_options.has_mlir)
                try self.run_xla(T, shape, a, b, c, .gpu)
            else
                return error.MlirDisabled,
        }
    }

    /// Execute TVM matmul (loads pre-tuned module from cache index).
    fn run_tvm(
        self: *Harness,
        comptime T: type,
        shape: Shape,
        a: []const T,
        b: []const T,
        c: []T,
        device: DeviceKind,
    ) !void {
        try tvm_ffi.ensure_loaded(self.allocator, .{});
        const cache_key = try std.fmt.allocPrint(self.allocator, "{d}x{d}x{d}", .{ shape.m, shape.n, shape.k });
        defer self.allocator.free(cache_key);

        const mem_cache = switch (device) {
            .cpu => &self.tvm_cpu_cache,
            .gpu => &self.tvm_gpu_cache,
        };

        const tuned = if (mem_cache.get(cache_key)) |module|
            module
        else blk: {
            const target_kind: tvm_tir.TargetKind = switch (device) {
                .cpu => .cpu,
                .gpu => .cuda,
            };
            const tvm_cache = try self.cache.subdir(self.io, "tvm", .{});

            const key = tvm_module.matmul_cache_key(target_kind, shape.m, shape.n, shape.k);

            const module = try self.allocator.create(tvm_module.TunedModule);
            errdefer self.allocator.destroy(module);

            module.* = try tvm_module.load_cached(self.io, self.allocator, tvm_cache, key.slice(), target_kind) orelse
                return error.NoTuningRecords;

            const owned_key = try self.allocator.dupe(u8, cache_key);
            try mem_cache.put(owned_key, module);
            break :blk module;
        };

        try tvm_adapter.execute_with_module(T, self.allocator, tuned, shape.m, shape.n, shape.k, a, b, c, device);
    }

    /// Execute XLA/PJRT matmul (compiles on-the-fly, caches per shape).
    fn run_xla(
        self: *Harness,
        comptime T: type,
        shape: Shape,
        a: []const T,
        b: []const T,
        c: []T,
        device: DeviceKind,
    ) !void {
        const ctx = switch (device) {
            .cpu => &self.xla_cpu_ctx,
            .gpu => &self.xla_gpu_ctx,
        };

        if (ctx.* == null) {
            ctx.* = try xla_adapter.XlaContext.init(self.allocator, device);
        }

        try ctx.*.?.execute(T, shape, a, b, c);
    }

    /// Print benchmark results in a human-readable table format.
    pub fn print_results(self: *const Harness) !void {
        var buffer: [8192]u8 = undefined;
        var stdout_writer = std.Io.File.stdout().writer(self.io, &buffer);
        const writer = &stdout_writer.interface;

        const header_sep = "\n" ++ "=" ** 60 ++ "\n";
        try writer.print("{s}Matmul Benchmark Results ({s}){s}", .{ header_sep, @tagName(self.config.dtype), header_sep });

        var seen_shapes = std.AutoHashMap(Shape, void).init(self.allocator);
        defer seen_shapes.deinit();

        for (self.config.shapes) |shape| {
            if (seen_shapes.contains(shape)) continue;
            try seen_shapes.put(shape, {});

            try writer.print("\nShape: {any}\n\n", .{shape});
            try writer.writeAll("Implementation            Median (us)    GFLOP/s    vs Naive\n");
            try writer.writeAll("---------------------------------------------------------\n");

            var naive_gflops: f64 = 1.0;
            for (self.results.items) |result| {
                if (result.shape.m == shape.m and result.shape.n == shape.n and result.shape.k == shape.k and
                    result.impl == .zig_naive)
                {
                    naive_gflops = result.gflops;
                    break;
                }
            }

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

        // Summary
        try writer.writeAll(header_sep ++ "Summary" ++ header_sep);

        if (self.results.items.len > 0) {
            var best_overall: ?BenchmarkResult = null;
            var best_hand_rolled: ?BenchmarkResult = null;

            for (self.results.items) |result| {
                if (!result.passed) continue;

                if (best_overall == null or result.gflops > best_overall.?.gflops) {
                    best_overall = result;
                }

                if (result.impl == .zig_naive and
                    (best_hand_rolled == null or result.gflops > best_hand_rolled.?.gflops))
                {
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

fn fill_random(comptime T: type, rng: *std.Random.DefaultPrng, slice: []T) void {
    for (slice) |*val| {
        val.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
    }
}

fn max_abs_error(comptime T: type, expected: []const T, actual: []const T) f64 {
    std.debug.assert(expected.len == actual.len);
    var result: f64 = 0.0;
    for (expected, actual) |e, a| {
        const err = @abs(@as(f64, @floatCast(e)) - @as(f64, @floatCast(a)));
        result = @max(result, err);
    }
    return result;
}

test "harness: init and deinit" {
    const allocator = std.testing.allocator;

    const cfg = BenchmarkConfig{
        .shapes = &[_]Shape{.{ .m = 4, .n = 4, .k = 4 }},
        .implementations = &[_]Implementation{.zig_naive},
        .bench_iters = 5,
    };

    var harness = try Harness.init(allocator, cfg);
    defer harness.deinit();

    try std.testing.expect(harness.results.items.len == 0);
}
