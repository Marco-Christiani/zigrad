const std = @import("std");
const zg = @import("zigrad");
const demos = @import("demos.zig");
const llama_demo = @import("llama_demo.zig");
const llm_demo = @import("llm_demo.zig");
const main_aot = @import("main_aot.zig");
const llama_model = @import("llama_model.zig");
const cli = @import("cli.zig");

// exports for cli gen step in build
pub const CommandT = cli.CommandT;
pub const setup_cmd = cli.setup_cmd;

pub fn main() !void {
    const gpa = std.heap.smp_allocator;

    // Parse args
    var cmd = cli.parse(gpa) catch |err| {
        if (err == error.HelpShown) return;
        std.log.err("failed to parse arguments: {s}", .{@errorName(err)});
        return err;
    };
    defer cmd.deinit();

    // Extract global opts
    var global_opts = try cli.get_global_opts(&cmd, gpa);
    const dump_pr_ptr = if (global_opts.dump_pr) |*cfg| cfg else null;
    const dump_mlir_ptr = if (global_opts.dump_mlir) |*cfg| cfg else null;
    const dump_kernels = global_opts.dump_kernels;
    const quiet = global_opts.quiet;

    // Commands that dont require PJRT backend
    if (cmd.matchSubCmd("print-pr")) |_| {
        return demos.print_pr(gpa);
    }
    if (cmd.matchSubCmd("tvm-zxpr")) |sub_cmd| {
        const opts = try sub_cmd.to(cli.TvmZxprOpts, .{});
        const sweep = opts.sweep_palettes;
        const palette = if (opts.palette) |p| std.meta.stringToEnum(zg.pr.zxpr.Palette, p) else null;
        return demos.print_tvm_kernelize_pr(gpa, sweep, palette);
    }
    if (cmd.matchSubCmd("tvm-attention-zxpr")) |_| {
        return demos.print_tvm_attention_pr(gpa, false, null);
    }
    if (cmd.matchSubCmd("tvm-dump-symbols")) |_| {
        return demos.dump_tvm_ffi_symbols(gpa);
    }
    if (cmd.matchSubCmd("tvm-check-compiler-load")) |_| {
        try zg.tvm.ffi.ensure_loaded(gpa, .{});
        std.log.info("tvm compiler load check passed", .{});
        return;
    }
    if (cmd.matchSubCmd("tvm-tune")) |sub_cmd| {
        const opts = try sub_cmd.to(cli.TvmTuneOpts, .{});

        const shape = try parse_shape(opts.shape orelse "128x128x128");

        const target_kind: zg.tvm.tir.TargetKind = if (opts.cuda or opts.gpu) .cuda else .cpu;
        const target_suffix: []const u8 = @tagName(target_kind);

        const work_dir = opts.work_dir orelse "artifacts/tvm_cache";
        const base_dir = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ work_dir, target_suffix });
        defer gpa.free(base_dir);

        try zg.tvm.ffi.ensure_loaded(gpa, .{});
        var ir_mod = try zg.tvm.tir.build_matmul_tir(gpa, shape.m, shape.n, shape.k);
        defer ir_mod.deinit();
        var target = try zg.tvm.tir.Target.create(gpa, target_kind);
        defer target.deinit();

        const M_i64: i64 = @intCast(shape.m);
        const N_i64: i64 = @intCast(shape.n);
        const K_i64: i64 = @intCast(shape.k);
        const shape_a = try gpa.dupe(i64, &[_]i64{ M_i64, K_i64 });
        defer gpa.free(shape_a);
        const shape_b = try gpa.dupe(i64, &[_]i64{ K_i64, N_i64 });
        defer gpa.free(shape_b);
        const shape_c = try gpa.dupe(i64, &[_]i64{ M_i64, N_i64 });
        defer gpa.free(shape_c);
        const tensor_shapes = try gpa.dupe([]const i64, &[_][]const i64{ shape_a, shape_b, shape_c });
        defer gpa.free(tensor_shapes);

        const key = try zg.tvm.module.matmul_cache_key(gpa, target_kind, shape.m, shape.n, shape.k);
        defer gpa.free(key);
        const full_work_dir = try zg.tvm.module.ensure_cache_dir(gpa, base_dir, key);
        defer gpa.free(full_work_dir);

        try zg.tvm.tune.tune(gpa, ir_mod, target, target_kind, tensor_shapes, .{
            .work_dir = full_work_dir,
            .max_trials = opts.trials orelse 64,
            .trials_per_iter = opts.trials_per_iter orelse 16,
        });

        const update = try zg.tvm.module.update_cache_from_work_dir(gpa, base_dir, full_work_dir, key, target_kind);
        gpa.free(update.stable_path);
        return;
    }
    if (cmd.matchSubCmd("tvm-run")) |sub_cmd| {
        const opts = try sub_cmd.to(cli.TvmRunOpts, .{});

        const shape = try parse_shape(opts.shape orelse "128x128x128");
        const target_kind: zg.tvm.tir.TargetKind = if (opts.cuda or opts.gpu) .cuda else .cpu;
        const work_dir = opts.work_dir orelse "artifacts/tvm_cache";

        return run_tvm_demo(gpa, shape.m, shape.n, shape.k, target_kind, work_dir);
    }
    if (cmd.matchSubCmd("benchmark")) |sub_cmd| {
        const opts = try sub_cmd.to(cli.BenchmarkOpts, .{});

        // struct to args array for run_benchmark_mode
        var args = std.ArrayList([]const u8).empty;
        defer args.deinit(gpa);

        if (opts.shapes) |s| {
            const arg = try std.fmt.allocPrint(gpa, "--shapes={s}", .{s});
            try args.append(gpa, arg);
        }
        if (opts.impls) |i| {
            const arg = try std.fmt.allocPrint(gpa, "--impls={s}", .{i});
            try args.append(gpa, arg);
        }
        if (opts.warmup) |w| {
            const arg = try std.fmt.allocPrint(gpa, "--warmup={d}", .{w});
            try args.append(gpa, arg);
        }
        if (opts.iters) |i| {
            const arg = try std.fmt.allocPrint(gpa, "--iters={d}", .{i});
            try args.append(gpa, arg);
        }
        if (opts.tvm_cache_dir) |t| {
            const arg = try std.fmt.allocPrint(gpa, "--tvm-cache-dir={s}", .{t});
            try args.append(gpa, arg);
        }

        defer for (args.items) |arg| gpa.free(arg);
        return run_benchmark_mode(gpa, args.items);
    }

    // Commands that require PJRT backend
    const plugin_path = std.process.getEnvVarOwned(gpa, "PJRT_PLUGIN_PATH") catch |err| {
        std.log.err("PJRT_PLUGIN_PATH not set ({s})", .{@errorName(err)});
        return err;
    };
    defer gpa.free(plugin_path);

    var backend = try zg.backend.PjrtBackend.init(gpa, plugin_path);
    defer backend.deinit();

    const devs = try backend.get_devices(gpa);
    defer gpa.free(devs);
    if (devs.len == 0) return error.NoDevices;
    const device = &devs[0];

    if (cmd.matchSubCmd("aot-demo")) |_| {
        return main_aot.run(gpa, &backend, device);
    }
    if (cmd.matchSubCmd("custom-call-neg")) |_| {
        return demos.run_custom_call_negative(gpa, &backend, device, dump_pr_ptr, dump_mlir_ptr);
    }
    if (cmd.matchSubCmd("kernel-provider-demo-pr")) |sub_cmd| {
        const opts = try sub_cmd.to(cli.KernelProviderDemoOpts, .{});
        const provider_list = try parse_provider_kinds(opts.provider orelse "tvm");
        return demos.run_kernel_provider_demo(gpa, &backend, device, dump_pr_ptr, dump_mlir_ptr, provider_list.slice(), .pr);
    }
    if (cmd.matchSubCmd("kernel-provider-demo-mlir")) |sub_cmd| {
        const opts = try sub_cmd.to(cli.KernelProviderDemoOpts, .{});
        const provider_list = try parse_provider_kinds(opts.provider orelse "tvm");
        return demos.run_kernel_provider_demo(gpa, &backend, device, dump_pr_ptr, dump_mlir_ptr, provider_list.slice(), .mlir);
    }
    if (cmd.matchSubCmd("vjp-demo")) |_| {
        return demos.run_vjp_demo(gpa, &backend, device, dump_pr_ptr, dump_mlir_ptr);
    }
    if (cmd.matchSubCmd("train-demo")) |sub_cmd| {
        const opts = try sub_cmd.to(cli.TrainDemoOpts, .{});
        const warmup_steps = opts.warmup orelse 0;
        const steps = opts.steps orelse 8;
        return demos.run_train_demo(gpa, plugin_path, dump_pr_ptr, dump_mlir_ptr, warmup_steps, steps, quiet);
    }
    if (cmd.matchSubCmd("llm-ft-demo")) |sub_cmd| {
        const opts = try sub_cmd.to(cli.TrainDemoOpts, .{});
        const warmup_steps = opts.warmup orelse 0;
        const steps = opts.steps orelse 8;
        return llm_demo.run_llm_ft_demo(gpa, plugin_path, dump_pr_ptr, dump_mlir_ptr, warmup_steps, steps, quiet);
    }
    if (cmd.matchSubCmd("llama-ft-demo-pr")) |sub_cmd| {
        const opts = try sub_cmd.to(cli.LlamaFtDemoOpts, .{});
        const dtype = if (opts.dtype) |d|
            std.meta.stringToEnum(zg.pr.DType, d) orelse return error.InvalidDType
        else
            zg.pr.DType.bf16;

        const kernel_provider = if (opts.kernel_provider) |provider_name|
            std.meta.stringToEnum(llama_demo.LlamaKernelProvider, provider_name) orelse return error.InvalidArgument
        else
            null;

        const cfg = llama_demo.LlamaDemoConfig{
            .train = opts.train,
            .dtype = dtype,
            .seq = opts.seq orelse 4,
            .batch = opts.batch orelse 1,
            .canonical_shapes = opts.canonical_shapes,
            .execute_only = opts.execute_only,
            .kernel_provider = kernel_provider,
        };

        return llama_demo.run_llama_ft_demo(
            gpa,
            plugin_path,
            dump_pr_ptr,
            dump_mlir_ptr,
            opts.warmup orelse 1,
            opts.steps orelse 4,
            quiet,
            .pr,
            cfg,
            dump_kernels,
        );
    }
    if (cmd.matchSubCmd("llama-ft-demo-mlir")) |sub_cmd| {
        const opts = try sub_cmd.to(cli.LlamaFtDemoOpts, .{});
        const dtype = if (opts.dtype) |d|
            std.meta.stringToEnum(zg.pr.DType, d) orelse return error.InvalidDType
        else
            zg.pr.DType.bf16;

        const kernel_provider = if (opts.kernel_provider) |provider_name|
            std.meta.stringToEnum(llama_demo.LlamaKernelProvider, provider_name) orelse return error.InvalidArgument
        else
            null;

        const cfg = llama_demo.LlamaDemoConfig{
            .train = opts.train,
            .dtype = dtype,
            .seq = opts.seq orelse 4,
            .batch = opts.batch orelse 1,
            .canonical_shapes = opts.canonical_shapes,
            .execute_only = opts.execute_only,
            .kernel_provider = kernel_provider,
        };

        return llama_demo.run_llama_ft_demo(
            gpa,
            plugin_path,
            dump_pr_ptr,
            dump_mlir_ptr,
            opts.warmup orelse 1,
            opts.steps orelse 4,
            quiet,
            .mlir,
            cfg,
            dump_kernels,
        );
    }
    if (cmd.matchSubCmd("jit-cache-save")) |sub_cmd| {
        const opts = try sub_cmd.to(cli.JitCacheOpts, .{});

        var program = try zg.frontend.build_demo_program(gpa);
        defer program.deinit();

        const mlir_bytes = try zg.lower.lower_program_to_mlir(gpa, &program, "main", .mlir_bytecode);
        defer gpa.free(mlir_bytes);

        const serialized = try backend.compile_serialized(device, mlir_bytes, true, .{});
        defer gpa.free(serialized);

        try demos.write_bytes_to_path(opts.path, serialized);
        std.log.info("wrote PJRT JIT cache artifact: {d} bytes -> {s}", .{ serialized.len, opts.path });
        return;
    }
    if (cmd.matchSubCmd("jit-cache-run")) |sub_cmd| {
        const opts = try sub_cmd.to(cli.JitCacheOpts, .{});

        const serialized = try demos.read_bytes_from_path(gpa, opts.path);
        defer gpa.free(serialized);

        var exe = try backend.load_serialized_executable(serialized, null);
        defer backend.deinit_executable(&exe);

        return demos.run_demo_executable(gpa, &backend, device, &exe);
    }

    // default to running demo program
    var program = try zg.frontend.build_demo_program(gpa);
    defer program.deinit();

    const lower_encoding: zg.pipeline.MlirEncoding = if (global_opts.dump_mlir != null) .text else .bytecode;
    var exe = try demos.compile_program(&backend, gpa, &program, device, .{
        .encoding = lower_encoding,
        .entry_name = "main",
    }, dump_pr_ptr, dump_mlir_ptr, null);
    defer backend.deinit_executable(&exe);

    return demos.run_demo_executable(gpa, &backend, device, &exe);
}

fn run_tvm_demo(
    gpa: std.mem.Allocator,
    M: usize,
    N: usize,
    K: usize,
    target_kind: zg.tvm.tir.TargetKind,
    base_work_dir: []const u8,
) !void {
    const tvm_runtime = zg.tvm.runtime;
    const dlpack_mod = zg.tvm.dlpack;

    const target_suffix: []const u8 = @tagName(target_kind);
    const base_dir = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ base_work_dir, target_suffix });
    defer gpa.free(base_dir);
    const key = try zg.tvm.module.matmul_cache_key(gpa, target_kind, M, N, K);
    defer gpa.free(key);

    try zg.tvm.ffi.ensure_loaded(gpa, .{});
    const tuned = try zg.tvm.module.load_cached(gpa, base_dir, key, target_kind) orelse return error.NoTuningRecords;
    var tuned_mut = tuned;
    defer tuned_mut.deinit();

    const a = try gpa.alloc(f32, M * K);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, K * N);
    defer gpa.free(b);
    const result = try gpa.alloc(f32, M * N);
    defer gpa.free(result);

    for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) * 0.1;
    for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) * 0.1;
    @memset(result, 0);

    var timer = try std.time.Timer.start();

    switch (target_kind) {
        .cpu => {
            var shape_a = [_]i64{ @intCast(M), @intCast(K) };
            var shape_b = [_]i64{ @intCast(K), @intCast(N) };
            var shape_c = [_]i64{ @intCast(M), @intCast(N) };

            var dl_a = dlpack_mod.ManagedTensor.borrowing(
                dlpack_mod.Tensor.init_contiguous(f32, @constCast(a), &shape_a),
            );
            var dl_b = dlpack_mod.ManagedTensor.borrowing(
                dlpack_mod.Tensor.init_contiguous(f32, @constCast(b), &shape_b),
            );
            var dl_c = dlpack_mod.ManagedTensor.borrowing(
                dlpack_mod.Tensor.init_contiguous(f32, result, &shape_c),
            );

            var t_a = try tvm_runtime.Tensor.from_dlpack(&dl_a);
            defer t_a.deinit();
            var t_b = try tvm_runtime.Tensor.from_dlpack(&dl_b);
            defer t_b.deinit();
            var t_c = try tvm_runtime.Tensor.from_dlpack(&dl_c);
            defer t_c.deinit();

            try tuned.invoke(gpa, &.{
                t_a.as_value(), t_b.as_value(), t_c.as_value(),
            });
        },
        .cuda => {
            var shape_a = [_]i64{ @intCast(M), @intCast(K) };
            var shape_b = [_]i64{ @intCast(K), @intCast(N) };
            var shape_c = [_]i64{ @intCast(M), @intCast(N) };

            const dl_a = try dlpack_mod.ManagedTensor.heap_borrowing(
                gpa,
                dlpack_mod.Tensor.init_contiguous(f32, @constCast(a), &shape_a),
            );
            const dl_b = try dlpack_mod.ManagedTensor.heap_borrowing(
                gpa,
                dlpack_mod.Tensor.init_contiguous(f32, @constCast(b), &shape_b),
            );
            const dl_c = try dlpack_mod.ManagedTensor.heap_borrowing(
                gpa,
                dlpack_mod.Tensor.init_contiguous(f32, result, &shape_c),
            );

            var t_a = try tvm_runtime.Tensor.from_dlpack(dl_a);
            defer t_a.deinit();
            var t_b = try tvm_runtime.Tensor.from_dlpack(dl_b);
            defer t_b.deinit();
            var t_c = try tvm_runtime.Tensor.from_dlpack(dl_c);
            defer t_c.deinit();

            try tuned.invoke(gpa, &.{
                t_a.as_value(), t_b.as_value(), t_c.as_value(),
            });
        },
    }

    const elapsed = timer.read();
    const ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    std.log.info("TVM matmul {d}x{d}x{d} ({s}): {d:.3} ms", .{ M, N, K, @tagName(target_kind), ms });

    const expected = @as(f32, @floatFromInt(K - 1)) * @as(f32, @floatFromInt(K - 1)) * 0.01 * @as(f32, @floatFromInt(K)) / 2.0;
    std.log.info("result[M-1,N-1] = {d:.6} (expected ~{d:.6})", .{ result[M * N - 1], expected });
}

fn run_benchmark_mode(gpa: std.mem.Allocator, args: []const []const u8) !void {
    var shapes = std.ArrayList(zg.benchmark.Shape).empty;
    defer shapes.deinit(gpa);
    var impls = std.ArrayList(zg.benchmark.Implementation).empty;
    defer impls.deinit(gpa);
    var warmup: usize = 10;
    var iters: usize = 100;
    var tvm_cache_dir: []const u8 = "artifacts/tvm_cache";

    for (args) |arg| {
        if (std.mem.startsWith(u8, arg, "--shapes=")) {
            const value = arg["--shapes=".len..];
            var shape_strs = std.mem.splitScalar(u8, value, ',');
            while (shape_strs.next()) |s| {
                try shapes.append(gpa, try parse_shape(s));
            }
            continue;
        }
        if (std.mem.startsWith(u8, arg, "--impls=")) {
            const value = arg["--impls=".len..];
            var impl_strs = std.mem.splitScalar(u8, value, ',');
            while (impl_strs.next()) |s| {
                if (std.meta.stringToEnum(zg.benchmark.Implementation, s)) |impl| {
                    try impls.append(gpa, impl);
                } else {
                    std.log.err("unknown implementation: {s}", .{s});
                    return error.InvalidArguments;
                }
            }
            continue;
        }
        if (std.mem.startsWith(u8, arg, "--warmup=")) {
            const value = arg["--warmup=".len..];
            warmup = try std.fmt.parseInt(usize, value, 10);
            continue;
        }
        if (std.mem.startsWith(u8, arg, "--iters=")) {
            const value = arg["--iters=".len..];
            iters = try std.fmt.parseInt(usize, value, 10);
            continue;
        }
        if (std.mem.startsWith(u8, arg, "--tvm-cache-dir=")) {
            tvm_cache_dir = arg["--tvm-cache-dir=".len..];
            continue;
        }
        std.log.err("unknown argument: {s}", .{arg});
        return error.InvalidArguments;
    }

    if (shapes.items.len == 0) try shapes.append(gpa, .{ .m = 128, .n = 128, .k = 128 });
    if (impls.items.len == 0) try impls.append(gpa, .zig_naive);

    const cfg = zg.benchmark.BenchmarkConfig{
        .shapes = shapes.items,
        .implementations = impls.items,
        .warmup_iters = warmup,
        .bench_iters = iters,
    };

    var harness = try zg.benchmark.Harness.init(gpa, cfg, tvm_cache_dir);
    defer harness.deinit();

    try harness.run();
    try harness.print_results();
}

const ProviderKindList = struct {
    buf: [2]demos.KernelProviderDemoKind,
    len: usize,

    fn slice(self: *const ProviderKindList) []const demos.KernelProviderDemoKind {
        return self.buf[0..self.len];
    }
};

fn parse_provider_kinds(s: []const u8) !ProviderKindList {
    var result: ProviderKindList = .{ .buf = undefined, .len = 0 };
    var it = std.mem.splitScalar(u8, s, ',');
    while (it.next()) |token| {
        const trimmed = std.mem.trim(u8, token, " ");
        if (result.len >= result.buf.len) return error.TooManyProviders;
        result.buf[result.len] = std.meta.stringToEnum(demos.KernelProviderDemoKind, trimmed) orelse return error.InvalidArgument;
        result.len += 1;
    }
    if (result.len == 0) return error.InvalidArgument;
    return result;
}

fn parse_shape(s: []const u8) !zg.benchmark.Shape {
    var parts = std.mem.splitScalar(u8, s, 'x');
    const m = try std.fmt.parseInt(usize, parts.next() orelse return error.InvalidShape, 10);
    const n = try std.fmt.parseInt(usize, parts.next() orelse return error.InvalidShape, 10);
    const k = try std.fmt.parseInt(usize, parts.next() orelse return error.InvalidShape, 10);
    return .{ .m = m, .n = n, .k = k };
}
