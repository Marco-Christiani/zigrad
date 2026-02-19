const std = @import("std");
const zg = @import("zigrad");
const demos = @import("demos.zig");
const llama_demo = @import("llama_demo.zig");
const llm_demo = @import("llm_demo.zig");
const main_aot = @import("main_aot.zig");
const llama_model = @import("llama_model.zig");

pub fn main() !void {
    // var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    // defer _ = gpa_state.deinit();
    // const gpa = gpa_state.allocator();
    // var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    // defer arena.deinit();
    // const gpa = arena.allocator();
    const gpa = std.heap.smp_allocator;

    var arg_it = std.process.args();
    _ = arg_it.next(); // argv0
    var mode: ?[]const u8 = null;
    var mode_args = std.ArrayList([]const u8).empty;
    defer mode_args.deinit(gpa);
    var dump_pr_cfg: zg.pipeline.DumpConfig = .{};
    var dump_mlir_cfg: zg.pipeline.DumpConfig = .{};
    var have_dump_pr = false;
    var have_dump_mlir = false;
    var quiet = false;

    while (arg_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try print_usage();
            return;
        }
        if (mode != null) {
            try mode_args.append(gpa, arg);
            continue;
        }
        if (std.mem.eql(u8, arg, "--dump-pr")) {
            dump_pr_cfg = .{ .target = .stdout };
            have_dump_pr = true;
            continue;
        }
        if (std.mem.startsWith(u8, arg, "--dump-pr=")) {
            const path = arg["--dump-pr=".len..];
            if (path.len == 0) {
                try print_usage();
                return error.InvalidArguments;
            }
            dump_pr_cfg = .{ .target = .file, .path = path };
            have_dump_pr = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--dump-mlir")) {
            dump_mlir_cfg = .{ .target = .stdout };
            have_dump_mlir = true;
            continue;
        }
        if (std.mem.startsWith(u8, arg, "--dump-mlir=")) {
            const path = arg["--dump-mlir=".len..];
            if (path.len == 0) {
                try print_usage();
                return error.InvalidArguments;
            }
            dump_mlir_cfg = .{ .target = .file, .path = path };
            have_dump_mlir = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--quiet")) {
            quiet = true;
            continue;
        }
        if (std.mem.startsWith(u8, arg, "--")) {
            try print_usage();
            return error.InvalidArguments;
        }
        mode = arg;
    }

    if (mode) |m| {
        if (std.mem.eql(u8, m, "print-pr")) {
            if (mode_args.items.len != 0 or have_dump_pr or have_dump_mlir) {
                try print_usage();
                return error.InvalidArguments;
            }
            return demos.print_pr(gpa);
        }
        if (std.mem.eql(u8, m, "tvm-zxpr")) {
            if (have_dump_pr or have_dump_mlir) {
                try print_usage();
                return error.InvalidArguments;
            }
            var sweep_palettes = false;
            var palette: ?zg.pr.zxpr.Palette = null;
            for (mode_args.items) |arg| {
                if (std.mem.eql(u8, arg, "--sweep-palettes")) {
                    sweep_palettes = true;
                    continue;
                }
                if (std.mem.startsWith(u8, arg, "--palette=")) {
                    const value = arg["--palette=".len..];
                    palette = parse_zxpr_palette(value) orelse {
                        try print_usage();
                        return error.InvalidArguments;
                    };
                    continue;
                }
                try print_usage();
                return error.InvalidArguments;
            }
            if (sweep_palettes and palette != null) {
                try print_usage();
                return error.InvalidArguments;
            }
            return demos.print_tvm_kernelize_pr(gpa, sweep_palettes, palette);
        }
        if (std.mem.eql(u8, m, "tvm-attention-zxpr")) {
            if (have_dump_pr or have_dump_mlir) {
                try print_usage();
                return error.InvalidArguments;
            }
            return demos.print_tvm_attention_pr(gpa, false, null);
        }
        if (std.mem.eql(u8, m, "tvm-dump-symbols")) {
            if (!zg.build_options.enable_tvm) return error.TvmNotEnabled;
            if (mode_args.items.len != 0 or have_dump_pr or have_dump_mlir) {
                try print_usage();
                return error.InvalidArguments;
            }
            return demos.dump_tvm_ffi_symbols(gpa);
        }
        if (std.mem.eql(u8, m, "tvm-tune")) {
            if (!zg.build_options.enable_tvm) return error.TvmNotEnabled;
            if (have_dump_pr or have_dump_mlir) {
                try print_usage();
                return error.InvalidArguments;
            }
            var shape: struct { M: usize = 128, N: usize = 128, K: usize = 128 } = .{};
            var target_kind: zg.tvm.tir.TargetKind = .cpu;
            var work_dir: []const u8 = "artifacts/tvm_cache";
            var max_trials: u32 = 64;
            var trials_per_iter: u32 = 16;

            for (mode_args.items) |arg| {
                if (std.mem.startsWith(u8, arg, "--shape=")) {
                    const value = arg["--shape=".len..];
                    // parse MxNxK format
                    var parts = std.mem.splitScalar(u8, value, 'x');
                    shape.M = std.fmt.parseInt(usize, parts.next() orelse {
                        try print_usage();
                        return error.InvalidArguments;
                    }, 10) catch {
                        try print_usage();
                        return error.InvalidArguments;
                    };
                    shape.N = std.fmt.parseInt(usize, parts.next() orelse {
                        try print_usage();
                        return error.InvalidArguments;
                    }, 10) catch {
                        try print_usage();
                        return error.InvalidArguments;
                    };
                    shape.K = std.fmt.parseInt(usize, parts.next() orelse {
                        try print_usage();
                        return error.InvalidArguments;
                    }, 10) catch {
                        try print_usage();
                        return error.InvalidArguments;
                    };
                    continue;
                }
                if (std.mem.startsWith(u8, arg, "--trials=")) {
                    const value = arg["--trials=".len..];
                    max_trials = std.fmt.parseInt(u32, value, 10) catch {
                        try print_usage();
                        return error.InvalidArguments;
                    };
                    continue;
                }
                if (std.mem.startsWith(u8, arg, "--trials-per-iter=")) {
                    const value = arg["--trials-per-iter=".len..];
                    trials_per_iter = std.fmt.parseInt(u32, value, 10) catch {
                        try print_usage();
                        return error.InvalidArguments;
                    };
                    continue;
                }
                if (std.mem.startsWith(u8, arg, "--work-dir=")) {
                    work_dir = arg["--work-dir=".len..];
                    continue;
                }
                if (std.mem.eql(u8, arg, "--cuda") or std.mem.eql(u8, arg, "--gpu")) {
                    target_kind = .cuda;
                    continue;
                }
                if (std.mem.eql(u8, arg, "--cpu")) {
                    target_kind = .cpu;
                    continue;
                }
                try print_usage();
                return error.InvalidArguments;
            }

            const target_suffix: []const u8 = switch (target_kind) {
                .cpu => "cpu",
                .cuda => "cuda",
            };
            const full_work_dir = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ work_dir, target_suffix });
            defer gpa.free(full_work_dir);

            try zg.tvm.ffi.ensure_loaded(gpa);
            var ir_mod = try zg.tvm.tir.build_matmul_tir(gpa, shape.M, shape.N, shape.K);
            defer ir_mod.deinit();
            var target = try zg.tvm.tir.Target.create(gpa, target_kind);
            defer target.deinit();

            // A[M,K], B[K,N], C[M,N]
            const M_i64: i64 = @intCast(shape.M);
            const N_i64: i64 = @intCast(shape.N);
            const K_i64: i64 = @intCast(shape.K);
            const shape_a = try gpa.dupe(i64, &[_]i64{ M_i64, K_i64 });
            defer gpa.free(shape_a);
            const shape_b = try gpa.dupe(i64, &[_]i64{ K_i64, N_i64 });
            defer gpa.free(shape_b);
            const shape_c = try gpa.dupe(i64, &[_]i64{ M_i64, N_i64 });
            defer gpa.free(shape_c);
            const tensor_shapes = try gpa.dupe([]const i64, &[_][]const i64{ shape_a, shape_b, shape_c });
            defer gpa.free(tensor_shapes);

            return zg.tvm.tune.tune(gpa, ir_mod, target, target_kind, tensor_shapes, .{
                .work_dir = full_work_dir,
                .max_trials = max_trials,
                .trials_per_iter = trials_per_iter,
            });
        }
        if (std.mem.eql(u8, m, "tvm-run")) {
            if (!zg.build_options.enable_tvm) return error.TvmNotEnabled;
            if (have_dump_pr or have_dump_mlir) {
                try print_usage();
                return error.InvalidArguments;
            }
            var shape: struct { M: usize = 128, N: usize = 128, K: usize = 128 } = .{};
            var target_kind: zg.tvm.tir.TargetKind = .cpu;
            var work_dir: []const u8 = "artifacts/tvm_cache";

            for (mode_args.items) |arg| {
                if (std.mem.startsWith(u8, arg, "--shape=")) {
                    const value = arg["--shape=".len..];
                    var parts = std.mem.splitScalar(u8, value, 'x');
                    shape.M = std.fmt.parseInt(usize, parts.next() orelse {
                        try print_usage();
                        return error.InvalidArguments;
                    }, 10) catch {
                        try print_usage();
                        return error.InvalidArguments;
                    };
                    shape.N = std.fmt.parseInt(usize, parts.next() orelse {
                        try print_usage();
                        return error.InvalidArguments;
                    }, 10) catch {
                        try print_usage();
                        return error.InvalidArguments;
                    };
                    shape.K = std.fmt.parseInt(usize, parts.next() orelse {
                        try print_usage();
                        return error.InvalidArguments;
                    }, 10) catch {
                        try print_usage();
                        return error.InvalidArguments;
                    };
                    continue;
                }
                if (std.mem.startsWith(u8, arg, "--work-dir=")) {
                    work_dir = arg["--work-dir=".len..];
                    continue;
                }
                if (std.mem.eql(u8, arg, "--cuda") or std.mem.eql(u8, arg, "--gpu")) {
                    target_kind = .cuda;
                    continue;
                }
                if (std.mem.eql(u8, arg, "--cpu")) {
                    target_kind = .cpu;
                    continue;
                }
                try print_usage();
                return error.InvalidArguments;
            }
            return run_tvm_demo(gpa, shape.M, shape.N, shape.K, target_kind, work_dir);
        }
        if (std.mem.eql(u8, m, "benchmark")) {
            if (!zg.build_options.enable_tvm) return error.TvmNotEnabled;
            if (have_dump_pr or have_dump_mlir) {
                try print_usage();
                return error.InvalidArguments;
            }
            return run_benchmark_mode(gpa, mode_args.items);
        }
    }

    const plugin_path = std.process.getEnvVarOwned(gpa, "PJRT_PLUGIN_PATH") catch |err| {
        std.log.err("PJRT_PLUGIN_PATH not set ({s})", .{@errorName(err)});
        return err;
    };
    defer gpa.free(plugin_path);

    // Initialize unified PJRT backend
    var backend = try zg.backend.PjrtBackend.init(gpa, plugin_path);
    defer backend.deinit();

    const devs = try backend.get_devices(gpa);
    defer gpa.free(devs);
    if (devs.len == 0) return error.NoDevices;
    const device = &devs[0];

    if (mode) |m| {
        if (std.mem.eql(u8, m, "aot-demo")) {
            if (mode_args.items.len != 0 or have_dump_pr or have_dump_mlir) {
                try print_usage();
                return error.InvalidArguments;
            }
            return main_aot.run(gpa, &backend, device);
        }
        if (std.mem.eql(u8, m, "custom-call-neg")) {
            if (mode_args.items.len != 0) {
                try print_usage();
                return error.InvalidArguments;
            }
            return demos.run_custom_call_negative(gpa, &backend, device, if (have_dump_pr) &dump_pr_cfg else null, if (have_dump_mlir) &dump_mlir_cfg else null);
        }
        if (std.mem.eql(u8, m, "vjp-demo")) {
            if (mode_args.items.len != 0) {
                try print_usage();
                return error.InvalidArguments;
            }
            return demos.run_vjp_demo(gpa, &backend, device, if (have_dump_pr) &dump_pr_cfg else null, if (have_dump_mlir) &dump_mlir_cfg else null);
        }
        if (std.mem.eql(u8, m, "train-demo")) {
            if (mode_args.items.len > 2) {
                try print_usage();
                return error.InvalidArguments;
            }

            const warmup_steps: usize = if (mode_args.items.len >= 1)
                std.fmt.parseInt(usize, mode_args.items[0], 10) catch {
                    try print_usage();
                    return error.InvalidArguments;
                }
            else
                0;

            const steps: usize = if (mode_args.items.len == 2)
                std.fmt.parseInt(usize, mode_args.items[1], 10) catch {
                    try print_usage();
                    return error.InvalidArguments;
                }
            else
                8;
            return demos.run_train_demo(
                gpa,
                plugin_path,
                if (have_dump_pr) &dump_pr_cfg else null,
                if (have_dump_mlir) &dump_mlir_cfg else null,
                warmup_steps,
                steps,
                quiet,
            );
        }
        if (std.mem.eql(u8, m, "llm-ft-demo")) {
            if (mode_args.items.len > 2) {
                try print_usage();
                return error.InvalidArguments;
            }

            const warmup_steps: usize = if (mode_args.items.len >= 1)
                std.fmt.parseInt(usize, mode_args.items[0], 10) catch {
                    try print_usage();
                    return error.InvalidArguments;
                }
            else
                0;

            const steps: usize = if (mode_args.items.len == 2)
                std.fmt.parseInt(usize, mode_args.items[1], 10) catch {
                    try print_usage();
                    return error.InvalidArguments;
                }
            else
                8;

            return llm_demo.run_llm_ft_demo(
                gpa,
                plugin_path,
                if (have_dump_pr) &dump_pr_cfg else null,
                if (have_dump_mlir) &dump_mlir_cfg else null,
                warmup_steps,
                steps,
                quiet,
            );
        }
        if (std.mem.eql(u8, m, "llama-ft-demo")) {
            var warmup_steps: usize = 1;
            var steps: usize = 4;
            var seq: usize = 4;
            var batch: usize = 1;
            var pos_index: usize = 0;
            var train_mode = false;
            var model_dtype: ?zg.pr.DType = null;
            var execute_only = false;

            for (mode_args.items) |arg| {
                if (std.mem.eql(u8, arg, "--train")) {
                    train_mode = true;
                    continue;
                }
                if (std.mem.startsWith(u8, arg, "--seq=")) {
                    const seq_str = arg["--seq=".len..];
                    seq = std.fmt.parseInt(usize, seq_str, 10) catch {
                        try print_usage();
                        return error.InvalidArguments;
                    };
                    continue;
                }
                if (std.mem.startsWith(u8, arg, "--batch=")) {
                    const batch_str = arg["--batch=".len..];
                    batch = std.fmt.parseInt(usize, batch_str, 10) catch {
                        try print_usage();
                        return error.InvalidArguments;
                    };
                    continue;
                }
                if (std.mem.startsWith(u8, arg, "--dtype=")) {
                    const dtype_str = arg["--dtype=".len..];
                    if (std.mem.eql(u8, dtype_str, "bf16")) {
                        model_dtype = .bf16;
                    } else if (std.mem.eql(u8, dtype_str, "f32")) {
                        model_dtype = .f32;
                    } else {
                        try print_usage();
                        return error.InvalidArguments;
                    }
                    continue;
                }
                if (std.mem.eql(u8, arg, "--execute-only")) {
                    execute_only = true;
                    continue;
                }

                const value = std.fmt.parseInt(usize, arg, 10) catch {
                    try print_usage();
                    return error.InvalidArguments;
                };
                if (pos_index == 0) {
                    warmup_steps = value;
                } else if (pos_index == 1) {
                    steps = value;
                } else {
                    try print_usage();
                    return error.InvalidArguments;
                }
                pos_index += 1;
            }

            const dtype = model_dtype orelse .bf16;
            const cfg = llama_demo.LlamaDemoConfig{
                .train = train_mode,
                .dtype = dtype,
                .seq = seq,
                .batch = batch,
                .execute_only = execute_only,
            };

            return llama_demo.run_llama_ft_demo(
                gpa,
                plugin_path,
                if (have_dump_pr) &dump_pr_cfg else null,
                if (have_dump_mlir) &dump_mlir_cfg else null,
                warmup_steps,
                steps,
                quiet,
                cfg,
            );
        }
        if (std.mem.eql(u8, m, "jit-cache-save")) {
            if (have_dump_pr or have_dump_mlir) {
                try print_usage();
                return error.InvalidArguments;
            }
            const path: ?[]const u8 = if (mode_args.items.len == 1) mode_args.items[0] else null;
            if (path == null) {
                try print_usage();
                return error.InvalidArguments;
            }

            var program = try zg.frontend.build_demo_program(gpa);
            defer program.deinit();
            // Lower PR -> MLIR
            const mlir_bytes = try zg.lower.lower_program_to_mlir(gpa, &program, "main", .mlir_bytecode);
            defer gpa.free(mlir_bytes);

            // Compile and serialize
            const serialized = try backend.compile_serialized(device, mlir_bytes, true, .{});
            defer gpa.free(serialized);

            try demos.write_bytes_to_path(path.?, serialized);
            std.log.info("wrote PJRT JIT cache artifact: {d} bytes -> {s}", .{ serialized.len, path.? });
            return;
        }
        if (std.mem.eql(u8, m, "jit-cache-run")) {
            if (have_dump_pr or have_dump_mlir) {
                try print_usage();
                return error.InvalidArguments;
            }
            const path: ?[]const u8 = if (mode_args.items.len == 1) mode_args.items[0] else null;
            if (path == null) {
                try print_usage();
                return error.InvalidArguments;
            }

            const serialized = try demos.read_bytes_from_path(gpa, path.?);
            defer gpa.free(serialized);

            var exe = try backend.load_serialized_executable(serialized, null);
            defer backend.deinit_executable(&exe);

            return demos.run_demo_executable(gpa, &backend, device, &exe);
        }
        std.log.err("unknown mode: {s}", .{m});
        try print_usage();
        return error.InvalidArguments;
    }

    var program = try zg.frontend.build_demo_program(gpa);
    defer program.deinit();

    const lower_encoding: zg.pipeline.MlirEncoding = if (have_dump_mlir) .text else .bytecode;
    var exe = try demos.compile_program(&backend, gpa, &program, device, .{
        .encoding = lower_encoding,
        .entry_name = "main",
    }, if (have_dump_pr) &dump_pr_cfg else null, if (have_dump_mlir) &dump_mlir_cfg else null);
    defer backend.deinit_executable(&exe);

    return demos.run_demo_executable(gpa, &backend, device, &exe);
}

fn print_usage() !void {
    var buffer: [8192]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&buffer);
    const out = &stdout_writer.interface;

    try out.writeAll(
        \\usage: zigrad [global options] [mode [args]]
        \\
        \\global options (must appear before mode):
        \\  -h, --help          show this help
        \\  --dump-pr           print PR (zxpr) to stdout (default/custom-call-neg/vjp-demo)
        \\  --dump-pr=PATH      write PR (zxpr) to PATH (default/custom-call-neg/vjp-demo)
        \\  --dump-mlir         print MLIR (text) to stdout (default/custom-call-neg/vjp-demo)
        \\  --dump-mlir=PATH    write MLIR (text) to PATH (default/custom-call-neg/vjp-demo)
        \\  --quiet             reduce output (train-demo/llm-ft-demo)
        \\
        \\modes:
        \\  print-pr                     prints the PR for the demo program
        \\  tvm-zxpr [--sweep-palettes] [--palette=<name>]  prints a kernelized TVM region in zxpr
        \\  tvm-attention-zxpr [--sweep-palettes] [--palette=<name>]  prints attention pattern with TVM annotation
        \\  tvm-dump-symbols             enumerates all available TVM FFI functions (requires -Dtvm)
        \\  tvm-tune [options]           runs TVM MetaSchedule autotuning on matmul (requires -Dtvm)
        \\      --shape=MxNxK            matmul dimensions (default: 128x128x128)
        \\      --trials=N               max tuning trials (default: 64)
        \\      --trials-per-iter=N      batch size per iteration (default: 16)
        \\      --work-dir=PATH          tuning cache directory (default: artifacts/tvm_cache)
        \\      --cuda/--gpu             tune for CUDA target
        \\      --cpu                    tune for CPU target (default)
        \\  tvm-run [options]            loads and runs a tuned TVM matmul (requires -Dtvm + prior tuning)
        \\      --shape=MxNxK            matmul dimensions (must match tuned shape)
        \\      --work-dir=PATH          tuning cache directory (default: artifacts/tvm_cache)
        \\      --cuda/--gpu             run on CUDA target
        \\      --cpu                    run on CPU target (default)
        \\  aot-demo                     runs the AOT compile+load demo
        \\  custom-call-neg              expects missing custom call handler
        \\  vjp-demo                     runs the reverse-mode demo
        \\  train-demo [warmup] [steps]  runs the frontend training demo
        \\  llm-ft-demo [warmup] [steps] runs a tiny LLM fine-tune demo
        \\  llama-ft-demo [warmup] [steps] [--train] [--dtype=bf16|f32] [--seq=N] [--batch=N] [--canonical-shapes] [--canonical-qkv] [--canonical-o] [--canonical-mlp] [--execute-only] runs a tiny Llama fine-tune demo
        \\  jit-cache-save <path>        writes PJRT JIT cache artifact
        \\  jit-cache-run <path>         loads and runs PJRT JIT cache artifact
        \\  benchmark [options]          runs matmul performance benchmarks
        \\      --shapes=MxNxK[,...]     comma-separated list of shapes (default: 128x128x128)
        \\      --impls=impl1[,...]      implementations to test: zig_naive, tvm_cpu, xla_cpu, etc (default: zig_naive)
        \\      --warmup=N               warmup iterations (default: 10)
        \\      --iters=N                benchmark iterations (default: 100)
        \\      --tvm-cache-dir=PATH     TVM module cache directory (default: artifacts/tvm_cache)
        \\
    );

    try out.flush();
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

    const target_suffix: []const u8 = switch (target_kind) {
        .cpu => "cpu",
        .cuda => "cuda",
    };
    const work_dir = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ base_work_dir, target_suffix });
    defer gpa.free(work_dir);

    // Load tuned module
    try zg.tvm.ffi.ensure_loaded(gpa);
    var tuned = try zg.tvm.module.load(gpa, .{ .work_dir = work_dir });
    defer tuned.deinit();

    // Allocate and fill inputs
    const a = try gpa.alloc(f32, M * K);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, K * N);
    defer gpa.free(b);
    const result = try gpa.alloc(f32, M * N);
    defer gpa.free(result);

    for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) * 0.1;
    for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) * 0.1;
    @memset(result, 0);

    // Execute
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

            var t_a = try tvm_runtime.Tensor.allocate(gpa, @constCast(a), &shape_a, .cuda);
            defer t_a.deinit();
            var t_b = try tvm_runtime.Tensor.allocate(gpa, @constCast(b), &shape_b, .cuda);
            defer t_b.deinit();

            const c_init = try gpa.alloc(f32, M * N);
            defer gpa.free(c_init);
            @memset(c_init, 0);
            var t_c = try tvm_runtime.Tensor.allocate(gpa, c_init, &shape_c, .cuda);
            defer t_c.deinit();

            try tuned.invoke(gpa, &.{
                t_a.as_value(), t_b.as_value(), t_c.as_value(),
            });

            try t_c.copy_to_host(gpa, result);
        },
    }

    const elapsed_ns = timer.read();
    const elapsed_us = @as(f64, @floatFromInt(elapsed_ns)) / 1000.0;

    // Verify with naive reference
    const ref = try gpa.alloc(f32, M * N);
    defer gpa.free(ref);
    @memset(ref, 0);
    for (0..M) |i| {
        for (0..K) |kk| {
            for (0..N) |j| {
                ref[i * N + j] += a[i * K + kk] * b[kk * N + j];
            }
        }
    }

    var max_err: f32 = 0;
    for (result, ref) |got, expected| {
        const diff = @abs(got - expected);
        if (diff > max_err) max_err = diff;
    }

    const pass = max_err < 1e-3;

    var buf: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&buf);
    const out = &stdout_writer.interface;
    try out.print("tvm-run: {d}x{d}x{d} ({s}) candidate={d} tune={d:.1}us exec={d:.1}us max_err={e:.3} {s}\n", .{
        M, N, K, target_suffix,
        tuned.best_candidate, tuned.best_time_us, elapsed_us, max_err,
        if (pass) "PASS" else "FAIL",
    });
    try out.flush();

    if (!pass) return error.VerificationFailed;
}

fn run_benchmark_mode(gpa: std.mem.Allocator, args: []const []const u8) !void {
    var shapes = try std.ArrayList(zg.benchmark.Shape).initCapacity(gpa, 4);
    defer shapes.deinit(gpa);
    var impls = try std.ArrayList(zg.benchmark.Implementation).initCapacity(gpa, 6);
    defer impls.deinit(gpa);
    var warmup_iters: usize = 10;
    var bench_iters: usize = 100;
    var tvm_cache_dir: []const u8 = "artifacts/tvm_cache";

    for (args) |arg| {
        if (std.mem.startsWith(u8, arg, "--shapes=")) {
            const value = arg["--shapes=".len..];
            var shape_iter = std.mem.splitScalar(u8, value, ',');
            while (shape_iter.next()) |shape_str| {
                const shape = try parse_shape(shape_str);
                try shapes.append(gpa, shape);
            }
        } else if (std.mem.startsWith(u8, arg, "--impls=")) {
            const value = arg["--impls=".len..];
            var impl_iter = std.mem.splitScalar(u8, value, ',');
            while (impl_iter.next()) |impl_str| {
                const impl = parse_impl(impl_str) orelse return error.InvalidImplementation;
                try impls.append(gpa, impl);
            }
        } else if (std.mem.startsWith(u8, arg, "--warmup=")) {
            const value = arg["--warmup=".len..];
            warmup_iters = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.startsWith(u8, arg, "--iters=")) {
            const value = arg["--iters=".len..];
            bench_iters = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.startsWith(u8, arg, "--tvm-cache-dir=")) {
            tvm_cache_dir = arg["--tvm-cache-dir=".len..];
        } else {
            std.log.err("unknown benchmark argument: {s}", .{arg});
            return error.InvalidArguments;
        }
    }

    // defaults
    if (shapes.items.len == 0) {
        try shapes.append(gpa, .{ .m = 128, .n = 128, .k = 128 });
    }
    if (impls.items.len == 0) {
        return error.InvalidArguments;
    }

    // Run benchmark
    const cfg = zg.benchmark.BenchmarkConfig{
        .shapes = shapes.items,
        .implementations = impls.items,
        .warmup_iters = warmup_iters,
        .bench_iters = bench_iters,
    };

    var harness = try zg.benchmark.Harness.init(gpa, cfg, tvm_cache_dir);
    defer harness.deinit();

    try harness.run();
    try harness.print_results();
}

fn parse_shape(s: []const u8) !zg.benchmark.Shape {
    var parts = std.mem.splitScalar(u8, s, 'x');
    const m = std.fmt.parseInt(usize, parts.next() orelse return error.InvalidShape, 10) catch return error.InvalidShape;
    const n = std.fmt.parseInt(usize, parts.next() orelse return error.InvalidShape, 10) catch return error.InvalidShape;
    const k = std.fmt.parseInt(usize, parts.next() orelse return error.InvalidShape, 10) catch return error.InvalidShape;
    return .{ .m = m, .n = n, .k = k };
}

fn parse_impl(s: []const u8) ?zg.benchmark.Implementation {
    if (std.mem.eql(u8, s, "blas")) return .blas;
    if (std.mem.eql(u8, s, "zig_naive")) return .zig_naive;
    if (std.mem.eql(u8, s, "tvm_cpu")) return .tvm_cpu;
    if (std.mem.eql(u8, s, "tvm_gpu")) return .tvm_gpu;
    if (std.mem.eql(u8, s, "xla_cpu")) return .xla_cpu;
    if (std.mem.eql(u8, s, "xla_gpu")) return .xla_gpu;
    return null;
}

fn parse_zxpr_palette(value: []const u8) ?zg.pr.zxpr.Palette {
    if (std.mem.eql(u8, value, "default")) return .default;
    if (std.mem.eql(u8, value, "alt_orange")) return .alt;
    if (std.mem.eql(u8, value, "nord")) return .nord;
    if (std.mem.eql(u8, value, "gruvbox_material")) return .gruvbox_material;
    if (std.mem.eql(u8, value, "flat_dark")) return .flat_dark;
    if (std.mem.eql(u8, value, "catppuccin")) return .catppuccin;
    if (std.mem.eql(u8, value, "tokyonight")) return .tokyonight;
    return null;
}
