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
            const serialized = try backend.compile_serialized(device, mlir_bytes, .mlir_bytecode, .{});
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
            defer exe.deinit();

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
    defer exe.deinit();

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
        \\  aot-demo                     runs the AOT compile+load demo
        \\  custom-call-neg              expects missing custom call handler
        \\  vjp-demo                     runs the reverse-mode demo
        \\  train-demo [warmup] [steps]  runs the frontend training demo
        \\  llm-ft-demo [warmup] [steps] runs a tiny LLM fine-tune demo
        \\  llama-ft-demo [warmup] [steps] [--train] [--dtype=bf16|f32] [--seq=N] [--batch=N] [--canonical-shapes] [--canonical-qkv] [--canonical-o] [--canonical-mlp] [--execute-only] runs a tiny Llama fine-tune demo
        \\  jit-cache-save <path>        writes PJRT JIT cache artifact
        \\  jit-cache-run <path>         loads and runs PJRT JIT cache artifact
        \\
    );

    try out.flush();
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
