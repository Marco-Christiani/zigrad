const std = @import("std");
const cova = @import("cova");
const zg = @import("zigrad");

pub const CommandT = cova.Command.Custom(.{
    .global_help_prefix = "zigrad",
    .allow_abbreviated_cmds = true,
});

pub const OptionT = CommandT.OptionT;
pub const ValueT = CommandT.ValueT;

/// Global options for all commands
pub const GlobalOpts = struct {
    help: bool = false,
    dump_pr: ?[]const u8 = null,
    dump_mlir: ?[]const u8 = null,
    quiet: bool = false,
};

/// TVM tune subcommand
pub const TvmTuneOpts = struct {
    shape: ?[]const u8 = "128x128x128",
    trials: ?u32 = 64,
    trials_per_iter: ?u32 = 16,
    work_dir: ?[]const u8 = "artifacts/tvm_cache",
    cuda: bool = false,
    gpu: bool = false,
    cpu: bool = false,
};

/// TVM run subcommand
pub const TvmRunOpts = struct {
    shape: ?[]const u8 = null,
    work_dir: ?[]const u8 = "artifacts/tvm_cache",
    cuda: bool = false,
    gpu: bool = false,
    cpu: bool = false,
};

/// TVM zxpr subcommand
pub const TvmZxprOpts = struct {
    sweep_palettes: bool = false,
    palette: ?[]const u8 = null,
};

pub const KernelProviderDemoOpts = struct {
    provider: ?[]const u8 = "tvm",
};

/// Train demo subcommand
pub const TrainDemoOpts = struct {
    warmup: ?u32 = null,
    steps: ?u32 = null,
};

/// Llama fine-tune demo
pub const LlamaFtDemoOpts = struct {
    warmup: ?u32 = null,
    steps: ?u32 = null,
    train: bool = false,
    dtype: ?[]const u8 = null,
    seq: ?u32 = null,
    batch: ?u32 = null,
    canonical_shapes: bool = false,
    execute_only: bool = false,
    kernel_provider: ?[]const u8 = null,
};

/// Benchmark subcommand
pub const BenchmarkOpts = struct {
    shapes: ?[]const u8 = "128x128x128",
    impls: ?[]const u8 = "zig_naive",
    warmup: ?u32 = 10,
    iters: ?u32 = 100,
    tvm_cache_dir: ?[]const u8 = "artifacts/tvm_cache",
};

/// JIT cache commands that take a path argument
pub const JitCacheOpts = struct {
    path: []const u8,
};

/// Root command setup with all subcommands
pub const setup_cmd: CommandT = .{
    .name = "zigrad",
    .description = "Zigrad: Zig-based automatic differentiation and compilation framework",
    .opts = &.{
        .{
            .name = "dump_pr",
            .long_name = "dump-pr",
            .description = "Dump PR: no value=zxpr to stdout, 'json'=JSON to stdout, PATH=write to file (.json extension selects JSON format)",
            .val = ValueT.ofType([]const u8, .{
                .name = "dump_pr_val",
                .description = "Format ('json') or path to write PR output",
                .default_val = "",
            }),
        },
        .{
            .name = "dump_mlir",
            .long_name = "dump-mlir",
            .description = "Print MLIR (text) to stdout, or write to PATH if value provided",
            .val = ValueT.ofType([]const u8, .{
                .name = "dump_mlir_val",
                .description = "Optional path to write MLIR output",
                .default_val = "",
            }),
        },
        .{
            .name = "quiet",
            .long_name = "quiet",
            .description = "Reduce output (train-demo/llm-ft-demo)",
            .val = ValueT.ofType(bool, .{
                .name = "quiet_val",
                .default_val = false,
            }),
        },
        .{
            .name = "dump_kernels",
            .long_name = "dump-kernels",
            .description = "Print kernelization summary table after pass (requires --kernel-provider)",
            .val = ValueT.ofType(bool, .{
                .name = "dump_kernels_val",
                .default_val = false,
            }),
        },
    },
    .sub_cmds = &.{
        .{
            .name = "print-pr",
            .description = "Print the PR for the demo program",
        },
        .{
            .name = "tvm-dump-symbols",
            .description = "Enumerate all available TVM FFI functions (requires TVM runtime libraries)",
        },
        .{
            .name = "tvm-check-compiler-load",
            .description = "Check that libtvm.so compiler can be loaded (host integration check)",
        },
        .{
            .name = "aot-demo",
            .description = "Run the AOT compile+load demo",
        },
        .{
            .name = "iree-aot-demo",
            .description = "Run the IREE AOT compile+execute demo (requires -Diree-backend=true and IREE_COMPILER_LIB env var)",
        },
        .{
            .name = "custom-call-neg",
            .description = "Expects missing custom call handler (should fail)",
        },
        CommandT.from(KernelProviderDemoOpts, .{
            .cmd_name = "kernel-provider-demo-pr",
            .cmd_description = "Run kernelized region demo on PR lane",
            .default_val_opts = true,
            .sub_descriptions = &.{
                .{ "provider", "Kernel provider(s): tvm, mirage, or comma-separated (e.g. tvm,mirage). Default: tvm" },
            },
        }),
        CommandT.from(KernelProviderDemoOpts, .{
            .cmd_name = "kernel-provider-demo-mlir",
            .cmd_description = "Run kernelized region demo on MLIR lane",
            .default_val_opts = true,
            .sub_descriptions = &.{
                .{ "provider", "Kernel provider(s): tvm, mirage, or comma-separated (e.g. tvm,mirage). Default: tvm" },
            },
        }),
        .{
            .name = "vjp-demo",
            .description = "Run the reverse-mode AD demo",
        },
        CommandT.from(TvmTuneOpts, .{
            .cmd_name = "tvm-tune",
            .cmd_description = "Run TVM MetaSchedule autotuning on matmul (requires TVM runtime libraries)",
            .default_val_opts = true,
            .sub_descriptions = &.{
                .{ "cuda", "Tune for CUDA target" },
                .{ "gpu", "Tune for GPU target (alias for --cuda)" },
                .{ "cpu", "Tune for CPU target (default)" },
                .{ "shape", "Matmul dimensions in MxNxK format (default: 128x128x128)" },
                .{ "trials", "Max tuning trials (default: 64)" },
                .{ "trials_per_iter", "Batch size per iteration (default: 16)" },
                .{ "work_dir", "Tuning cache directory (default: artifacts/tvm_cache)" },
            },
        }),
        CommandT.from(TvmRunOpts, .{
            .cmd_name = "tvm-run",
            .cmd_description = "Load and run a tuned TVM matmul (requires TVM runtime libraries + prior tuning)",
            .default_val_opts = true,
            .sub_descriptions = &.{
                .{ "cuda", "Run on CUDA target" },
                .{ "gpu", "Run on GPU target (alias for --cuda)" },
                .{ "cpu", "Run on CPU target (default)" },
                .{ "shape", "Matmul dimensions (must match tuned shape)" },
                .{ "work_dir", "Tuning cache directory (default: artifacts/tvm_cache)" },
            },
        }),
        CommandT.from(TvmZxprOpts, .{
            .cmd_name = "tvm-zxpr",
            .cmd_description = "Print a kernelized TVM region in zxpr format",
            .default_val_opts = true,
            .sub_descriptions = &.{
                .{ "sweep_palettes", "Try all available color palettes" },
                .{ "palette", "Use specific color palette" },
            },
        }),
        CommandT.from(TvmZxprOpts, .{
            .cmd_name = "tvm-attention-zxpr",
            .cmd_description = "Print attention pattern with TVM annotation",
            .default_val_opts = true,
            .sub_descriptions = &.{
                .{ "sweep_palettes", "Try all available color palettes" },
                .{ "palette", "Use specific color palette" },
            },
        }),
        CommandT.from(TrainDemoOpts, .{
            .cmd_name = "train-demo",
            .cmd_description = "Run the frontend training demo",
            .default_val_opts = true,
            .sub_descriptions = &.{
                .{ "warmup", "Number of warmup iterations" },
                .{ "steps", "Number of training steps" },
            },
        }),
        CommandT.from(TrainDemoOpts, .{
            .cmd_name = "llm-ft-demo",
            .cmd_description = "Run a tiny LLM fine-tune demo",
            .default_val_opts = true,
            .sub_descriptions = &.{
                .{ "warmup", "Number of warmup iterations" },
                .{ "steps", "Number of training steps" },
            },
        }),
        CommandT.from(LlamaFtDemoOpts, .{
            .cmd_name = "llama-ft-demo-pr",
            .cmd_description = "Run a tiny Llama fine-tune demo on PR lane",
            .default_val_opts = true,
            .sub_descriptions = &.{
                .{ "warmup", "Number of warmup iterations" },
                .{ "steps", "Number of training steps" },
                .{ "train", "Enable training mode" },
                .{ "dtype", "Data type: bf16 or f32" },
                .{ "seq", "Sequence length" },
                .{ "batch", "Batch size" },
                .{ "canonical_shapes", "Use canonical shapes" },
                .{ "execute_only", "Execute only, skip compilation" },
                .{ "kernel_provider", "Kernel provider: mirage (enables kernelize pass)" },
            },
        }),
        CommandT.from(LlamaFtDemoOpts, .{
            .cmd_name = "llama-ft-demo-mlir",
            .cmd_description = "Run a tiny Llama fine-tune demo on MLIR lane",
            .default_val_opts = true,
            .sub_descriptions = &.{
                .{ "warmup", "Number of warmup iterations" },
                .{ "steps", "Number of training steps" },
                .{ "train", "Enable training mode" },
                .{ "dtype", "Data type: bf16 or f32" },
                .{ "seq", "Sequence length" },
                .{ "batch", "Batch size" },
                .{ "canonical_shapes", "Use canonical shapes" },
                .{ "execute_only", "Execute only, skip compilation" },
                .{ "kernel_provider", "Kernel provider: mirage (enables kernelize pass)" },
            },
        }),
        CommandT.from(JitCacheOpts, .{
            .cmd_name = "jit-cache-save",
            .cmd_description = "Write PJRT JIT cache artifact",
            .default_val_opts = true,
            .sub_descriptions = &.{
                .{ "path", "Path to save the JIT cache artifact" },
            },
        }),
        CommandT.from(JitCacheOpts, .{
            .cmd_name = "jit-cache-run",
            .cmd_description = "Load and run PJRT JIT cache artifact",
            .default_val_opts = true,
            .sub_descriptions = &.{
                .{ "path", "Path to the JIT cache artifact" },
            },
        }),
        CommandT.from(BenchmarkOpts, .{
            .cmd_name = "benchmark",
            .cmd_description = "Run matmul performance benchmarks",
            .default_val_opts = true,
            .sub_descriptions = &.{
                .{ "shapes", "Comma-separated list of shapes (default: 128x128x128)" },
                .{ "impls", "Implementations to test: zig_naive, tvm_cpu, xla_cpu, etc (default: zig_naive)" },
                .{ "warmup", "Warmup iterations (default: 10)" },
                .{ "iters", "Benchmark iterations (default: 100)" },
                .{ "tvm_cache_dir", "TVM module cache directory (default: artifacts/tvm_cache)" },
            },
        }),
    },
};

/// Parse command-line arguments and return initialized command
pub fn parse(allocator: std.mem.Allocator) !CommandT {
    const cmd_ptr = try setup_cmd.init(allocator, .{});
    errdefer cmd_ptr.deinit();

    var args_iter = try cova.ArgIteratorGeneric.init(allocator);
    defer args_iter.deinit();

    var stdout_file = std.fs.File.stdout();
    var stdout_buf: [8192]u8 = undefined;
    var stdout_writer = stdout_file.writer(stdout_buf[0..]);
    const stdout = &stdout_writer.interface;

    cova.parseArgs(&args_iter, CommandT, cmd_ptr, stdout, .{
        .skip_first_arg = true,
        .auto_handle_usage_help = true,
    }) catch |err| switch (err) {
        error.UsageHelpCalled => return error.HelpShown,
        else => return err,
    };

    return cmd_ptr.*;
}

const GlobalOptsResult = struct {
    dump_pr: ?zg.pipeline.DumpConfig = null,
    dump_mlir: ?zg.pipeline.DumpConfig = null,
    dump_kernels: bool = false,
    quiet: bool = false,
};

/// Extract global options from parsed command
pub fn get_global_opts(cmd: *const CommandT, allocator: std.mem.Allocator) !GlobalOptsResult {
    var opts = try cmd.getOpts(.{});
    defer opts.deinit(allocator);

    var result: GlobalOptsResult = .{};

    if (opts.get("dump_pr")) |opt| {
        if (opt.val.isSet()) {
            const val = try opt.val.getAs([]const u8);
            result.dump_pr = parse_dump_pr_value(val);
        }
    }

    if (opts.get("dump_mlir")) |opt| {
        if (opt.val.isSet()) {
            const val = try opt.val.getAs([]const u8);
            result.dump_mlir = if (val.len > 0) .{ .target = .file, .path = val } else .{ .target = .stdout };
        }
    }

    if (opts.get("dump_kernels")) |opt| {
        if (opt.val.isSet()) {
            result.dump_kernels = try opt.val.getAs(bool);
        }
    }

    if (opts.get("quiet")) |opt| {
        if (opt.val.isSet()) {
            result.quiet = try opt.val.getAs(bool);
        }
    }

    return result;
}

/// Parse `--dump-pr` value into a DumpConfig.
///
/// - empty string -> zxpr to stdout
/// - "json" -> json to stdout
/// - path ending in ".json" -> json to file
/// - any other path -> zxpr to file
fn parse_dump_pr_value(val: []const u8) zg.pipeline.DumpConfig {
    if (val.len == 0) return .{ .target = .stdout, .format = .zxpr };
    if (std.mem.eql(u8, val, "json")) return .{ .target = .stdout, .format = .json };
    const format: zg.pipeline.DumpFormat = if (std.mem.endsWith(u8, val, ".json")) .json else .zxpr;
    return .{ .target = .file, .path = val, .format = format };
}
