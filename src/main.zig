const std = @import("std");
const zg = @import("zigrad");
const build_options = zg.build_options;
const demos = @import("demos.zig");
const llama_demo = @import("llama_demo.zig");
const llm_demo = @import("llm_demo.zig");
const main_aot = @import("main_aot.zig");
const llama_model = @import("llama_model.zig");
const cli = @import("cli.zig");
const log = std.log.scoped(.@"zg/main");

pub fn main(init: std.process.Init) !void {
    const env = zg.RuntimeEnv.from_init(init);
    const gpa = env.allocator;

    var parsed = cli.parse(env, init.minimal.args) catch |err| {
        if (err == error.HelpShown) return;
        if (err == error.InvalidArguments) std.process.exit(2);
        return err;
    };
    defer parsed.deinit();

    if (comptime build_options.has_tvm) {
        try zg.tvm.runtime.configure(.from_environ(
            env.environ,
            tvm_surface_for_command(parsed.invocation.command),
        ));
    }

    var global = parsed.invocation.global;
    const dumps = DumpOptions{
        .pr = if (global.dump_pr) |*config| config else null,
        .mlir = if (global.dump_mlir) |*config| config else null,
        .optimized_hlo = if (global.dump_optimized_hlo) |*config| config else null,
        .kernels = global.dump_kernels,
        .quiet = global.quiet,
    };

    return switch (parsed.invocation.command) {
        .pr => |command| dispatch_pr(env, gpa, command),
        .tvm => |command| dispatch_tvm(env, gpa, command),
        .iree => |command| dispatch_iree(env, gpa, command, dumps),
        .pjrt => |command| dispatch_pjrt(env, gpa, .{ .pjrt = command }, dumps),
        .demo => |command| dispatch_pjrt(env, gpa, .{ .demo = command }, dumps),
    };
}

fn tvm_surface_for_command(command: cli.Command) zg.tvm.runtime.Surface {
    return switch (command) {
        .tvm => |tvm_command| switch (tvm_command) {
            .symbols => |opts| if (opts.load_compiler) .compiler else .ffi,
            .check_load, .tune => .compiler,
            .run => .runtime,
            .render_matmul, .render_attention => .ffi,
        },
        .demo => |demo_command| switch (demo_command) {
            .kernel_provider => .compiler,
            .llama_finetune => |opts| if (opts.kernel_provider) |provider|
                if (std.mem.eql(u8, provider, "tvm")) .compiler else .ffi
            else
                .ffi,
            else => .ffi,
        },
        else => .ffi,
    };
}

const DumpOptions = struct {
    pr: ?*zg.pr.dump.Config,
    mlir: ?*zg.output.Config,
    optimized_hlo: ?*zg.output.Config,
    kernels: bool,
    quiet: bool,
};

fn dispatch_pr(env: zg.RuntimeEnv, gpa: std.mem.Allocator, command: cli.PrCommand) !void {
    return switch (command) {
        .print_demo => demos.print_pr(env.io, gpa, env.environ),
        .render => |opts| zg.pr.tool.run(
            env.io,
            gpa,
            opts.path,
            .{ .render = opts.format },
        ),
        .info => |opts| zg.pr.tool.run(env.io, gpa, opts.path, .info),
    };
}

fn dispatch_tvm(env: zg.RuntimeEnv, gpa: std.mem.Allocator, command: cli.TvmCommand) !void {
    switch (command) {
        .render_matmul => |opts| {
            const palette = try parse_palette(opts.palette);
            return try demos.print_tvm_kernelize_pr(
                env.io,
                gpa,
                env.environ,
                opts.sweep_palettes,
                palette,
            );
        },
        .render_attention => |opts| {
            const palette = try parse_palette(opts.palette);
            return try demos.print_tvm_attention_pr(
                env.io,
                gpa,
                env.environ,
                opts.sweep_palettes,
                palette,
            );
        },
        .symbols => |opts| {
            if (comptime !build_options.has_tvm) return try require_tvm();
            return try demos.dump_tvm_ffi_symbols(env.io, gpa, opts.load_compiler);
        },
        .check_load => {
            if (comptime !build_options.has_tvm) return try require_tvm();
            try zg.tvm.runtime.ensure_loaded(.compiler);
            log.info("tvm compiler load check passed", .{});
        },
        .tune => |opts| {
            if (comptime !build_options.has_tvm) return try require_tvm();
            const shape = try parse_shape(opts.shape orelse "128x128x128");
            const target_kind: zg.tvm.TargetKind = if (opts.cuda or opts.gpu) .cuda else .cpu;
            const selected_device = zg.Device{
                .platform = if (target_kind == .cuda) .cuda else .cpu,
            };
            const compile_config = try zg.tvm.CompileConfig.from_environ(
                env.environ,
                target_kind,
            );
            const cache = try zg.Cache.init(env.io, env.environ, .{});
            _ = try zg.tvm.tune_matmul(
                env.io,
                gpa,
                cache,
                .{
                    .m = @intCast(shape.m),
                    .n = @intCast(shape.n),
                    .k = @intCast(shape.k),
                },
                .{
                    .compile = compile_config,
                    .device = selected_device,
                    .max_trials = opts.trials orelse 64,
                    .trials_per_iter = opts.trials_per_iter orelse 16,
                },
            );
        },
        .run => |opts| {
            if (comptime !build_options.has_tvm) return try require_tvm();
            const shape = try parse_shape(opts.shape orelse "128x128x128");
            const target_kind: zg.tvm.TargetKind = if (opts.cuda or opts.gpu) .cuda else .cpu;
            const cache = try zg.Cache.init(env.io, env.environ, .{});
            return try run_tvm_demo(
                env.io,
                gpa,
                @intCast(shape.m),
                @intCast(shape.n),
                @intCast(shape.k),
                target_kind,
                cache,
            );
        },
    }
}

fn dispatch_iree(
    env: zg.RuntimeEnv,
    gpa: std.mem.Allocator,
    command: cli.IreeCommand,
    dumps: DumpOptions,
) !void {
    if (comptime build_options.has_iree and build_options.has_mlir) {
        return switch (command) {
            .demo => run_iree_demo(env.io, gpa, env.environ, dumps.pr, dumps.mlir),
            .compile => |opts| run_iree_aot_compile(
                env.io,
                gpa,
                env.environ,
                opts,
                dumps.mlir,
            ),
        };
    }
    log.err("IREE commands require the opt-in IREE and MLIR integrations", .{});
    return error.IreeBackendDisabled;
}

const PjrtWork = union(enum) {
    pjrt: cli.PjrtCommand,
    demo: cli.DemoCommand,
};

fn dispatch_pjrt(
    env: zg.RuntimeEnv,
    gpa: std.mem.Allocator,
    work: PjrtWork,
    dumps: DumpOptions,
) !void {
    if (comptime build_options.has_pjrt) {
        const plugin_path = env.environ.get("PJRT_PLUGIN_PATH") orelse {
            log.err("PJRT_PLUGIN_PATH is not set", .{});
            return error.PjrtPluginPathNotSet;
        };

        const pjrt_options = zg.pjrt.config.from_environ(env.environ) catch |err| {
            log.err("invalid PJRT runtime configuration: {s}", .{@errorName(err)});
            return err;
        };
        var pjrt_client = try zg.pjrt.Client.init(gpa, plugin_path, pjrt_options);
        defer pjrt_client.deinit();

        const devices = try pjrt_client.get_devices(gpa);
        defer gpa.free(devices);
        if (devices.len == 0) {
            log.err("no devices available from PJRT", .{});
            return error.NoDevices;
        }
        const device = devices[0];
        var execution = try zg.pjrt.Execution.init(&pjrt_client, device, .{});
        var compilation_context = zg.compilation.Context{
            .allocator = gpa,
            .io = env.io,
            .device = execution.interface.device,
        };
        var backend = zg.pjrt.Backend.init(&execution, .{});
        return switch (work) {
            .pjrt => |command| dispatch_pjrt_artifact(
                env,
                command,
                &compilation_context,
                &pjrt_client,
                &execution,
                &backend,
            ),
            .demo => |command| dispatch_pjrt_demo(
                env,
                command,
                &compilation_context,
                &pjrt_client,
                &execution,
                &backend,
                dumps,
                dumps.quiet,
            ),
        };
    }

    log.err("this command requires the opt-in PJRT integration", .{});
    return error.PjrtDisabled;
}

fn dispatch_pjrt_artifact(
    env: zg.RuntimeEnv,
    command: cli.PjrtCommand,
    compilation_context: *zg.compilation.Context,
    client: *zg.pjrt.Client,
    execution: *zg.pjrt.Execution,
    backend: *zg.pjrt.Backend,
) !void {
    const gpa = compilation_context.allocator;
    return switch (command) {
        .aot_demo => {
            if (comptime !build_options.has_mlir) {
                log.err("pjrt aot-demo requires the opt-in MLIR integration", .{});
                return error.MlirDisabled;
            }
            return try main_aot.run(
                compilation_context,
                client,
                execution,
                backend,
            );
        },
        .cache => |cache| switch (cache) {
            .save => |opts| {
                if (comptime !build_options.has_mlir) {
                    log.err("pjrt cache save requires the opt-in MLIR integration", .{});
                    return error.MlirDisabled;
                }

                var program = try demos.build_demo_program(gpa);
                defer program.deinit();

                var pipeline = try zg.pjrt.pipeline.create(gpa, backend, .{
                    .stablehlo = .{ .entry_name = "main" },
                });
                defer pipeline.deinit();
                var loaded_program = try pipeline.run(
                    zg.Executor.LoadedProgram,
                    &program,
                    compilation_context,
                );
                defer loaded_program.deinit();
                const serialized = try (try execution.loaded(loaded_program)).serialize(
                    client.api,
                    gpa,
                );
                defer gpa.free(serialized);

                try demos.write_bytes_to_path(env.io, opts.path, serialized);
                log.info(
                    "wrote PJRT executable artifact: {d} bytes -> {s}",
                    .{ serialized.len, opts.path },
                );
            },
            .run => |opts| {
                const serialized = try demos.read_bytes_from_path(env.io, gpa, opts.path);
                defer gpa.free(serialized);

                var artifact = zg.pjrt.Artifact{
                    .client = client,
                    .loaded = try client.load_serialized_executable(serialized, null),
                };
                errdefer artifact.deinit();
                var loaded_program = try backend.loader.interface.load(&artifact);
                defer loaded_program.deinit();
                return try demos.run_demo_executable(gpa, loaded_program);
            },
        },
    };
}

fn dispatch_pjrt_demo(
    env: zg.RuntimeEnv,
    command: cli.DemoCommand,
    compilation_context: *zg.compilation.Context,
    client: *zg.pjrt.Client,
    execution: *zg.pjrt.Execution,
    backend: *zg.pjrt.Backend,
    dumps: DumpOptions,
    quiet: bool,
) !void {
    if (comptime !build_options.has_mlir) {
        log.err("the current executable demos require the opt-in MLIR integration", .{});
        return error.MlirDisabled;
    }

    return switch (command) {
        .custom_call_negative => {
            var pipeline = try zg.pjrt.pipeline.create(
                compilation_context.allocator,
                backend,
                demo_pipeline_options(dumps, execution, "main"),
            );
            defer pipeline.deinit();
            return try demos.run_custom_call_negative(compilation_context, &pipeline);
        },
        .kernel_provider => |opts| {
            const providers = try parse_provider_kinds(opts.provider orelse "tvm");
            return try demos.run_kernel_provider_demo(
                compilation_context,
                client,
                execution,
                backend,
                env.environ,
                demo_pipeline_options(dumps, execution, "main"),
                dumps.kernels,
                providers.slice(),
            );
        },
        .vjp => {
            var pipeline = try zg.pjrt.pipeline.create(
                compilation_context.allocator,
                backend,
                demo_pipeline_options(dumps, execution, "main_vjp"),
            );
            defer pipeline.deinit();
            return try demos.run_vjp_demo(compilation_context, &pipeline);
        },
        .train => |opts| {
            var pipeline = try zg.pjrt.pipeline.create(
                compilation_context.allocator,
                backend,
                demo_pipeline_options(dumps, execution, "train_step"),
            );
            defer pipeline.deinit();
            return try demos.run_train_demo(
                compilation_context,
                &pipeline,
                opts.warmup orelse 0,
                opts.steps orelse 8,
                quiet,
            );
        },
        .llm_train => |opts| {
            var pipeline = try zg.pjrt.pipeline.create(
                compilation_context.allocator,
                backend,
                demo_pipeline_options(dumps, execution, "llm_ft_step"),
            );
            defer pipeline.deinit();
            return try llm_demo.run_llm_train_demo(
                compilation_context,
                &pipeline,
                env.environ,
                opts.warmup orelse 0,
                opts.steps orelse 8,
                quiet,
            );
        },
        .llama_finetune => |opts| {
            const dtype = std.meta.stringToEnum(zg.DType, opts.dtype) orelse
                return error.InvalidDType;
            const kernel_provider = if (opts.kernel_provider) |name|
                std.meta.stringToEnum(llama_demo.LlamaKernelProvider, name) orelse
                    return error.InvalidArgument
            else
                null;
            const config = llama_demo.LlamaDemoConfig{
                .train = opts.train,
                .dtype = dtype,
                .seq = opts.seq,
                .batch = opts.batch orelse 1,
                .execute_only = opts.execute_only,
                .kernel_provider = kernel_provider,
            };

            var pipeline = try zg.pjrt.pipeline.create(
                compilation_context.allocator,
                backend,
                demo_pipeline_options(dumps, execution, "llama_ft_step"),
            );
            defer pipeline.deinit();
            return try llama_demo.run_llama_ft_demo(
                compilation_context,
                &pipeline,
                env.environ,
                opts.warmup,
                opts.steps,
                quiet,
                config,
            );
        },
    };
}

fn demo_pipeline_options(
    dumps: DumpOptions,
    execution: *zg.pjrt.Execution,
    entry_name: []const u8,
) zg.pjrt.pipeline.Options {
    return .{
        .stablehlo = .{
            .entry_name = entry_name,
            .dump_pr = if (dumps.pr) |config| .{ .config = config.* } else null,
            .dump_stablehlo = if (dumps.mlir) |config| .{ .config = config.* } else null,
        },
        .dump_optimized_hlo = if (dumps.optimized_hlo) |config| .{
            .execution = execution,
            .config = config.*,
        } else null,
    };
}

fn parse_palette(name: ?[]const u8) !?zg.pr.zxpr.style.Palette {
    const value = name orelse return null;
    return std.meta.stringToEnum(zg.pr.zxpr.style.Palette, value) orelse
        error.InvalidPalette;
}

fn require_tvm() error{TvmUnavailable}!void {
    log.err("this command requires the opt-in TVM integration", .{});
    return error.TvmUnavailable;
}

fn run_tvm_demo(
    io: std.Io,
    gpa: std.mem.Allocator,
    m: i64,
    n: i64,
    k: i64,
    target_kind: if (build_options.has_tvm) zg.tvm.TargetKind else void,
    artifact_cache: zg.Cache,
) !void {
    const shape: zg.tvm.MatmulShape = .{ .m = m, .n = n, .k = k };
    const selected_device = zg.Device{
        .platform = if (target_kind == .cuda) .cuda else .cpu,
    };
    const cached = try zg.tvm.CachedMatmul.load(
        io,
        gpa,
        artifact_cache,
        shape,
        target_kind,
        selected_device,
    ) orelse return error.NoTuningRecords;
    defer cached.deinit();

    const a = try gpa.alloc(f32, @intCast(m * k));
    defer gpa.free(a);
    const b = try gpa.alloc(f32, @intCast(k * n));
    defer gpa.free(b);
    const result = try gpa.alloc(f32, @intCast(m * n));
    defer gpa.free(result);

    for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) * 0.1;
    for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) * 0.1;
    @memset(result, 0);

    const start = std.Io.Timestamp.now(io, .awake);
    try cached.execute(a, b, result);

    const elapsed = start.untilNow(io, .awake);
    const ms = @as(f64, @floatFromInt(elapsed.toNanoseconds())) / 1_000_000.0;
    std.log.info("TVM matmul {d}x{d}x{d} ({s}): {d:.3} ms", .{ m, n, k, @tagName(target_kind), ms });

    var expected: f32 = 0;
    for (0..@intCast(k)) |inner| {
        const a_val: f32 = @as(f32, @floatFromInt((@as(usize, @intCast(m - 1)) * @as(usize, @intCast(k)) + inner) % 7)) * 0.1;
        const b_val: f32 = @as(f32, @floatFromInt((inner * @as(usize, @intCast(n)) + @as(usize, @intCast(n - 1))) % 11)) * 0.1;
        expected += a_val * b_val;
    }
    std.log.info("result[m-1,n-1] = {d:.6} (expected ~{d:.6})", .{ result[@as(usize, @intCast(m * n - 1))], expected });
}

// TODO(cli): Replace this fixed buffer when provider registration becomes dynamic.
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

fn run_iree_demo(
    io: std.Io,
    gpa: std.mem.Allocator,
    environ: *const std.process.Environ.Map,
    dump_pr: ?*zg.pr.dump.Config,
    dump_mlir: ?*zg.output.Config,
) !void {
    const config = zg.iree.Config.from_environ(environ);
    var runtime = try zg.iree.Runtime.init(gpa, config.runtime);
    defer runtime.deinit();
    var execution = zg.iree.Execution.init(gpa, &runtime, config.runtime);

    var program = try demos.build_demo_program(gpa);
    defer program.deinit();

    var compilation_context = zg.compilation.Context{
        .allocator = gpa,
        .io = io,
        .device = execution.interface.device,
    };
    var backend = zg.iree.Backend.init(&execution, config.compiler, "module.main");
    var pipeline = try zg.iree.pipeline.create(gpa, .{ .loaded = &backend }, .{
        .stablehlo = .{
            .entry_name = "main",
            .dump_pr = if (dump_pr) |selected| .{ .config = selected.* } else null,
            .dump_stablehlo = if (dump_mlir) |selected| .{ .config = selected.* } else null,
        },
    });
    defer pipeline.deinit();
    var loaded_program = try pipeline.run(
        zg.Executor.LoadedProgram,
        &program,
        &compilation_context,
    );
    defer loaded_program.deinit();
    try demos.run_demo_executable(gpa, loaded_program);
}

fn run_iree_aot_compile(
    io: std.Io,
    gpa: std.mem.Allocator,
    environ: *const std.process.Environ.Map,
    opts: cli.IreeCompileOpts,
    dump_mlir: ?*zg.output.Config,
) !void {
    var config = zg.iree.Config.from_environ(environ);
    if (opts.backend) |target_backend|
        config.compiler.target_backend = target_backend;

    var program = try demos.build_demo_program(gpa);
    defer program.deinit();

    var compilation_context = zg.compilation.Context{
        .allocator = gpa,
        .io = io,
    };
    var compiler = zg.iree.Compiler{ .config = config.compiler };
    var pipeline = try zg.iree.pipeline.create(gpa, .{ .vmfb = &compiler }, .{
        .stablehlo = .{
            .entry_name = "main",
            .dump_stablehlo = if (dump_mlir) |selected| .{ .config = selected.* } else null,
        },
    });
    defer pipeline.deinit();
    var vmfb = try pipeline.run(
        zg.iree.Artifact,
        &program,
        &compilation_context,
    );
    defer vmfb.deinit();

    const output_path = opts.output orelse "demo.vmfb";
    try demos.write_bytes_to_path(io, output_path, vmfb.bytes);
    std.log.info("wrote {d} bytes VMFB -> {s}", .{ vmfb.bytes.len, output_path });
}

const MatmulShape = struct { m: i64, n: i64, k: i64 };

fn parse_shape(s: []const u8) !MatmulShape {
    var parts = std.mem.splitScalar(u8, s, 'x');
    const m = try std.fmt.parseInt(i64, parts.next() orelse return error.InvalidShape, 10);
    const n = try std.fmt.parseInt(i64, parts.next() orelse return error.InvalidShape, 10);
    const k = try std.fmt.parseInt(i64, parts.next() orelse return error.InvalidShape, 10);
    return .{ .m = m, .n = n, .k = k };
}
