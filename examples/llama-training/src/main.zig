//! LLaMA 3.2 training and inference through selectable Zigrad backends.

const std = @import("std");
const zg = @import("zigrad");

const training = @import("training.zig");
const log = std.log.scoped(.@"zg/example/llama_training");

const default_weights_path = "./weights/llama-3.2-1b-instruct/model.safetensors";

const BackendKind = enum {
    pjrt,
    iree,
};

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub fn main(init: std.process.Init) !void {
    const env = zg.RuntimeEnv.from_init(init);
    const args = try init.minimal.args.toSlice(env.allocator);
    defer env.allocator.free(args);

    var backend_kind: BackendKind = .pjrt;
    var options = training.Options{
        .weights_path = env.environ.get("ZG_LLAMA_SAFETENSORS_PATH") orelse
            default_weights_path,
    };
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            std.debug.print("{s}", .{usage});
            return;
        } else if (value(arg, "--backend=")) |name| {
            backend_kind = std.meta.stringToEnum(BackendKind, name) orelse
                return error.InvalidBackend;
        } else if (value(arg, "--mode=")) |name| {
            options.mode = std.meta.stringToEnum(training.Mode, name) orelse
                return error.InvalidMode;
        } else if (value(arg, "--dtype=")) |name| {
            const dtype = std.meta.stringToEnum(zg.DType, name) orelse
                return error.InvalidDType;
            options.dtype = switch (dtype) {
                .bf16, .f16, .f32, .f64 => dtype,
                else => return error.InvalidDType,
            };
        } else if (value(arg, "--sequence-length=")) |raw| {
            options.sequence_length = try std.fmt.parseInt(i64, raw, 10);
        } else if (value(arg, "--batch=")) |raw| {
            options.batch = try std.fmt.parseInt(i64, raw, 10);
        } else if (value(arg, "--warmup-steps=")) |raw| {
            options.warmup_steps = try std.fmt.parseInt(usize, raw, 10);
        } else if (value(arg, "--steps=")) |raw| {
            options.steps = try std.fmt.parseInt(usize, raw, 10);
        } else if (value(arg, "--weights=")) |path| {
            options.weights_path = path;
        } else if (std.mem.eql(u8, arg, "--quiet")) {
            options.quiet = true;
        } else {
            log.err("unknown argument '{s}'", .{arg});
            return error.InvalidArgument;
        }
    }

    return switch (backend_kind) {
        .pjrt => if (comptime zg.build_options.has_pjrt)
            run_pjrt(env, options)
        else
            unavailable("PJRT", "-Dpjrt=true"),
        .iree => if (comptime zg.build_options.has_iree)
            run_iree(env, options)
        else
            unavailable("IREE", "-Diree=true"),
    };
}

fn run_pjrt(env: zg.RuntimeEnv, options: training.Options) !void {
    const plugin_path = env.environ.get("PJRT_PLUGIN_PATH") orelse {
        log.err("set PJRT_PLUGIN_PATH to a PJRT plugin", .{});
        return error.MissingPlugin;
    };
    const client_options = try zg.pjrt.config.from_environ(env.environ);
    var client = try zg.pjrt.Client.init(env.allocator, plugin_path, client_options);
    defer client.deinit();
    const devices = try client.get_devices(env.allocator);
    defer env.allocator.free(devices);
    if (devices.len == 0) return error.NoDevices;

    var execution = try zg.pjrt.Execution.init(&client, devices[0], .{});
    var backend = zg.pjrt.Backend.init(&execution, .{});
    var ctx = zg.CompilationCtx{
        .allocator = env.allocator,
        .io = env.io,
        .device = execution.interface.device,
    };
    return try training.run(&ctx, &backend.interface, options);
}

fn run_iree(env: zg.RuntimeEnv, options: training.Options) !void {
    const config = zg.iree.Config.from_environ(env.environ);
    var runtime = try zg.iree.Runtime.init(env.allocator, .registered, config.runtime);
    defer runtime.deinit();
    var execution = zg.iree.Execution.init(env.allocator, &runtime, config.runtime);
    var backend = zg.iree.Backend.init(&execution, config.compiler, "module.main");
    var ctx = zg.CompilationCtx{
        .allocator = env.allocator,
        .io = env.io,
        .device = execution.interface.device,
    };
    return try training.run(&ctx, &backend.interface, options);
}

fn unavailable(comptime name: []const u8, comptime build_option: []const u8) error{BackendUnavailable} {
    log.err("{s} is unavailable, rebuild with {s}", .{ name, build_option });
    return error.BackendUnavailable;
}

fn value(argument: []const u8, prefix: []const u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, argument, prefix)) return null;
    return argument[prefix.len..];
}

const usage =
    \\Usage: llama-training [options]
    \\
    \\Options:
    \\  --backend=pjrt|iree
    \\  --mode=training|inference
    \\  --dtype=bf16|f16|f32|f64
    \\  --sequence-length=N
    \\  --batch=N
    \\  --warmup-steps=N
    \\  --steps=N
    \\  --weights=PATH
    \\  --quiet
    \\  --help, -h
    \\
    \\The default checkpoint path is ./weights/llama-3.2-1b-instruct/model.safetensors.
    \\ZG_LLAMA_SAFETENSORS_PATH overrides it when --weights is absent.
    \\PJRT requires PJRT_PLUGIN_PATH. IREE reads ZG_IREE_* configuration.
;
