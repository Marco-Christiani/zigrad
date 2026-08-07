//! MetaSchedule autotuning for TVM.
//!
//! Runs TVM's MetaSchedule search to find optimal schedules for a given
//!  IRModule and target.
//! Uses a random cost model with evolutionary search.
//! Builder and runner callbacks receive provider state through TVM's
//!  userdata pointer.
const std = @import("std");
const device = @import("../device.zig");
const tir = @import("../c/tvm/tir.zig");
const runtime = @import("../c/tvm/runtime.zig");
const ms = @import("../c/tvm/meta_schedule.zig");
const compile = @import("../c/tvm/compile.zig");
const api = @import("../c/tvm/api.zig");
const c = @import("../c/tvm/c.zig");
const dlpack = @import("../c/dlpack.zig");
const Cache = @import("../cache.zig").Cache;
const build_options = @import("build_options");
const config = @import("config.zig");
const integration_runtime = @import("runtime.zig");
const export_mod = @import("export.zig");
const Linker = @import("../toolchain/linker.zig").Linker;
const Value = api.Value;
const Array = api.Array;
const IRModule = tir.IRModule;
const RuntimeModule = runtime.RuntimeModule;
const Target = tir.Target;
const Tensor = runtime.Tensor;
const TargetKind = @import("config.zig").TargetKind;
const MetaSchedule = ms.MetaSchedule;
const nvrtc_callback = if (build_options.has_nvrtc) @import("nvrtc_callback.zig") else struct {};

const log = std.log.scoped(.@"zg/tvm_tune");

pub const TuneOpts = struct {
    /// Target-specific compiler inputs resolved by application composition.
    compile: config.CompileConfig,

    /// Device used for candidate compilation and measurement.
    device: device.Device,

    /// NVRTC architecture resolved from the TVM CUDA target.
    gpu_arch: ?[]const u8,

    /// Directory containing this workload's tuning state and candidates.
    work_cache: Cache,

    /// Maximum measured candidates.
    max_trials: u32 = 64,

    /// Candidates submitted per tuning iteration.
    trials_per_iter: u32 = 16,
};

/// Persistent state for MetaSchedule callbacks.
/// Passed as userdata to builder/runner callbacks via TVM's
///  TVMFFIFunctionCreate self pointer.
const TuneState = struct {
    io: std.Io,
    allocator: std.mem.Allocator,
    target: Target,
    target_kind: TargetKind,
    linker: Linker,
    device_ordinal: i32,
    work_cache: Cache,
    build_counter: u32 = 0,
    /// Counter value at the start of this tuning run (before any new candidates).
    initial_counter: u32 = 0,
    max_trials: u32,
    /// Tensor shapes for the workload (A, B, C for matmul).
    tensor_shapes: []const []const i64,
};

/// Tune an IRModule via MetaSchedule, returning the tuned IRModule.
///
/// Runs evolutionary search with a random cost model. The builder callback
///  compiles TIR candidates to .so artifacts, the runner callback loads and
///  benchmarks them. Results are persisted to a JSON store in work_dir.
pub fn tune(
    io: std.Io,
    allocator: std.mem.Allocator,
    ir_mod: IRModule,
    target: Target,
    tensor_shapes: []const []const i64,
    opts: TuneOpts,
) !void {
    try integration_runtime.ensure_loaded(.compiler);
    const kind = opts.compile.target;

    if (kind == .cuda) {
        if (comptime build_options.has_nvrtc) {
            const nvrtc_config = opts.compile.nvrtc orelse {
                log.err("TVM CUDA tuning requires resolved NVRTC configuration", .{});
                return error.MissingNvrtcConfig;
            };
            const gpu_arch = opts.gpu_arch orelse {
                log.err("TVM CUDA target has no resolved NVRTC architecture", .{});
                return error.MissingNvrtcArchitecture;
            };
            nvrtc_callback.register(
                allocator,
                nvrtc_config,
                gpu_arch,
            ) catch |err| {
                log.err("failed to register NVRTC callback: {s}", .{@errorName(err)});
                return err;
            };
        } else {
            log.err("TVM CUDA tuning requires the opt-in NVRTC integration", .{});
            return error.NvrtcDisabled;
        }
        try load_cuda_intrinsics(io, allocator);
    }

    const work_dir = opts.work_cache.path();
    std.Io.Dir.cwd().createDirPath(io, work_dir) catch {};

    try register_cpu_count(allocator);

    const schedule_rules = try MetaSchedule.schedule_rules(allocator, kind);
    log.debug("created ScheduleRules", .{});

    const space_gen = try MetaSchedule.space_generator(allocator, schedule_rules);
    log.debug("created SpaceGenerator", .{});

    const search_strategy = try MetaSchedule.search_strategy(allocator, .{});
    log.debug("created SearchStrategy", .{});

    // JSON database
    var workload = try opts.work_cache.join("workload.json");
    const workload_path = workload.pathZ();
    var record = try opts.work_cache.join("tuning_record.json");
    const record_path = record.pathZ();

    const database = try MetaSchedule.json_database(allocator, workload_path, record_path);
    log.debug("created JSONDatabase", .{});

    const logger_val = try make_noop_callback();
    defer logger_val.decref();

    const tune_context = try MetaSchedule.tune_context(allocator, .{
        .ir_mod = ir_mod.as_value(),
        .target = target.as_value(),
        .space_gen = space_gen,
        .search_strat = search_strategy,
        .task_name = "main",
        .logger = logger_val,
    });
    log.debug("created TuneContext", .{});

    // Seed build_counter from persisted state so incremental tuning
    // doesn't overwrite previous candidates.
    const persisted = read_tune_state(io, allocator, opts.work_cache);
    const initial_counter = persisted.next_candidate;
    if (initial_counter > 0) {
        log.info("resuming from candidate {d} (found {d} existing)", .{ initial_counter, initial_counter });
    }

    var state = TuneState{
        .io = io,
        .allocator = allocator,
        .target = target,
        .target_kind = kind,
        .linker = opts.compile.linker,
        .device_ordinal = opts.device.ordinal,
        .work_cache = opts.work_cache,
        .build_counter = initial_counter,
        .initial_counter = initial_counter,
        .max_trials = opts.max_trials,
        .tensor_shapes = tensor_shapes,
    };

    // Builder callback
    const builder_func = try api.create_packed_func(@ptrCast(&state), build_callback, null);
    defer builder_func.decref();
    const builder = try MetaSchedule.py_builder(allocator, builder_func);
    log.debug("created PyBuilder", .{});

    // Runner callback
    const runner_func = try api.create_packed_func(@ptrCast(&state), run_callback, null);
    defer runner_func.decref();
    const runner = try MetaSchedule.py_runner(allocator, runner_func);
    log.debug("created PyRunner", .{});

    // Cost model (random)
    const cost_model = try make_random_cost_model(allocator);
    log.debug("created PyCostModel", .{});

    // Task scheduler
    const task_scheduler = try MetaSchedule.task_scheduler(allocator, .{ .logger = logger_val });
    log.debug("created TaskScheduler", .{});

    // Run tuning
    log.info("starting tuning ({d} max trials, {d} per iter)...", .{ opts.max_trials, opts.trials_per_iter });

    var contexts_arr = try Array.from_values(allocator, &.{tune_context});
    defer contexts_arr.deinit();
    var weights_arr = try Array.from_values(allocator, &.{Value.float(1.0)});
    defer weights_arr.deinit();

    const add_to_db = try MetaSchedule.add_to_database(allocator);
    var callbacks_arr = try Array.from_values(allocator, &.{add_to_db});
    defer callbacks_arr.deinit();

    const max_trials: i64 = @intCast(opts.max_trials);
    MetaSchedule.run_tune(allocator, .{
        .scheduler = task_scheduler,
        .contexts = contexts_arr.as_value(),
        .weights = weights_arr.as_value(),
        .max_trials = max_trials,
        .max_trials_global = max_trials,
        .trials_per_iter = @intCast(opts.trials_per_iter),
        .builder = builder,
        .runner = runner,
        .callbacks = callbacks_arr.as_value(),
        .database = database,
        .cost_model = cost_model,
    }) catch |err| {
        log.err("TaskSchedulerTune failed: {s}", .{@errorName(err)});
        return err;
    };

    // Persist build counter so the next run continues from where we left off.
    write_tune_state(io, allocator, opts.work_cache, .{
        .next_candidate = @atomicLoad(u32, &state.build_counter, .seq_cst),
    });

    log.info("tuning complete. results: {s}", .{record_path});
}

// Callbacks

/// Builder callback: compiles TIR candidates to .so artifacts.
fn build_callback(
    self_ptr: ?*anyopaque,
    args: [*c]const c.TVMFFIAny,
    num_args: i32,
    result: [*c]c.TVMFFIAny,
) callconv(.c) c_int {
    const state: *TuneState = @ptrCast(@alignCast(self_ptr orelse {
        log.err("build_callback: null state", .{});
        return -1;
    }));

    if (num_args != 1) {
        log.err("build_callback: expected 1 arg, got {d}", .{num_args});
        return -1;
    }

    build_callback_impl(state, args[0], result) catch |err| {
        log.err("build_callback failed: {s}", .{@errorName(err)});
        return -1;
    };
    return 0;
}

fn build_callback_impl(state: *TuneState, inputs_array_raw: c.TVMFFIAny, result: *c.TVMFFIAny) !void {
    const allocator = state.allocator;

    var inputs = try Array.wrap(.{ .raw = inputs_array_raw });
    defer inputs.deinit();

    const num_inputs = try inputs.len(allocator);
    log.info("building {d} candidates", .{num_inputs});

    var results_list = std.ArrayList(Value).empty;
    defer results_list.deinit(allocator);

    for (0..num_inputs) |i| {
        const input = try inputs.get(allocator, i);

        // Extract mod from BuilderInput
        const mod_val = api.get_field(input, "mod") catch {
            try results_list.append(allocator, try make_builder_error(allocator, "failed to get mod"));
            continue;
        };

        // Lower and compile the candidate.
        var ir_mod = IRModule{ .handle = .{ .ptr = mod_val.as_object() orelse {
            try results_list.append(allocator, try make_builder_error(allocator, "mod not an object"));
            continue;
        } }, .type_index = mod_val.raw.type_index };
        ir_mod.handle.incref();

        const compiled_module_result = compile.lower_and_compile(allocator, &ir_mod, state.target, state.target_kind);
        var compiled_module = compiled_module_result catch {
            try results_list.append(allocator, try make_builder_error(allocator, "compilation failed"));
            continue;
        };
        defer compiled_module.deinit();

        // Export to .so
        const build_id = @atomicRmw(u32, &state.build_counter, .Add, 1, .seq_cst);
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "candidate_{d}.so", .{build_id}) catch unreachable;
        var so = state.work_cache.join(name) catch {
            try results_list.append(allocator, try make_builder_error(allocator, "path too long"));
            continue;
        };
        const so_path = so.pathZ();

        export_mod.export_shared(
            compiled_module,
            state.io,
            allocator,
            so_path,
            state.target_kind,
            state.linker,
        ) catch |err| {
            log.err("candidate {d} export failed: {s}", .{ build_id, @errorName(err) });
            try results_list.append(allocator, try make_builder_error(allocator, "export failed"));
            continue;
        };

        const br = try MetaSchedule.builder_result(allocator, so_path, null);
        try results_list.append(allocator, br);
        log.debug("built trial {d}/{d} (total {d}/{d})", .{
            i + 1, num_inputs, build_id + 1, state.initial_counter + state.max_trials,
        });
    }

    var results_arr = try Array.from_values(allocator, results_list.items);
    defer results_arr.deinit();
    // Transfer ownership to caller via result pointer
    results_arr.handle.incref();
    result.* = results_arr.as_value().raw;
}

/// Runner callback: loads and benchmarks compiled .so artifacts.
fn run_callback(
    self_ptr: ?*anyopaque,
    args: [*c]const c.TVMFFIAny,
    num_args: i32,
    result: [*c]c.TVMFFIAny,
) callconv(.c) c_int {
    const state: *TuneState = @ptrCast(@alignCast(self_ptr orelse {
        log.err("run_callback: null state", .{});
        return -1;
    }));

    if (num_args != 1) {
        log.err("run_callback: expected 1 arg, got {d}", .{num_args});
        return -1;
    }

    run_callback_impl(state, args[0], result) catch |err| {
        log.err("run_callback failed: {s}", .{@errorName(err)});
        return -1;
    };
    return 0;
}

fn run_callback_impl(state: *TuneState, inputs_array_raw: c.TVMFFIAny, result: *c.TVMFFIAny) !void {
    const allocator = state.allocator;

    var inputs = try Array.wrap(.{ .raw = inputs_array_raw });
    defer inputs.deinit();

    const num_inputs = try inputs.len(allocator);
    log.info("running {d} candidates", .{num_inputs});

    var results_list = std.ArrayList(Value).empty;
    defer results_list.deinit(allocator);

    for (0..num_inputs) |i| {
        const input = try inputs.get(allocator, i);

        // Get artifact_path
        const path_val = api.get_field(input, "artifact_path") catch {
            try results_list.append(allocator, try make_runner_error(allocator, "no artifact_path"));
            continue;
        };
        var path_val_mut = path_val;
        const artifact_path_slice = path_val_mut.as_string(allocator) catch {
            try results_list.append(allocator, try make_runner_error(allocator, "bad path"));
            continue;
        };
        defer allocator.free(artifact_path_slice);

        const artifact_path = try std.fmt.allocPrintSentinel(allocator, "{s}", .{artifact_path_slice}, 0);
        defer allocator.free(artifact_path);

        // Load the candidate and resolve its entry function.
        var loaded = RuntimeModule.load_from_file(allocator, artifact_path) catch {
            try results_list.append(allocator, try make_runner_error(allocator, "load failed"));
            continue;
        };
        defer loaded.deinit();

        const func = loaded.get_function(allocator, "main", true) catch {
            try results_list.append(allocator, try make_runner_error(allocator, "GetFunction failed"));
            continue;
        };
        defer func.decref();
        const func_handle = func.as_object() orelse {
            try results_list.append(allocator, try make_runner_error(allocator, "null function"));
            continue;
        };

        // Allocate tensors and benchmark
        const run_time = benchmark_kernel(state, func_handle) catch {
            try results_list.append(allocator, try make_runner_error(allocator, "benchmark failed"));
            continue;
        };

        const future = try make_runner_success(allocator, run_time);
        try results_list.append(allocator, future);
        log.debug("candidate {d}: {d:.6}s", .{ i, run_time });
    }

    var results_arr = try Array.from_values(allocator, results_list.items);
    defer results_arr.deinit();
    // Transfer ownership to caller via result pointer
    results_arr.handle.incref();
    result.* = results_arr.as_value().raw;
}

/// Benchmark a compiled kernel function. Returns median time in seconds.
fn benchmark_kernel(state: *TuneState, func: c.TVMFFIObjectHandle) !f64 {
    const allocator = state.allocator;
    const io = state.io;
    const dev_type: dlpack.DeviceType = switch (state.target_kind) {
        .cpu => .cpu,
        .cuda => .cuda,
    };

    // Allocate the candidate's input tensors.
    var tensors = std.ArrayList(Tensor).empty;
    defer {
        for (tensors.items) |*t| t.deinit();
        tensors.deinit(allocator);
    }

    for (state.tensor_shapes) |shape| {
        var size: usize = 1;
        for (shape) |dim| size *= @intCast(dim);

        const data = try allocator.alloc(f32, size);
        defer allocator.free(data);
        for (data, 0..) |*v, idx| v.* = @as(f32, @floatFromInt(idx % 10)) * 0.1;

        const shape_copy = try allocator.dupe(i64, shape);
        defer allocator.free(shape_copy);

        const tensor = try Tensor.allocate(
            allocator,
            data,
            shape_copy,
            dev_type,
            state.device_ordinal,
        );
        try tensors.append(allocator, tensor);
    }

    // Build the packed call arguments.
    var call_args = try allocator.alloc(Value, tensors.items.len);
    defer allocator.free(call_args);
    for (tensors.items, 0..) |t, j| {
        call_args[j] = t.as_value();
    }

    // Warmup
    _ = try api.call_handle(allocator, func, call_args);

    // Timed runs (5 iterations, take median)
    const num_runs: usize = 5;
    var times: [5]f64 = std.mem.zeroes([5]f64);
    for (0..num_runs) |run_idx| {
        const start = std.Io.Timestamp.now(io, .awake);
        _ = try api.call_handle(allocator, func, call_args);
        const elapsed = start.untilNow(io, .awake);
        times[run_idx] = @as(f64, @floatFromInt(elapsed.toNanoseconds())) / 1e9;
    }
    std.mem.sort(f64, &times, {}, std.sort.asc(f64));
    return times[num_runs / 2];
}

// CUDA tensor intrinsics

/// Load and register CUDA tensor intrinsics (WMMA, MMA) from pre-serialized
/// JSON files in `artifacts/cuda_intrinsics/`.
///
/// These intrinsics are required for CUDA MetaSchedule tuning. Without them,
///  the schedule space generator cannot emit tensor core instructions.
/// Pre-generated by `scripts/generate_cuda_intrinsics.py`.
/// Safe to call multiple times (only loads once).
var cuda_intrinsics_loaded: bool = false;

fn load_cuda_intrinsics(io: std.Io, allocator: std.mem.Allocator) !void {
    if (cuda_intrinsics_loaded) return;

    const intrinsics_dir = "artifacts/cuda_intrinsics";
    var dir = std.Io.Dir.cwd().openDir(io, intrinsics_dir, .{ .iterate = true }) catch |err| {
        log.err("failed to open {s}: {s}", .{ intrinsics_dir, @errorName(err) });
        log.err("run: python3 scripts/generate_cuda_intrinsics.py", .{});
        return err;
    };
    defer dir.close(io);

    var loaded_count: usize = 0;
    var iter = dir.iterate();
    while (try iter.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".json")) continue;

        const json_data = dir.readFileAlloc(io, entry.name, allocator, .limited(1_000_000)) catch |err| {
            log.warn("failed to read {s}: {s}", .{ entry.name, @errorName(err) });
            continue;
        };
        defer allocator.free(json_data);

        const parsed = std.json.parseFromSlice(
            struct { name: []const u8, desc: []const u8, impl: []const u8 },
            allocator,
            json_data,
            .{},
        ) catch |err| {
            log.warn("failed to parse {s}: {s}", .{ entry.name, @errorName(err) });
            continue;
        };
        defer parsed.deinit();

        const data = parsed.value;
        const name_z = std.fmt.allocPrintSentinel(allocator, "{s}", .{data.name}, 0) catch continue;
        defer allocator.free(name_z);
        const desc_z = std.fmt.allocPrintSentinel(allocator, "{s}", .{data.desc}, 0) catch continue;
        defer allocator.free(desc_z);
        const impl_z = std.fmt.allocPrintSentinel(allocator, "{s}", .{data.impl}, 0) catch continue;
        defer allocator.free(impl_z);

        const desc_func = api.load_json(allocator, desc_z) catch continue;
        const impl_func = api.load_json(allocator, impl_z) catch continue;
        const intrin = tir.tensor_intrin(allocator, desc_func, impl_func) catch continue;
        tir.register_tensor_intrin(allocator, name_z, intrin, false) catch continue;

        loaded_count += 1;
    }

    log.info("loaded {d} CUDA tensor intrinsics", .{loaded_count});
    cuda_intrinsics_loaded = true;
}

// Helpers.

/// Persisted per-shape tuning state, stored as `state.json` in the work dir.
const PersistedTuneState = struct {
    next_candidate: u32 = 0,
};

fn read_tune_state(io: std.Io, allocator: std.mem.Allocator, work_cache: Cache) PersistedTuneState {
    const state_file = work_cache.join("state.json") catch return .{};
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, state_file.path(), allocator, .limited(4096)) catch return .{};
    defer allocator.free(bytes);
    const parsed = std.json.parseFromSlice(PersistedTuneState, allocator, bytes, .{
        .ignore_unknown_fields = true,
    }) catch return .{};
    defer parsed.deinit();
    return parsed.value;
}

fn write_tune_state(io: std.Io, allocator: std.mem.Allocator, work_cache: Cache, state: PersistedTuneState) void {
    const state_file = work_cache.join("state.json") catch return;
    const path = state_file.path();
    const bytes = std.json.Stringify.valueAlloc(allocator, state, .{}) catch return;
    defer allocator.free(bytes);
    var file = std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true }) catch return;
    defer file.close(io);
    file.writeStreamingAll(io, bytes) catch {};
}

fn make_noop_callback() !Value {
    const noop = struct {
        fn f(_: ?*anyopaque, _: [*c]const c.TVMFFIAny, _: i32, result: [*c]c.TVMFFIAny) callconv(.c) c_int {
            result.* = Value.none().raw;
            return 0;
        }
    }.f;
    return try api.create_packed_func(null, noop, null);
}

fn make_random_cost_model(allocator: std.mem.Allocator) !Value {
    const noop = struct {
        fn f(_: ?*anyopaque, _: [*c]const c.TVMFFIAny, _: i32, result: [*c]c.TVMFFIAny) callconv(.c) c_int {
            result.* = Value.none().raw;
            return 0;
        }
    }.f;

    const predict = struct {
        fn f(_: ?*anyopaque, args: [*c]const c.TVMFFIAny, num_args: i32, result: [*c]c.TVMFFIAny) callconv(.c) c_int {
            if (num_args < 3) {
                result.* = Value.none().raw;
                return -1;
            }
            const candidates = Value{ .raw = args[1] };
            const return_ptr = args[2];

            // Get candidate count via no-allocator path (callback constraint)
            var len_out: c.TVMFFIAny = Value.none().raw;
            var len_args_arr = [_]c.TVMFFIAny{candidates.raw};
            var name_arr: c.TVMFFIByteArray = .{ .data = "ffi.ArraySize", .size = 13 };
            var func_handle: c.TVMFFIObjectHandle = null;
            if (c.TVMFFIFunctionGetGlobal(&name_arr, &func_handle) != 0 or func_handle == null) {
                result.* = Value.none().raw;
                return -1;
            }
            defer _ = c.TVMFFIObjectDecRef(func_handle);
            if (c.TVMFFIFunctionCall(func_handle, &len_args_arr, 1, &len_out) != 0) {
                result.* = Value.none().raw;
                return -1;
            }
            const n: usize = @intCast((Value{ .raw = len_out }).as_int() orelse 0);

            // Write random scores
            if (return_ptr.type_index == c.kTVMFFIOpaquePtr and return_ptr.unnamed_1.v_ptr != null) {
                const scores: [*]f64 = @ptrCast(@alignCast(return_ptr.unnamed_1.v_ptr));
                var prng = std.Random.DefaultPrng.init(42);
                for (0..n) |i| scores[i] = prng.random().float(f64);
            }

            result.* = Value.none().raw;
            return 0;
        }
    }.f;

    const as_string = struct {
        fn f(_: ?*anyopaque, _: [*c]const c.TVMFFIAny, _: i32, result: [*c]c.TVMFFIAny) callconv(.c) c_int {
            const str_val = api.make_tvm_string("ZigRandomModel") catch {
                result.* = Value.none().raw;
                return -1;
            };
            result.* = str_val.raw;
            return 0;
        }
    }.f;

    const noop_val = try api.create_packed_func(null, noop, null);
    defer noop_val.decref();
    const predict_val = try api.create_packed_func(null, predict, null);
    defer predict_val.decref();
    const as_string_val = try api.create_packed_func(null, as_string, null);
    defer as_string_val.decref();

    return try MetaSchedule.py_cost_model(allocator, noop_val, noop_val, noop_val, predict_val, as_string_val);
}

fn register_cpu_count(allocator: std.mem.Allocator) !void {
    const cpu_count_cb = struct {
        fn f(_: ?*anyopaque, _: [*c]const c.TVMFFIAny, _: i32, result: [*c]c.TVMFFIAny) callconv(.c) c_int {
            result.* = Value.int(@intCast(std.Thread.getCpuCount() catch 1)).raw;
            return 0;
        }
    }.f;

    const func_val = try api.create_packed_func(null, cpu_count_cb, null);
    const func_handle = func_val.as_object() orelse return error.TvmCallFailed;
    defer _ = c.TVMFFIObjectDecRef(func_handle);

    for ([_][]const u8{ "meta_schedule._cpu_count", "meta_schedule.cpu_count" }) |name| {
        api.set_global(name, func_handle, true) catch {};
    }
    _ = allocator;
}

fn make_builder_error(allocator: std.mem.Allocator, msg: []const u8) !Value {
    const msg_z = try std.fmt.allocPrintSentinel(allocator, "{s}", .{msg}, 0);
    defer allocator.free(msg_z);
    return try MetaSchedule.builder_result(allocator, null, msg_z);
}

/// Create a RunnerFuture wrapping a RunnerResult with an error message.
fn make_runner_error(allocator: std.mem.Allocator, msg: []const u8) !Value {
    const msg_z = try std.fmt.allocPrintSentinel(allocator, "{s}", .{msg}, 0);
    defer allocator.free(msg_z);
    const rr = try MetaSchedule.runner_result(allocator, null, msg_z);
    return try MetaSchedule.runner_future(allocator, rr);
}

/// Create a RunnerFuture wrapping a RunnerResult with timing data.
fn make_runner_success(allocator: std.mem.Allocator, run_secs: f64) !Value {
    var run_secs_arr = try Array.from_values(allocator, &.{Value.float(run_secs)});
    defer run_secs_arr.deinit();
    const rr = try MetaSchedule.runner_result(allocator, run_secs_arr.as_value(), null);
    return try MetaSchedule.runner_future(allocator, rr);
}
