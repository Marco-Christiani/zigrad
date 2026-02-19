//! MetaSchedule autotuning for TVM.
//!
//! Runs TVM's MetaSchedule search to find optimal schedules for a given
//! IRModule and target. Uses a random cost model with evolutionary search.
//! Builder and runner callbacks receive provider state through TVM's
//! userdata pointer — no globals.
const std = @import("std");
const tvm_types = @import("../ffi/tvm/types.zig");
const api = @import("../ffi/tvm/api.zig");
const c = @import("../ffi/tvm/c.zig");
const dlpack = @import("../ffi/dlpack.zig");
const Value = api.Value;
const Array = tvm_types.Array;
const IRModule = tvm_types.IRModule;
const RuntimeModule = tvm_types.RuntimeModule;
const Target = tvm_types.Target;
const Tensor = tvm_types.Tensor;
const TargetKind = tvm_types.TargetKind;
const MetaSchedule = tvm_types.MetaSchedule;
const nvrtc_callback = @import("nvrtc_callback.zig");

const log = std.log.scoped(.@"zg/tvm_tune");

pub const TuneOpts = struct {
    work_dir: []const u8 = "artifacts/tvm_cache",
    max_trials: u32 = 64,
    trials_per_iter: u32 = 16,
};

/// Persistent state for MetaSchedule callbacks. Passed as userdata to
/// builder/runner callbacks via TVM's TVMFFIFunctionCreate self pointer.
const TuneState = struct {
    allocator: std.mem.Allocator,
    target: Target,
    target_kind: TargetKind,
    work_dir: []const u8,
    build_counter: u32 = 0,
    /// Tensor shapes for the workload (A, B, C for matmul).
    tensor_shapes: []const []const i64,
};

/// Tune an IRModule via MetaSchedule, returning the tuned IRModule.
///
/// Runs evolutionary search with a random cost model. The builder callback
/// compiles TIR candidates to .so artifacts; the runner callback loads and
/// benchmarks them. Results are persisted to a JSON database in work_dir.
pub fn tune(
    allocator: std.mem.Allocator,
    ir_mod: IRModule,
    target: Target,
    kind: TargetKind,
    tensor_shapes: []const []const i64,
    opts: TuneOpts,
) !void {
    try api.ensure_loaded(allocator);

    if (kind == .cuda) {
        nvrtc_callback.register(allocator) catch |err| {
            log.warn("failed to register NVRTC callback: {s}", .{@errorName(err)});
        };
    }

    std.fs.cwd().makePath(opts.work_dir) catch {};

    try register_cpu_count(allocator);

    const schedule_rules = try MetaSchedule.schedule_rules(allocator, kind);
    log.debug("created ScheduleRules", .{});

    const space_gen = try MetaSchedule.space_generator(allocator, schedule_rules);
    log.debug("created SpaceGenerator", .{});

    const search_strategy = try MetaSchedule.search_strategy(allocator, .{});
    log.debug("created SearchStrategy", .{});

    // JSON database
    const workload_path = try std.fmt.allocPrint(allocator, "{s}/workload.json", .{opts.work_dir});
    defer allocator.free(workload_path);
    const record_path = try std.fmt.allocPrint(allocator, "{s}/tuning_record.json", .{opts.work_dir});
    defer allocator.free(record_path);

    const workload_z = try api.cstr_alloc(allocator, workload_path);
    defer allocator.free(workload_z);
    const record_z = try api.cstr_alloc(allocator, record_path);
    defer allocator.free(record_z);

    const database = try MetaSchedule.json_database(allocator, workload_z, record_z);
    log.debug("created JSONDatabase", .{});

    // TuneContext
    const main_z = try api.cstr_alloc(allocator, "main");
    defer allocator.free(main_z);

    const logger_val = try make_noop_callback();
    defer logger_val.decref();

    const tune_context = try MetaSchedule.tune_context(allocator, .{
        .ir_mod = ir_mod.as_value(),
        .target = target.as_value(),
        .space_gen = space_gen,
        .search_strat = search_strategy,
        .task_name = main_z,
        .logger = logger_val,
    });
    log.debug("created TuneContext", .{});

    // Tune state (passed as userdata to callbacks)
    var state = TuneState{
        .allocator = allocator,
        .target = target,
        .target_kind = kind,
        .work_dir = opts.work_dir,
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

    log.info("tuning complete. results: {s}", .{record_path});
}

// ============================================================================
// Callbacks
// ============================================================================

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

        // Lower and build
        var ir_mod = IRModule{ .handle = .{ .ptr = mod_val.as_object() orelse {
            try results_list.append(allocator, try make_builder_error(allocator, "mod not an object"));
            continue;
        } }, .type_index = mod_val.raw.type_index };
        ir_mod.handle.incref();

        const built_mod_result = tvm_types.lower_and_build(allocator, &ir_mod, state.target, state.target_kind);
        var built_mod = built_mod_result catch {
            try results_list.append(allocator, try make_builder_error(allocator, "compilation failed"));
            continue;
        };
        defer built_mod.deinit();

        // Export to .so
        const build_id = @atomicRmw(u32, &state.build_counter, .Add, 1, .seq_cst);
        const so_path = try std.fmt.allocPrint(allocator, "{s}/candidate_{d}.so", .{ state.work_dir, build_id });

        built_mod.export_shared(allocator, so_path, state.target_kind) catch {
            allocator.free(so_path);
            try results_list.append(allocator, try make_builder_error(allocator, "export failed"));
            continue;
        };

        const so_z = api.cstr_alloc(allocator, so_path) catch {
            allocator.free(so_path);
            try results_list.append(allocator, try make_builder_error(allocator, "alloc failed"));
            continue;
        };
        allocator.free(so_path);
        defer allocator.free(so_z);

        const br = try MetaSchedule.builder_result(allocator, so_z, null);
        try results_list.append(allocator, br);
        log.debug("built candidate {d}", .{i});
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
        const artifact_path = path_val_mut.as_string(allocator) catch {
            try results_list.append(allocator, try make_runner_error(allocator, "bad path"));
            continue;
        };
        defer allocator.free(artifact_path);

        // Load module and get main function via typed wrappers
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
    const dev_type: dlpack.DeviceType = switch (state.target_kind) {
        .cpu => .cpu,
        .cuda => .cuda,
    };

    // Allocate tensors via typed Tensor wrapper
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

        const tensor = try Tensor.allocate(allocator, data, shape_copy, dev_type);
        try tensors.append(allocator, tensor);
    }

    // Build call args from typed Tensor values
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
        const start = std.time.nanoTimestamp();
        _ = try api.call_handle(allocator, func, call_args);
        const end = std.time.nanoTimestamp();
        times[run_idx] = @as(f64, @floatFromInt(end - start)) / 1e9;
    }
    std.mem.sort(f64, &times, {}, std.sort.asc(f64));
    return times[num_runs / 2];
}

// ============================================================================
// Helpers
// ============================================================================

fn make_noop_callback() !Value {
    const noop = struct {
        fn f(_: ?*anyopaque, _: [*c]const c.TVMFFIAny, _: i32, result: [*c]c.TVMFFIAny) callconv(.c) c_int {
            result.* = Value.none().raw;
            return 0;
        }
    }.f;
    return api.create_packed_func(null, noop, null);
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

    return MetaSchedule.py_cost_model(allocator, noop_val, noop_val, noop_val, predict_val, as_string_val);
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
    const msg_z = try api.cstr_alloc(allocator, msg);
    defer allocator.free(msg_z);
    return MetaSchedule.builder_result(allocator, null, msg_z);
}

/// Create a RunnerFuture wrapping a RunnerResult with an error message.
fn make_runner_error(allocator: std.mem.Allocator, msg: []const u8) !Value {
    const msg_z = try api.cstr_alloc(allocator, msg);
    defer allocator.free(msg_z);
    const rr = try MetaSchedule.runner_result(allocator, null, msg_z);
    return MetaSchedule.runner_future(allocator, rr);
}

/// Create a RunnerFuture wrapping a RunnerResult with timing data.
fn make_runner_success(allocator: std.mem.Allocator, run_secs: f64) !Value {
    var run_secs_arr = try Array.from_values(allocator, &.{Value.float(run_secs)});
    defer run_secs_arr.deinit();
    const rr = try MetaSchedule.runner_result(allocator, run_secs_arr.as_value(), null);
    return MetaSchedule.runner_future(allocator, rr);
}
