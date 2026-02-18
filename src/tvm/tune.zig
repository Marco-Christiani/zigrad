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
const IRModule = tvm_types.IRModule;
const Target = tvm_types.Target;
const TargetKind = @import("../ffi/tvm/types.zig").TargetKind;
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
    // Ensure TVM compiler is loaded
    try api.ensure_loaded(allocator);

    // Register NVRTC callback for CUDA targets
    if (kind == .cuda) {
        nvrtc_callback.register(allocator) catch |err| {
            log.warn("failed to register NVRTC callback: {s}", .{@errorName(err)});
        };
    }

    // Ensure work directory exists
    std.fs.cwd().makePath(opts.work_dir) catch {};

    // Register cpu_count helper required by MetaSchedule
    try register_cpu_count(allocator);

    // Schedule rules
    const rules_fn_name = switch (kind) {
        .cpu => "meta_schedule.ScheduleRuleDefaultLLVM",
        .cuda => "meta_schedule.ScheduleRuleDefaultCUDA",
    };
    const schedule_rules = try api.call_global(allocator, rules_fn_name, &.{});
    log.debug("created ScheduleRules", .{});

    // SpaceGenerator
    const space_gen = try api.call_global(allocator, "meta_schedule.SpaceGeneratorPostOrderApply", &.{
        Value.none(), // f_block_filter
        schedule_rules, // sch_rules
        Value.none(), // postprocs
        Value.none(), // mutator_probs
    });
    log.debug("created SpaceGenerator", .{});

    // Search strategy (evolutionary)
    const search_strategy = try api.call_global(allocator, "meta_schedule.SearchStrategyEvolutionarySearch", &.{
        Value.int(512), // population_size
        Value.float(0.2), // init_measured_ratio
        Value.int(50), // init_min_unmeasured
        Value.int(5), // max_fail_count
        Value.int(3), // genetic_num_iters
        Value.float(0.85), // genetic_mutate_prob
        Value.int(10), // genetic_max_fail_count
        Value.float(0.05), // eps_greedy
    });
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
    const structural_z = try api.cstr_alloc(allocator, "structural");
    defer allocator.free(structural_z);

    const database = try api.call_global(allocator, "meta_schedule.DatabaseJSONDatabase", &.{
        Value.str(workload_z),
        Value.str(record_z),
        Value.boolean(true), // allow_missing
        Value.str(structural_z),
    });
    log.debug("created JSONDatabase", .{});

    // TuneContext
    const main_z = try api.cstr_alloc(allocator, "main");
    defer allocator.free(main_z);

    const logger_val = try make_noop_callback();
    defer logger_val.decref();

    const tune_context = try api.call_global(allocator, "meta_schedule.TuneContext", &.{
        ir_mod.as_value(),
        target.as_value(),
        space_gen,
        search_strategy,
        Value.str(main_z), // task_name
        Value.int(1), // num_threads
        Value.int(42), // rand_state
        logger_val, // logger
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

    const builder = try api.call_global(allocator, "meta_schedule.BuilderPyBuilder", &.{builder_func});
    log.debug("created PyBuilder", .{});

    // Runner callback
    const runner_func = try api.create_packed_func(@ptrCast(&state), run_callback, null);
    defer runner_func.decref();

    const runner = try api.call_global(allocator, "meta_schedule.RunnerPyRunner", &.{runner_func});
    log.debug("created PyRunner", .{});

    // Cost model (random)
    const cost_model = try make_random_cost_model();
    log.debug("created PyCostModel", .{});

    // Task scheduler
    const task_scheduler = try api.call_global(allocator, "meta_schedule.TaskSchedulerGradientBased", &.{
        logger_val, // f_logging
        Value.float(0.8), // alpha
        Value.int(3), // window_size
        Value.int(42), // seed
    });
    log.debug("created TaskScheduler", .{});

    // Run tuning
    log.info("starting tuning ({d} max trials, {d} per iter)...", .{ opts.max_trials, opts.trials_per_iter });

    const contexts_array = try api.call_global(allocator, "ffi.Array", &.{tune_context});
    const weights_array = try api.call_global(allocator, "ffi.Array", &.{Value.float(1.0)});

    const add_to_db = try api.call_global(allocator, "meta_schedule.MeasureCallbackAddToDatabase", &.{});
    const callbacks_array = try api.call_global(allocator, "ffi.Array", &.{add_to_db});

    _ = api.call_global(allocator, "meta_schedule.TaskSchedulerTune", &.{
        task_scheduler,
        contexts_array,
        weights_array,
        Value.int(@intCast(opts.max_trials)),
        Value.int(@intCast(opts.max_trials)),
        Value.int(@intCast(opts.trials_per_iter)),
        builder,
        runner,
        callbacks_array,
        database,
        cost_model,
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
    const inputs_array = Value{ .raw = inputs_array_raw };

    // Get array length
    const len_val = try api.call_global(allocator, "ffi.ArraySize", &.{inputs_array});
    const num_inputs: usize = @intCast(len_val.as_int() orelse return error.TvmCallFailed);
    log.info("building {d} candidates", .{num_inputs});

    var results_list = std.ArrayList(Value).empty;
    defer results_list.deinit(allocator);

    for (0..num_inputs) |i| {
        const input = try api.call_global(allocator, "ffi.ArrayGetItem", &.{ inputs_array, Value.int(@intCast(i)) });

        // Extract mod from BuilderInput
        const mod_val = ffi_get_attr(allocator, input, "mod") catch {
            try results_list.append(allocator, try make_builder_error(allocator, "failed to get mod"));
            continue;
        };

        // Lower and build
        var ir_mod = IRModule{ .handle = .{ .ptr = mod_val.as_object() orelse {
            try results_list.append(allocator, try make_builder_error(allocator, "mod not an object"));
            continue;
        } } };
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

        const br = try api.call_global(allocator, "meta_schedule.BuilderResult", &.{
            Value.str(so_z), Value.none(),
        });
        try results_list.append(allocator, br);
        log.debug("built candidate {d}", .{i});
    }

    // Convert to raw array for ffi.Array
    const raw_results = try allocator.alloc(Value, results_list.items.len);
    defer allocator.free(raw_results);
    @memcpy(raw_results, results_list.items);

    const array_result = try api.call_global(allocator, "ffi.Array", raw_results);
    result.* = array_result.raw;
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
    const inputs_array = Value{ .raw = inputs_array_raw };

    const len_val = try api.call_global(allocator, "ffi.ArraySize", &.{inputs_array});
    const num_inputs: usize = @intCast(len_val.as_int() orelse return error.TvmCallFailed);
    log.info("running {d} candidates", .{num_inputs});

    var results_list = std.ArrayList(Value).empty;
    defer results_list.deinit(allocator);

    for (0..num_inputs) |i| {
        const input = try api.call_global(allocator, "ffi.ArrayGetItem", &.{ inputs_array, Value.int(@intCast(i)) });

        // Get artifact_path
        const path_val = ffi_get_attr(allocator, input, "artifact_path") catch {
            try results_list.append(allocator, try make_runner_error(allocator, "no artifact_path"));
            continue;
        };
        var path_val_mut = path_val;
        const artifact_path = path_val_mut.as_string(allocator) catch {
            try results_list.append(allocator, try make_runner_error(allocator, "bad path"));
            continue;
        };
        defer allocator.free(artifact_path);

        // Load module
        const path_z = api.cstr_alloc(allocator, artifact_path) catch {
            try results_list.append(allocator, try make_runner_error(allocator, "alloc failed"));
            continue;
        };
        defer allocator.free(path_z);

        const loaded = api.call_global(allocator, "runtime.ModuleLoadFromFile", &.{
            Value.str(path_z), Value.str(""),
        }) catch {
            try results_list.append(allocator, try make_runner_error(allocator, "load failed"));
            continue;
        };
        defer loaded.decref();

        // Get main function
        const main_z = api.cstr_alloc(allocator, "main") catch {
            try results_list.append(allocator, try make_runner_error(allocator, "alloc failed"));
            continue;
        };
        defer allocator.free(main_z);

        const func = api.call_global(allocator, "ffi.ModuleGetFunction", &.{
            loaded, Value.str(main_z), Value.boolean(true),
        }) catch {
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

    const raw_results = try allocator.alloc(Value, results_list.items.len);
    defer allocator.free(raw_results);
    @memcpy(raw_results, results_list.items);

    const array_result = try api.call_global(allocator, "ffi.Array", raw_results);
    result.* = array_result.raw;
}

/// Benchmark a compiled kernel function. Returns median time in seconds.
fn benchmark_kernel(state: *TuneState, func: c.TVMFFIObjectHandle) !f64 {
    const allocator = state.allocator;
    const dev_type: i32 = switch (state.target_kind) {
        .cpu => @intFromEnum(dlpack.DeviceType.cpu),
        .cuda => @intFromEnum(dlpack.DeviceType.cuda),
    };

    // Allocate tensors
    var tensor_handles = std.ArrayList(c.TVMFFIObjectHandle).empty;
    defer {
        for (tensor_handles.items) |t| _ = c.TVMFFIObjectDecRef(t);
        tensor_handles.deinit(allocator);
    }

    for (state.tensor_shapes) |shape| {
        var size: usize = 1;
        for (shape) |dim| size *= @intCast(dim);

        const data = try allocator.alloc(f32, size);
        defer allocator.free(data);
        for (data, 0..) |*v, idx| v.* = @as(f32, @floatFromInt(idx % 10)) * 0.1;

        const shape_copy = try allocator.dupe(i64, shape);
        defer allocator.free(shape_copy);

        // Allocate via TVM FFI
        var shape_args = try allocator.alloc(Value, shape.len);
        defer allocator.free(shape_args);
        for (shape, 0..) |dim, j| shape_args[j] = Value.int(dim);

        const shape_obj = try api.call_global(allocator, "ffi.Shape", shape_args);
        defer shape_obj.decref();

        var dtype_raw = std.mem.zeroes(c.TVMFFIAny);
        dtype_raw.type_index = c.kTVMFFIDataType;
        dtype_raw.unnamed_1.v_dtype = .{ .code = @intCast(@intFromEnum(dlpack.DataTypeCode.float)), .bits = 32, .lanes = 1 };
        const dtype_val = Value{ .raw = dtype_raw };

        var device_raw = std.mem.zeroes(c.TVMFFIAny);
        device_raw.type_index = c.kTVMFFIDevice;
        device_raw.unnamed_1.v_device = .{ .device_type = @intCast(dev_type), .device_id = 0 };
        const device_val = Value{ .raw = device_raw };

        const tensor = try api.call_global(allocator, "runtime.TVMTensorAllocWithScope", &.{
            shape_obj, dtype_val, device_val, Value.none(),
        });

        // Copy data
        var data_ptr_raw = std.mem.zeroes(c.TVMFFIAny);
        data_ptr_raw.type_index = c.kTVMFFIOpaquePtr;
        data_ptr_raw.unnamed_1.v_int64 = @bitCast(@intFromPtr(data.ptr));
        const nbytes = data.len * @sizeOf(f32);
        _ = try api.call_global(allocator, "runtime.TVMTensorCopyFromBytes", &.{
            tensor, Value{ .raw = data_ptr_raw }, Value.int(@intCast(nbytes)),
        });

        try tensor_handles.append(allocator, tensor.as_object() orelse return error.TvmCallFailed);
    }

    // Build call args
    var call_args = try allocator.alloc(c.TVMFFIAny, tensor_handles.items.len);
    defer allocator.free(call_args);
    for (tensor_handles.items, 0..) |t, j| {
        call_args[j] = Value.from_object(t, c.kTVMFFITensor).raw;
    }

    // Warmup
    var out: c.TVMFFIAny = undefined;
    try api.call(allocator, func, call_args, &out);

    // Timed runs (5 iterations, take median)
    const num_runs: usize = 5;
    var times: [5]f64 = std.mem.zeroes([5]f64);
    for (0..num_runs) |run_idx| {
        const start = std.time.nanoTimestamp();
        try api.call(allocator, func, call_args, &out);
        const end = std.time.nanoTimestamp();
        times[run_idx] = @as(f64, @floatFromInt(end - start)) / 1e9;
    }
    std.mem.sort(f64, &times, {}, std.sort.asc(f64));
    return times[num_runs / 2];
}

// ============================================================================
// Helpers
// ============================================================================

fn ffi_get_attr(allocator: std.mem.Allocator, obj: Value, attr_name: []const u8) !Value {
    const attr_z = try api.cstr_alloc(allocator, attr_name);
    defer allocator.free(attr_z);
    return api.call_global(allocator, "ir.BaseFunc_Attrs", &.{ obj, Value.str(attr_z) }) catch {
        // Try generic object attribute access
        return api.call_global(allocator, "ffi.ObjectGetAttr", &.{ obj, Value.str(attr_z) });
    };
}

fn make_noop_callback() !Value {
    const noop = struct {
        fn f(_: ?*anyopaque, _: [*c]const c.TVMFFIAny, _: i32, result: [*c]c.TVMFFIAny) callconv(.c) c_int {
            result.* = Value.none().raw;
            return 0;
        }
    }.f;
    return api.create_packed_func(null, noop, null);
}

fn make_random_cost_model() !Value {
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

            // Get candidate count (no-allocator path)
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

    return api.call_global(std.heap.c_allocator, "meta_schedule.CostModelPyCostModel", &.{
        noop_val, // f_load
        noop_val, // f_save
        noop_val, // f_update
        predict_val, // f_predict
        as_string_val, // f_as_string
    });
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
    return api.call_global(allocator, "meta_schedule.BuilderResult", &.{ Value.none(), Value.str(msg_z) });
}

fn make_runner_error(allocator: std.mem.Allocator, msg: []const u8) !Value {
    const msg_z = try api.cstr_alloc(allocator, msg);
    defer allocator.free(msg_z);
    return api.call_global(allocator, "meta_schedule.RunnerFuture", &.{
        Value.none(), // run_secs
        Value.str(msg_z), // error_msg
    });
}

fn make_runner_success(allocator: std.mem.Allocator, run_secs: f64) !Value {
    return api.call_global(allocator, "meta_schedule.RunnerFuture", &.{
        Value.float(run_secs),
        Value.none(), // no error
    });
}
