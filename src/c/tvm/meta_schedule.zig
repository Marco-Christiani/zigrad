//! MetaSchedule constructors for TVM auto-tuning objects.
//!
//! Covers the current `tvm/meta_schedule/` constructors used by tuning.
//!
//! Functions use compile-time checked names and return values managed by the
//!  caller.
const std = @import("std");
const api = @import("api.zig");
const c = @import("c.zig");
const Value = api.Value;
const TargetKind = @import("../../tvm/config.zig").TargetKind;

/// MetaSchedule object constructors.
pub const MetaSchedule = struct {
    /// Create schedule rules for the given target kind.
    pub fn schedule_rules(allocator: std.mem.Allocator, kind: TargetKind) !Value {
        const name = switch (kind) {
            .cpu => "meta_schedule.ScheduleRuleDefaultLLVM",
            .cuda => "meta_schedule.ScheduleRuleDefaultCUDA",
        };
        return try api.call_global(allocator, name, &.{});
    }

    /// Create a SpaceGenerator with post-order apply.
    pub fn space_generator(allocator: std.mem.Allocator, rules: Value) !Value {
        return try api.call_global(allocator, "meta_schedule.SpaceGeneratorPostOrderApply", &.{
            Value.none(), // f_block_filter
            rules,
            Value.none(), // postprocs
            Value.none(), // mutator_probs
        });
    }

    pub const SearchStrategyOpts = struct {
        population_size: i64 = 512,
        init_measured_ratio: f64 = 0.2,
        init_min_unmeasured: i64 = 50,
        max_fail_count: i64 = 5,
        genetic_num_iters: i64 = 3,
        genetic_mutate_prob: f64 = 0.85,
        genetic_max_fail_count: i64 = 10,
        eps_greedy: f64 = 0.05,
    };

    /// Create an evolutionary search strategy.
    pub fn search_strategy(allocator: std.mem.Allocator, opts: SearchStrategyOpts) !Value {
        return try api.call_global(allocator, "meta_schedule.SearchStrategyEvolutionarySearch", &.{
            Value.int(opts.population_size),
            Value.float(opts.init_measured_ratio),
            Value.int(opts.init_min_unmeasured),
            Value.int(opts.max_fail_count),
            Value.int(opts.genetic_num_iters),
            Value.float(opts.genetic_mutate_prob),
            Value.int(opts.genetic_max_fail_count),
            Value.float(opts.eps_greedy),
        });
    }

    /// Create a JSON database for persisting tuning records.
    pub fn json_database(
        allocator: std.mem.Allocator,
        workload_path: [:0]const u8,
        record_path: [:0]const u8,
    ) !Value {
        return try api.call_global(allocator, "meta_schedule.DatabaseJSONDatabase", &.{
            Value.str(workload_path),
            Value.str(record_path),
            Value.boolean(true), // allow_missing
            Value.str("structural"),
        });
    }

    pub const TuneContextOpts = struct {
        ir_mod: Value,
        target: Value,
        space_gen: Value,
        search_strat: Value,
        task_name: [:0]const u8,
        num_threads: i64 = 1,
        rand_state: i64 = 42,
        logger: Value,
    };

    /// Create a TuneContext.
    pub fn tune_context(allocator: std.mem.Allocator, opts: TuneContextOpts) !Value {
        return try api.call_global(allocator, "meta_schedule.TuneContext", &.{
            opts.ir_mod,
            opts.target,
            opts.space_gen,
            opts.search_strat,
            Value.str(opts.task_name),
            Value.int(opts.num_threads),
            Value.int(opts.rand_state),
            opts.logger,
        });
    }

    /// Create a PyBuilder wrapping a packed function callback.
    pub fn py_builder(allocator: std.mem.Allocator, func: Value) !Value {
        return try api.call_global(allocator, "meta_schedule.BuilderPyBuilder", &.{func});
    }

    /// Create a PyRunner wrapping a packed function callback.
    pub fn py_runner(allocator: std.mem.Allocator, func: Value) !Value {
        return try api.call_global(allocator, "meta_schedule.RunnerPyRunner", &.{func});
    }

    /// Create a PyCostModel with load/save/update/predict/as_string callbacks.
    pub fn py_cost_model(
        allocator: std.mem.Allocator,
        f_load: Value,
        f_save: Value,
        f_update: Value,
        f_predict: Value,
        f_as_string: Value,
    ) !Value {
        return try api.call_global(allocator, "meta_schedule.CostModelPyCostModel", &.{
            f_load, f_save, f_update, f_predict, f_as_string,
        });
    }

    pub const TaskSchedulerOpts = struct {
        logger: Value,
        alpha: f64 = 0.8,
        window_size: i64 = 3,
        seed: i64 = 42,
    };

    /// Create a gradient-based task scheduler.
    pub fn task_scheduler(allocator: std.mem.Allocator, opts: TaskSchedulerOpts) !Value {
        return try api.call_global(allocator, "meta_schedule.TaskSchedulerGradientBased", &.{
            opts.logger,
            Value.float(opts.alpha),
            Value.int(opts.window_size),
            Value.int(opts.seed),
        });
    }

    pub const RunTuneOpts = struct {
        scheduler: Value,
        contexts: Value,
        weights: Value,
        max_trials: i64,
        max_trials_global: i64,
        trials_per_iter: i64,
        builder: Value,
        runner: Value,
        callbacks: Value,
        database: Value,
        cost_model: Value,
    };

    /// Run the tuning loop.
    pub fn run_tune(allocator: std.mem.Allocator, opts: RunTuneOpts) !void {
        _ = try api.call_global(allocator, "meta_schedule.TaskSchedulerTune", &.{
            opts.scheduler,
            opts.contexts,
            opts.weights,
            Value.int(opts.max_trials),
            Value.int(opts.max_trials_global),
            Value.int(opts.trials_per_iter),
            opts.builder,
            opts.runner,
            opts.callbacks,
            opts.database,
            opts.cost_model,
        });
    }

    /// Create a BuilderResult (success or error).
    pub fn builder_result(allocator: std.mem.Allocator, artifact_path: ?[:0]const u8, error_msg: ?[:0]const u8) !Value {
        return try api.call_global(allocator, "meta_schedule.BuilderResult", &.{
            if (artifact_path) |p| Value.str(p) else Value.none(),
            if (error_msg) |m| Value.str(m) else Value.none(),
        });
    }

    /// Create a RunnerResult (success or error).
    ///
    /// For success: pass `run_secs` as an Array of floats, `error_msg` as null.
    /// For error: pass `run_secs` as null, `error_msg` as the message.
    pub fn runner_result(allocator: std.mem.Allocator, run_secs: ?Value, error_msg: ?[:0]const u8) !Value {
        return try api.call_global(allocator, "meta_schedule.RunnerResult", &.{
            run_secs orelse Value.none(),
            if (error_msg) |m| Value.str(m) else Value.none(),
        });
    }

    /// Create a RunnerFuture wrapping a RunnerResult with trivial done/result callbacks.
    pub fn runner_future(allocator: std.mem.Allocator, result: Value) !Value {
        const done_cb = struct {
            fn f(_: ?*anyopaque, _: [*c]const c.TVMFFIAny, _: i32, ret: [*c]c.TVMFFIAny) callconv(.c) c_int {
                ret.* = Value.boolean(true).raw;
                return 0;
            }
        }.f;
        const f_done = try api.create_packed_func(null, done_cb, null);
        defer f_done.decref();

        const ResultHolder = struct {
            raw: c.TVMFFIAny,
            fn callback(self_ptr: ?*anyopaque, _: [*c]const c.TVMFFIAny, _: i32, ret: [*c]c.TVMFFIAny) callconv(.c) c_int {
                const self: *@This() = @ptrCast(@alignCast(self_ptr orelse return -1));
                ret.* = self.raw;
                return 0;
            }
        };
        const holder = try std.heap.c_allocator.create(ResultHolder);
        holder.raw = result.raw;

        const f_result = try api.create_packed_func(@ptrCast(holder), ResultHolder.callback, struct {
            fn dtor(self_ptr: ?*anyopaque) callconv(.c) void {
                const self: *ResultHolder = @ptrCast(@alignCast(self_ptr orelse return));
                std.heap.c_allocator.destroy(self);
            }
        }.dtor);
        defer f_result.decref();

        return try api.call_global(allocator, "meta_schedule.RunnerFuture", &.{ f_done, f_result });
    }

    /// Create a MeasureCallbackAddToDatabase callback.
    pub fn add_to_database(allocator: std.mem.Allocator) !Value {
        return try api.call_global(allocator, "meta_schedule.MeasureCallbackAddToDatabase", &.{});
    }
};
