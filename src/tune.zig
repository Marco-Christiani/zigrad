//! Kernel tuning for the KP system.
//!
//! Walks a PR program for provider-request functions, invokes providers to compile
//!  each candidate, and records selections in a `KernelStore`.
//! Provider dispatch entries are registered in a `DispatchRegistry` for
//!  execute-time resolution.
//!
//! This decouples tuning from kernelization. Kernelization consults a
//!  pre-computed store and does not call providers.
//!
//! Usage:
//! ```zig
//! var result = try tune(io, allocator, &program, providers, .{
//!     .device = selected_device,
//!     .evaluator = evaluator,
//! });
//! defer result.deinit();
//! // Pass `result.store` to the PR kernelization operation.
//! // Pass `result.dispatch_registry` to the execution integration.
//! ```
const std = @import("std");
const device = @import("device.zig");
const fingerprint = @import("pr/analysis/fingerprint.zig");
const pr = @import("pr/pr.zig");
const kernel = @import("kernel.zig");

const log = std.log.scoped(.@"zg/tune");

/// Options for the tuning process.
pub const TuneOpts = struct {
    /// Print a summary table of kernel selections after completion.
    dump_results: bool = false,
    /// Device for provider target resolution.
    device: device.Device,

    /// Evaluates the original callable and compiled provider candidates.
    ///
    /// A null evaluator retains the original callable. Compilation alone does
    ///  not establish that a provider candidate is preferable.
    evaluator: ?Evaluator = null,
};

/// Candidate selected by an evaluator.
pub const Evaluation = struct {
    candidate: union(enum) {
        original,
        provider: usize,
    },
    /// Evidence summary copied into the kernel store.
    reason: []const u8,
    /// Aggregate timings when the decision came from target measurement.
    timing: ?kernel.TimingEvidence = null,
};

/// Failures produced while comparing executable candidates.
pub const EvaluationError = error{
    EvaluationFailed,
    InvalidEvaluation,
} || std.mem.Allocator.Error;

/// Compiled provider implementation available to an evaluator.
pub const AvailableCandidate = struct {
    provider_name: []const u8,
    artifact: kernel.Artifact,
};

/// Type-erased selection among an original callable and provider artifacts.
///
/// The implementation supplies the evidence and policy supporting its result.
pub const Evaluator = struct {
    ptr: *anyopaque,
    evaluate_fn: *const fn (
        ptr: *anyopaque,
        func: pr.Function,
        candidates: []const AvailableCandidate,
        selected_device: device.Device,
        allocator: std.mem.Allocator,
    ) EvaluationError!Evaluation,

    /// Compare both executable implementations of one callable.
    pub fn evaluate(
        self: Evaluator,
        func: pr.Function,
        candidates: []const AvailableCandidate,
        selected_device: device.Device,
        allocator: std.mem.Allocator,
    ) EvaluationError!Evaluation {
        return try self.evaluate_fn(
            self.ptr,
            func,
            candidates,
            selected_device,
            allocator,
        );
    }
};

/// Executable implementation passed to a measurement callback.
pub const MeasurableCandidate = union(enum) {
    original,
    provider: *const AvailableCandidate,
};

/// Comparable result returned by a target-specific measurement callback.
pub const Measurement = struct {
    elapsed_ns: u64,
    /// Whether this implementation matched the original callable's results.
    correct: bool,
};

/// Selects the fastest correct candidate from target measurements.
///
/// The callback owns warmup, repeated sampling, synchronization, input choice,
///  and aggregation. Every returned duration must cover the same callable
///  boundary under the same measurement policy.
pub const MeasuredEvaluator = struct {
    ptr: *anyopaque,
    measure_fn: *const fn (
        ptr: *anyopaque,
        func: pr.Function,
        candidate: MeasurableCandidate,
        selected_device: device.Device,
        allocator: std.mem.Allocator,
    ) EvaluationError!Measurement,

    /// Return the evaluator interface used by `tune`.
    pub fn interface(self: *MeasuredEvaluator) Evaluator {
        return .{
            .ptr = @ptrCast(self),
            .evaluate_fn = evaluate,
        };
    }

    fn evaluate(
        ptr: *anyopaque,
        func: pr.Function,
        candidates: []const AvailableCandidate,
        selected_device: device.Device,
        allocator: std.mem.Allocator,
    ) EvaluationError!Evaluation {
        const self: *MeasuredEvaluator = @ptrCast(@alignCast(ptr));
        const original = try self.measure_fn(
            self.ptr,
            func,
            .original,
            selected_device,
            allocator,
        );
        if (!original.correct) return error.EvaluationFailed;

        var fastest_ns = original.elapsed_ns;
        var fastest: ?usize = null;
        for (candidates, 0..) |*candidate, index| {
            const measured = try self.measure_fn(
                self.ptr,
                func,
                .{ .provider = candidate },
                selected_device,
                allocator,
            );
            if (measured.correct and measured.elapsed_ns < fastest_ns) {
                fastest_ns = measured.elapsed_ns;
                fastest = index;
            }
        }

        return if (fastest) |index|
            .{
                .candidate = .{ .provider = index },
                .reason = "provider measured faster than the original callable",
                .timing = .{
                    .original_ns = original.elapsed_ns,
                    .selected_ns = fastest_ns,
                },
            }
        else
            .{
                .candidate = .original,
                .reason = "no correct provider measured faster than the original callable",
                .timing = .{
                    .original_ns = original.elapsed_ns,
                    .selected_ns = fastest_ns,
                },
            };
    }
};

/// Tuning result containing populated store and dispatch registry.
///
/// Caller owns both and must call `deinit()` when done.
pub const TuneResult = struct {
    store: kernel.KernelStore,
    dispatch_registry: kernel.DispatchRegistry,

    pub fn deinit(self: *TuneResult) void {
        self.dispatch_registry.deinit();
        self.store.deinit();
    }
};

/// Tune a program: walk all functions for provider requests, invoke
/// providers, and record selections in the returned store.
///
/// Provider regions must first pass through
///  `pr.transform.kernelize.OutlineCandidates`.
///
/// On success, callers pass `result.store` to a kernelization operation and
///  `result.dispatch_registry` to the execution integration. Providers are
///  finalized before return.
pub fn tune(
    io: std.Io,
    allocator: std.mem.Allocator,
    program: *const pr.Program,
    providers: []const kernel.KernelProvider,
    opts: TuneOpts,
) !TuneResult {
    try kernel.validate_providers(providers);
    try kernel.require_outlined_requests(program);
    defer for (providers) |provider| provider.finalize();

    var store = kernel.KernelStore.init(allocator);
    errdefer store.deinit();

    var dispatch_registry = kernel.DispatchRegistry.init(allocator);
    errdefer dispatch_registry.deinit();

    const tune_start = std.Io.Timestamp.now(io, .awake);

    var total_selected: usize = 0;
    var total_dedup: usize = 0;
    var total_original: usize = 0;

    for (program.functions()) |func| {
        const stats = try tune_function(
            allocator,
            func,
            providers,
            &store,
            &dispatch_registry,
            opts.device,
            opts.evaluator,
        );
        total_selected += stats.selected;
        total_dedup += stats.dedup;
        total_original += stats.original;
    }

    log.info("tuning completed: {d} provider, {d} dedup, {d} original in {d:.2}ms", .{
        total_selected, total_dedup, total_original, ns_to_ms(@intCast(tune_start.untilNow(io, .awake).toNanoseconds())),
    });

    if (opts.dump_results) {
        dump_store_summary(io, &store);
    }

    return .{
        .store = store,
        .dispatch_registry = dispatch_registry,
    };
}

const TuneStats = struct {
    selected: usize = 0,
    dedup: usize = 0,
    original: usize = 0,
};

fn tune_function(
    allocator: std.mem.Allocator,
    func: pr.Function,
    providers: []const kernel.KernelProvider,
    store: *kernel.KernelStore,
    dispatch_registry: *kernel.DispatchRegistry,
    selected_device: device.Device,
    evaluator: ?Evaluator,
) !TuneStats {
    var stats: TuneStats = .{};
    const request = (try kernel.requested_providers(func)) orelse return stats;

    const function_fingerprint = try fingerprint.function(allocator, func);
    const selection_key = try kernel.make_selection_key(
        allocator,
        request,
        selected_device,
        function_fingerprint,
    );
    defer allocator.free(selection_key.bytes);

    if (store.get(selection_key) != null) {
        stats.dedup += 1;
        log.debug("dedup: function '{s}' selection already in store", .{func.name});
        return stats;
    }

    var candidates = std.ArrayList(AvailableCandidate).empty;
    defer candidates.deinit(allocator);
    defer for (candidates.items) |candidate| allocator.free(candidate.artifact.data);

    for (0..request.len()) |request_index| {
        const provider_name = request.at(request_index);
        const provider = find_provider(providers, provider_name) orelse {
            log.debug("no provider named '{s}' for function '{s}'", .{ provider_name, func.name });
            continue;
        };
        const artifact = provider.compile(func, selected_device, allocator) catch |err| switch (err) {
            error.Unsupported => {
                log.debug("provider '{s}' cannot handle function '{s}'", .{ provider_name, func.name });
                continue;
            },
            else => {
                log.err("provider '{s}' failed to compile function '{s}': {s}", .{ provider_name, func.name, @errorName(err) });
                return err;
            },
        };
        try candidates.append(allocator, .{
            .provider_name = provider_name,
            .artifact = artifact,
        });
    }

    if (candidates.items.len == 0) {
        stats.original += 1;
        try store.put(selection_key, .{
            .candidate = .original,
            .reason = "no requested provider produced an artifact",
        });
        return stats;
    }

    const evaluation = if (evaluator) |selected_evaluator|
        try selected_evaluator.evaluate(
            func,
            candidates.items,
            selected_device,
            allocator,
        )
    else
        Evaluation{
            .candidate = .original,
            .reason = "no candidate evaluator was configured",
        };

    switch (evaluation.candidate) {
        .original => {
            stats.original += 1;
            try store.put(selection_key, .{
                .candidate = .original,
                .reason = evaluation.reason,
                .timing = evaluation.timing,
            });
        },
        .provider => |candidate_index| {
            if (candidate_index >= candidates.items.len) return error.InvalidEvaluation;
            const selected = candidates.items[candidate_index];
            stats.selected += 1;
            candidates.items[candidate_index].artifact.data = &.{};
            try store.put(selection_key, .{
                .candidate = .{ .provider = .{
                    .provider_name = selected.provider_name,
                    .artifact = selected.artifact,
                } },
                .reason = evaluation.reason,
                .timing = evaluation.timing,
            });
            try register_provider_dispatch(
                dispatch_registry,
                find_provider(providers, selected.provider_name).?,
            );
        },
    }

    log.debug("evaluated {d} provider candidates for function '{s}': {s}", .{
        candidates.items.len,
        func.name,
        evaluation.reason,
    });

    return stats;
}

/// Register a provider's dispatch entry in the registry (idempotent).
///
/// Precondition: if `dispatch_fn` is set, `dispatch_ctx` must also be set.
fn register_provider_dispatch(dispatch_registry: *kernel.DispatchRegistry, provider: kernel.KernelProvider) !void {
    if (provider.dispatch_fn) |dfn| {
        try dispatch_registry.register(provider.name, .{
            .dispatch_fn = dfn,
            .prepare_fn = provider.prepare_fn,
            .dispatch_ctx = provider.dispatch_ctx orelse unreachable,
        });
    }
}

fn find_provider(providers: []const kernel.KernelProvider, name: []const u8) ?kernel.KernelProvider {
    for (providers) |p| {
        if (std.mem.eql(u8, p.name, name)) return p;
    }
    return null;
}

fn dump_store_summary(io: std.Io, store: *const kernel.KernelStore) void {
    var buf: [4096]u8 = undefined;
    var writer = std.Io.File.stdout().writer(io, &buf);
    const out = &writer.interface;

    out.writeAll("\n=== Tuning Summary ===\n") catch return;

    var it = store.selections.iterator();
    while (it.next()) |entry| {
        const selection = entry.value_ptr.*;
        switch (selection.candidate) {
            .provider => |stored| {
                out.print("  [+] {s}: provider={s}, bytes={d}, ws={d}", .{
                    entry.key_ptr.*,
                    stored.provider_name,
                    stored.artifact.data.len,
                    stored.artifact.workspace_bytes,
                }) catch return;
                dump_timing(out, selection.timing);
            },
            .original => {
                out.print("  [=] {s}: original, reason={s}", .{
                    entry.key_ptr.*,
                    selection.reason,
                }) catch return;
                dump_timing(out, selection.timing);
            },
        }
    }
    out.writeAll("\n") catch {};
    out.flush() catch {};
}

fn dump_timing(out: *std.Io.Writer, timing: ?kernel.TimingEvidence) void {
    if (timing) |measured| {
        out.print(", original={d}ns, selected={d}ns", .{
            measured.original_ns,
            measured.selected_ns,
        }) catch return;
    }
    out.writeByte('\n') catch {};
}

fn ns_to_ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / std.time.ns_per_ms;
}

// Tests.

const TestProvider = struct {
    name: []const u8,
    unsupported: bool,
    calls: usize = 0,

    fn interface(self: *TestProvider) kernel.KernelProvider {
        return .{
            .name = self.name,
            .ptr = @ptrCast(self),
            .compile_fn = compile,
        };
    }

    fn compile(
        ptr: *anyopaque,
        _: pr.Function,
        _: device.Device,
        allocator: std.mem.Allocator,
    ) kernel.CompileError!kernel.Artifact {
        const self: *TestProvider = @ptrCast(@alignCast(ptr));
        self.calls += 1;
        if (self.unsupported) return error.Unsupported;

        return .{
            .data = try allocator.dupe(u8, self.name),
        };
    }
};

const TestEvaluator = struct {
    select_provider: bool,
    calls: usize = 0,

    fn interface(self: *TestEvaluator) Evaluator {
        return .{
            .ptr = @ptrCast(self),
            .evaluate_fn = evaluate,
        };
    }

    fn evaluate(
        ptr: *anyopaque,
        _: pr.Function,
        candidates: []const AvailableCandidate,
        _: device.Device,
        _: std.mem.Allocator,
    ) EvaluationError!Evaluation {
        const self: *TestEvaluator = @ptrCast(@alignCast(ptr));
        self.calls += 1;
        return .{
            .candidate = if (self.select_provider) .{ .provider = candidates.len - 1 } else .original,
            .reason = if (self.select_provider)
                "test measurement selected provider"
            else
                "test measurement selected original",
        };
    }
};

test "MeasuredEvaluator selects the fastest correct implementation" {
    const testing = std.testing;
    const Measurements = struct {
        fn measure(
            _: *anyopaque,
            _: pr.Function,
            candidate: MeasurableCandidate,
            _: device.Device,
            _: std.mem.Allocator,
        ) EvaluationError!Measurement {
            return switch (candidate) {
                .original => .{ .elapsed_ns = 100, .correct = true },
                .provider => |value| if (std.mem.eql(u8, value.provider_name, "fast"))
                    .{ .elapsed_ns = 80, .correct = true }
                else
                    .{ .elapsed_ns = 40, .correct = false },
            };
        }
    };

    var measurement_context: u8 = 0;
    var evaluator = MeasuredEvaluator{
        .ptr = @ptrCast(&measurement_context),
        .measure_fn = Measurements.measure,
    };
    const candidates = [_]AvailableCandidate{
        .{ .provider_name = "fast", .artifact = .{ .data = &.{} } },
        .{ .provider_name = "incorrect", .artifact = .{ .data = &.{} } },
    };
    const selected = try evaluator.interface().evaluate(
        undefined,
        &candidates,
        .{ .platform = .cpu },
        testing.allocator,
    );
    switch (selected.candidate) {
        .original => return error.TestUnexpectedResult,
        .provider => |index| try testing.expectEqual(@as(usize, 0), index),
    }
    try testing.expectEqual(@as(u64, 100), selected.timing.?.original_ns);
    try testing.expectEqual(@as(u64, 80), selected.timing.?.selected_ns);
}

fn expect_provider_selections(first_unsupported: bool) !void {
    const testing = std.testing;
    const selected_device = device.Device{ .platform = .cpu };

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const first_input = try builder.param_tensor(.f32, &.{2});
    const second_input = try builder.param_tensor(.f32, &.{2});

    try builder.push_region("first_region", &.{kernel.provider_annotation("first")});
    const first_output = try builder.emit(.{ .exp = {} }, &.{first_input});
    try builder.pop_region();
    try builder.push_region("second_region", &.{kernel.provider_annotation("second")});
    const second_output = try builder.emit(.{ .exp = {} }, &.{second_input});
    try builder.pop_region();

    const function = try builder.finish(.{ .returns = &.{ first_output, second_output } });
    _ = try program.add_function(function);

    var outline_ctx = @import("compilation.zig").Context{
        .allocator = testing.allocator,
        .io = testing.io,
    };
    _ = try (@import("pr/transform/kernelize.zig").OutlineCandidates{}).run(&program, &outline_ctx);

    var first = TestProvider{ .name = "first", .unsupported = first_unsupported };
    var second = TestProvider{ .name = "second", .unsupported = !first_unsupported };
    var evaluator = TestEvaluator{ .select_provider = true };
    var result = try tune(testing.io, testing.allocator, &program, &.{
        first.interface(),
        second.interface(),
    }, .{
        .device = selected_device,
        .evaluator = evaluator.interface(),
    });
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), first.calls);
    try testing.expectEqual(@as(usize, 1), second.calls);
    try testing.expectEqual(@as(usize, 1), evaluator.calls);
    try testing.expectEqual(@as(usize, 2), result.store.selections.count());

    const first_fingerprint = try fingerprint.function(testing.allocator, program.functions()[1]);
    const second_fingerprint = try fingerprint.function(testing.allocator, program.functions()[2]);

    const first_key = try kernel.make_selection_key(
        testing.allocator,
        .{ .one = "first" },
        selected_device,
        first_fingerprint,
    );
    defer testing.allocator.free(first_key.bytes);
    const second_key = try kernel.make_selection_key(
        testing.allocator,
        .{ .one = "second" },
        selected_device,
        second_fingerprint,
    );
    defer testing.allocator.free(second_key.bytes);

    const first_selection = result.store.get(first_key) orelse return error.TestUnexpectedResult;
    const second_selection = result.store.get(second_key) orelse return error.TestUnexpectedResult;
    if (first_unsupported) {
        switch (first_selection.candidate) {
            .original => {},
            .provider => return error.TestUnexpectedResult,
        }
        switch (second_selection.candidate) {
            .provider => {},
            .original => return error.TestUnexpectedResult,
        }
    } else {
        switch (first_selection.candidate) {
            .provider => {},
            .original => return error.TestUnexpectedResult,
        }
        switch (second_selection.candidate) {
            .original => {},
            .provider => return error.TestUnexpectedResult,
        }
    }
}

test tune {
    const testing = std.testing;

    // A program without provider functions requires no tuning work.
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const func = try b.finish(.{ .returns = &.{x} });
    _ = try program.add_function(func);

    var result = try tune(std.testing.io, testing.allocator, &program, &.{}, .{
        .device = .{ .platform = .cpu },
    });
    defer result.deinit();

    try testing.expectEqual(@as(usize, 0), result.store.selections.count());
    try testing.expectEqual(@as(usize, 0), result.dispatch_registry.entries.count());
}

test "tune requires outlined provider requests" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const input = try builder.param_tensor(.f32, &.{2});
    try builder.push_region("candidate", &.{kernel.provider_annotation("test")});
    const output = try builder.exp(input);
    try builder.pop_region();
    _ = try program.add_function(try builder.finish(.{ .returns = &.{output} }));

    try testing.expectError(
        error.ProviderRegionNotOutlined,
        tune(testing.io, testing.allocator, &program, &.{}, .{
            .device = .{ .platform = .cpu },
        }),
    );
}

test "tune isolates provider selections for equal functions" {
    try expect_provider_selections(true);
    try expect_provider_selections(false);
}

test "tune evaluates all providers for one callable" {
    const testing = std.testing;
    const selected_device = device.Device{ .platform = .cpu };

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const input = try builder.param_tensor(.f32, &.{2});
    try builder.push_region("candidate", &.{kernel.providers_annotation(&.{ "first", "second" })});
    const output = try builder.exp(input);
    try builder.pop_region();
    _ = try program.add_function(try builder.finish(.{ .returns = &.{output} }));

    var outline_ctx = @import("compilation.zig").Context{
        .allocator = testing.allocator,
        .io = testing.io,
    };
    _ = try (@import("pr/transform/kernelize.zig").OutlineCandidates{}).run(&program, &outline_ctx);

    var first = TestProvider{ .name = "first", .unsupported = false };
    var second = TestProvider{ .name = "second", .unsupported = false };
    var evaluator = TestEvaluator{ .select_provider = true };
    var result = try tune(testing.io, testing.allocator, &program, &.{
        first.interface(),
        second.interface(),
    }, .{
        .device = selected_device,
        .evaluator = evaluator.interface(),
    });
    defer result.deinit();

    const function_fingerprint = try fingerprint.function(testing.allocator, program.functions()[1]);
    const key = try kernel.make_selection_key(
        testing.allocator,
        .{ .many = &.{ "first", "second" } },
        selected_device,
        function_fingerprint,
    );
    defer testing.allocator.free(key.bytes);
    const selection = result.store.get(key) orelse return error.TestUnexpectedResult;
    switch (selection.candidate) {
        .original => return error.TestUnexpectedResult,
        .provider => |selected| try testing.expectEqualStrings("second", selected.provider_name),
    }
    try testing.expectEqual(@as(usize, 1), first.calls);
    try testing.expectEqual(@as(usize, 1), second.calls);
    try testing.expectEqual(@as(usize, 1), evaluator.calls);
}

test "tune retains original without an evaluator" {
    const testing = std.testing;
    const selected_device = device.Device{ .platform = .cpu };

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const input = try builder.param_tensor(.f32, &.{2});
    try builder.push_region("candidate", &.{kernel.provider_annotation("provider")});
    const output = try builder.exp(input);
    try builder.pop_region();
    _ = try program.add_function(try builder.finish(.{ .returns = &.{output} }));
    var outline_ctx = @import("compilation.zig").Context{
        .allocator = testing.allocator,
        .io = testing.io,
    };
    _ = try (@import("pr/transform/kernelize.zig").OutlineCandidates{}).run(&program, &outline_ctx);

    var provider = TestProvider{ .name = "provider", .unsupported = false };
    var result = try tune(testing.io, testing.allocator, &program, &.{provider.interface()}, .{
        .device = selected_device,
    });
    defer result.deinit();

    const function_fingerprint = try fingerprint.function(testing.allocator, program.functions()[1]);
    const key = try kernel.make_selection_key(
        testing.allocator,
        .{ .one = "provider" },
        selected_device,
        function_fingerprint,
    );
    defer testing.allocator.free(key.bytes);
    const selection = result.store.get(key) orelse return error.TestUnexpectedResult;
    switch (selection.candidate) {
        .original => {},
        .provider => return error.TestUnexpectedResult,
    }
    try testing.expectEqual(@as(usize, 0), result.dispatch_registry.entries.count());
}
