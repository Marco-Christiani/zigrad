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
    try kernel.require_outlined_requests(program);

    var store = kernel.KernelStore.init(allocator);
    errdefer store.deinit();

    var dispatch_registry = kernel.DispatchRegistry.init(allocator);
    errdefer dispatch_registry.deinit();

    const tune_start = std.Io.Timestamp.now(io, .awake);

    var total_selected: usize = 0;
    var total_dedup: usize = 0;
    var total_original: usize = 0;

    for (program.functions) |func| {
        const stats = try tune_function(
            allocator,
            func,
            providers,
            &store,
            &dispatch_registry,
            opts.device,
        );
        total_selected += stats.selected;
        total_dedup += stats.dedup;
        total_original += stats.original;
    }

    // Finalize all providers after tuning completes.
    for (providers) |provider| {
        provider.finalize();
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
) !TuneStats {
    var stats: TuneStats = .{};
    const provider_name = (try kernel.requested_provider(func)) orelse return stats;
    const provider = find_provider(providers, provider_name) orelse {
        log.debug("no provider named '{s}' for function '{s}', skipping", .{ provider_name, func.name });
        return stats;
    };

    const function_fingerprint = try fingerprint.function(allocator, func);
    const selection_key = try kernel.make_selection_key(
        allocator,
        provider_name,
        selected_device,
        function_fingerprint,
    );
    defer allocator.free(selection_key.bytes);

    if (store.get(selection_key) != null) {
        stats.dedup += 1;
        log.debug("dedup: provider '{s}' function '{s}' selection already in store", .{ provider_name, func.name });
        return stats;
    }

    const compiled = provider.compile(func, selected_device, allocator) catch |err| switch (err) {
        error.Unsupported => {
            stats.original += 1;
            log.debug("provider '{s}' cannot handle function '{s}', selecting the original candidate", .{ provider_name, func.name });
            try store.put(selection_key, .{
                .candidate = .original,
                .reason = "provider does not support the function",
            });
            return stats;
        },
        else => {
            log.err("provider '{s}' failed to compile function '{s}': {s}", .{ provider_name, func.name, @errorName(err) });
            return err;
        },
    };

    stats.selected += 1;
    try store.put(selection_key, .{
        .candidate = .{ .provider = .{
            .provider_name = provider_name,
            .artifact = compiled,
        } },
        .reason = "provider artifact is available",
    });
    try register_provider_dispatch(dispatch_registry, provider);

    log.debug("compiled kernel for function '{s}' via provider '{s}'", .{ func.name, provider_name });

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
                out.print("  [+] {s}: provider={s}, bytes={d}, ws={d}\n", .{
                    entry.key_ptr.*,
                    stored.provider_name,
                    stored.artifact.data.len,
                    stored.artifact.workspace_bytes,
                }) catch return;
            },
            .original => {
                out.print("  [=] {s}: original, reason={s}\n", .{
                    entry.key_ptr.*,
                    selection.reason,
                }) catch return;
            },
        }
    }
    out.writeAll("\n") catch {};
    out.flush() catch {};
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

    const function = try builder.finish(&.{ first_output, second_output });
    try program.add_function(function);

    var outline_ctx = @import("compilation.zig").Context{
        .allocator = testing.allocator,
        .io = testing.io,
    };
    _ = try (@import("pr/transform/kernelize.zig").OutlineCandidates{}).run(&program, &outline_ctx);

    var first = TestProvider{ .name = "first", .unsupported = first_unsupported };
    var second = TestProvider{ .name = "second", .unsupported = !first_unsupported };
    var result = try tune(testing.io, testing.allocator, &program, &.{
        first.interface(),
        second.interface(),
    }, .{ .device = selected_device });
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), first.calls);
    try testing.expectEqual(@as(usize, 1), second.calls);
    try testing.expectEqual(@as(usize, 2), result.store.selections.count());

    const first_fingerprint = try fingerprint.function(testing.allocator, program.functions[1]);
    const second_fingerprint = try fingerprint.function(testing.allocator, program.functions[2]);

    const first_key = try kernel.make_selection_key(
        testing.allocator,
        "first",
        selected_device,
        first_fingerprint,
    );
    defer testing.allocator.free(first_key.bytes);
    const second_key = try kernel.make_selection_key(
        testing.allocator,
        "second",
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
    const func = try b.finish(&.{x});
    try program.add_function(func);

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
    try program.add_function(try builder.finish(&.{output}));

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
