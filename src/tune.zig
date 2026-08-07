//! Kernel tuning for the KP system.
//!
//! Walks a PR program for kernelizable regions, invokes providers to compile
//!  each candidate, and records the results as decisions in a `KernelStore`.
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
const region_view = @import("pr/region_view.zig");
const pr = @import("pr/pr.zig");
const kernel = @import("pr/kernel.zig");

const log = std.log.scoped(.@"zg/tune");

/// Options for the tuning process.
pub const TuneOpts = struct {
    /// Print a summary table of tuning decisions after completion.
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

/// Candidate region identified during program scanning.
const TuneCandidate = struct {
    region: pr.Region,
    provider: kernel.KernelProvider,
    provider_name: []const u8,
    func_name: []const u8,
};

/// Tune a program: walk all functions for kernelizable regions, invoke
/// providers, and record decisions in the returned store.
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
    var store = kernel.KernelStore.init(allocator);
    errdefer store.deinit();

    var dispatch_registry = kernel.DispatchRegistry.init(allocator);
    errdefer dispatch_registry.deinit();

    const tune_start = std.Io.Timestamp.now(io, .awake);

    var total_compiled: usize = 0;
    var total_dedup: usize = 0;
    var total_negative: usize = 0;

    for (program.functions) |func| {
        const stats = try tune_function(
            allocator,
            func,
            providers,
            &store,
            &dispatch_registry,
            opts.device,
        );
        total_compiled += stats.compiled;
        total_dedup += stats.dedup;
        total_negative += stats.negative;
    }

    // Finalize all providers after tuning completes.
    for (providers) |provider| {
        provider.finalize();
    }

    log.info("tuning completed: {d} compiled, {d} dedup, {d} negative in {d:.2}ms", .{
        total_compiled, total_dedup, total_negative, ns_to_ms(@intCast(tune_start.untilNow(io, .awake).toNanoseconds())),
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
    compiled: usize = 0,
    dedup: usize = 0,
    negative: usize = 0,
};

fn tune_function(
    allocator: std.mem.Allocator,
    func: pr.Function,
    providers: []const kernel.KernelProvider,
    store: *kernel.KernelStore,
    dispatch_registry: *kernel.DispatchRegistry,
    selected_device: device.Device,
) !TuneStats {
    if (func.regions.len == 0) return .{};

    var stats: TuneStats = .{};

    // Collect candidates.
    var candidates = try std.ArrayList(TuneCandidate).initCapacity(allocator, func.regions.len);
    defer candidates.deinit(allocator);

    for (func.regions) |region| {
        const provider_name = region.annotation.kernelize orelse continue;
        const provider = find_provider(providers, provider_name) orelse {
            log.debug("no provider named '{s}' for region '{s}', skipping", .{ provider_name, region.name });
            continue;
        };
        try candidates.append(allocator, .{
            .region = region,
            .provider = provider,
            .provider_name = provider_name,
            .func_name = func.name,
        });
    }

    for (candidates.items) |candidate| {
        if (is_region_nested(candidate.region, candidates.items)) continue;

        const desc = try region_view.describe(allocator, func, candidate.region);
        defer desc.deinit(allocator);

        if (desc.outputs.len == 0) continue;

        const region_signature = try kernel.compute_region_signature(allocator, desc);
        defer allocator.free(region_signature.bytes);
        const decision_key = try kernel.make_decision_key(
            allocator,
            candidate.provider_name,
            selected_device,
            region_signature,
        );
        defer allocator.free(decision_key.bytes);

        // Already tuned (cross-function dedup via store)?
        if (store.get(decision_key) != null) {
            stats.dedup += 1;
            log.debug("dedup: provider '{s}' region '{s}' decision already in store", .{
                candidate.provider_name,
                candidate.region.name,
            });
            continue;
        }

        // Invoke provider.
        const compiled = candidate.provider.compile(desc, selected_device, allocator) catch |err| switch (err) {
            error.Unsupported => {
                stats.negative += 1;
                log.debug("provider '{s}' cannot handle region '{s}', recording negative", .{
                    candidate.provider_name, candidate.region.name,
                });
                try store.put_negative(decision_key, "unsupported");
                continue;
            },
            else => {
                log.err("provider '{s}' failed to compile region '{s}': {s}", .{
                    candidate.provider_name, candidate.region.name, @errorName(err),
                });
                return err;
            },
        };

        stats.compiled += 1;

        // Record profitable decision.
        try store.put_profitable(
            decision_key,
            candidate.provider_name,
            compiled,
            .take,
        );

        // Register dispatch from provider (idempotent).
        try register_provider_dispatch(dispatch_registry, candidate.provider);

        log.debug("compiled kernel for region '{s}' via provider '{s}'", .{
            candidate.region.name, candidate.provider_name,
        });
    }

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

fn is_region_nested(region: pr.Region, candidates: []const TuneCandidate) bool {
    for (candidates) |other| {
        if (other.region.id == region.id) continue;
        if (other.region.op_ids.len <= region.op_ids.len) continue;
        if (region_ids_subset(region.op_ids, other.region.op_ids)) return true;
    }
    return false;
}

fn region_ids_subset(needle: []const u32, haystack: []const u32) bool {
    for (needle) |id| {
        var found = false;
        for (haystack) |other_id| {
            if (other_id == id) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

fn dump_store_summary(io: std.Io, store: *const kernel.KernelStore) void {
    var buf: [4096]u8 = undefined;
    var writer = std.Io.File.stdout().writer(io, &buf);
    const out = &writer.interface;

    out.writeAll("\n=== Tuning Summary ===\n") catch return;

    var it = store.decisions.iterator();
    while (it.next()) |entry| {
        switch (entry.value_ptr.*) {
            .profitable => |stored| {
                out.print("  [+] {s}: provider={s}, bytes={d}, ws={d}\n", .{
                    entry.key_ptr.*,
                    stored.provider_name,
                    stored.artifact.data.len,
                    stored.artifact.workspace_bytes,
                }) catch return;
            },
            .negative => |reason| {
                out.print("  [-] {s}: {s}\n", .{ entry.key_ptr.*, reason }) catch return;
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
        _: region_view.RegionView,
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

fn expect_provider_decisions(first_unsupported: bool) !void {
    const testing = std.testing;
    const selected_device = device.Device{ .platform = .cpu };

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const first_input = try builder.param_tensor(.f32, &.{2});
    const second_input = try builder.param_tensor(.f32, &.{2});

    try builder.push_region("first_region", .{ .kernelize = "first" });
    const first_output = try builder.emit(.{ .exp = {} }, &.{first_input});
    try builder.pop_region();
    try builder.push_region("second_region", .{ .kernelize = "second" });
    const second_output = try builder.emit(.{ .exp = {} }, &.{second_input});
    try builder.pop_region();

    const function = try builder.finish(&.{ first_output, second_output });
    try program.add_function(function);

    var first = TestProvider{ .name = "first", .unsupported = first_unsupported };
    var second = TestProvider{ .name = "second", .unsupported = !first_unsupported };
    var result = try tune(testing.io, testing.allocator, &program, &.{
        first.interface(),
        second.interface(),
    }, .{ .device = selected_device });
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), first.calls);
    try testing.expectEqual(@as(usize, 1), second.calls);
    try testing.expectEqual(@as(usize, 2), result.store.decisions.count());

    const desc = try region_view.describe(testing.allocator, function, function.regions[0]);
    defer desc.deinit(testing.allocator);
    const signature = try kernel.compute_region_signature(testing.allocator, desc);
    defer testing.allocator.free(signature.bytes);

    const first_key = try kernel.make_decision_key(
        testing.allocator,
        "first",
        selected_device,
        signature,
    );
    defer testing.allocator.free(first_key.bytes);
    const second_key = try kernel.make_decision_key(
        testing.allocator,
        "second",
        selected_device,
        signature,
    );
    defer testing.allocator.free(second_key.bytes);

    const first_decision = result.store.get(first_key) orelse return error.TestUnexpectedResult;
    const second_decision = result.store.get(second_key) orelse return error.TestUnexpectedResult;
    if (first_unsupported) {
        switch (first_decision) {
            .negative => {},
            .profitable => return error.TestUnexpectedResult,
        }
        switch (second_decision) {
            .profitable => {},
            .negative => return error.TestUnexpectedResult,
        }
    } else {
        switch (first_decision) {
            .profitable => {},
            .negative => return error.TestUnexpectedResult,
        }
        switch (second_decision) {
            .negative => {},
            .profitable => return error.TestUnexpectedResult,
        }
    }
}

test tune {
    const testing = std.testing;

    // A program without regions requires no tuning work.
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

    // No regions -> no decisions.
    try testing.expectEqual(@as(usize, 0), result.store.decisions.count());
    try testing.expectEqual(@as(usize, 0), result.dispatch_registry.entries.count());
}

test "tune isolates provider decisions for identical regions" {
    try expect_provider_decisions(true);
    try expect_provider_decisions(false);
}
