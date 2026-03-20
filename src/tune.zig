/// Standalone Tuning Tool
///
/// Walks a PR program for kernelizable regions, invokes providers to compile
/// each candidate, and records the results as tuning decisions in a `KernelStore`.
/// Provider dispatch entries are registered in a `DispatchRegistry` for
/// execute-time resolution.
///
/// This decouples tuning (an expensive, user-controlled operation) from the
/// pipeline (which consults a pre-computed store, never calls providers).
///
/// Usage:
///   var result = try tune(allocator, &program, providers);
///   defer result.deinit();
///   // result.store -> pass to pipeline via CompileConfig.kernel_store
///   // result.dispatch_registry -> pass to ExecuteOptions.dispatch_registry
const std = @import("std");
const pr = @import("pr/pr.zig");
const kernel = @import("kernel.zig");
const kernelize = @import("pipeline/kernelize.zig");

const log = std.log.scoped(.@"zg/tune");

/// Options for the tuning process.
pub const TuneOpts = struct {
    /// Print a summary table of tuning decisions after completion.
    dump_results: bool = false,
    /// Device targeting context passed to providers during compilation.
    compile_ctx: kernel.CompileContext = .{},
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
/// On success, callers should pass `result.store` to the pipeline via
/// `CompileConfig.kernel_store` and `result.dispatch_registry` to
/// `ExecuteOptions.dispatch_registry`. Providers are finalized before return.
pub fn tune(
    allocator: std.mem.Allocator,
    program: *const pr.Program,
    providers: []const kernel.KernelProvider,
    opts: TuneOpts,
) !TuneResult {
    var store = kernel.KernelStore.init(allocator);
    errdefer store.deinit();

    var dispatch_registry = kernel.DispatchRegistry.init(allocator);
    errdefer dispatch_registry.deinit();

    // Shape cache spans all functions for cross-function deduplication.
    var shape_cache = std.StringHashMap(ShapeCacheEntry).init(allocator);
    defer {
        var it = shape_cache.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.data);
        }
        shape_cache.deinit();
    }

    var timer = std.time.Timer.start() catch null;

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
            &shape_cache,
            opts.compile_ctx,
        );
        total_compiled += stats.compiled;
        total_dedup += stats.dedup;
        total_negative += stats.negative;
    }

    // Finalize all providers after tuning completes.
    for (providers) |provider| {
        provider.finalize();
    }

    if (timer) |*t| {
        log.info("tuning completed: {d} compiled, {d} dedup, {d} negative in {d:.2}ms", .{
            total_compiled, total_dedup, total_negative, ns_to_ms(t.read()),
        });
    }

    if (opts.dump_results) {
        dump_store_summary(&store);
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

/// Cached artifact template keyed by region shape signature.
const ShapeCacheEntry = struct {
    data: []const u8,
    workspace_bytes: usize,
    provider_name: []const u8,
};

fn tune_function(
    allocator: std.mem.Allocator,
    func: pr.Function,
    providers: []const kernel.KernelProvider,
    store: *kernel.KernelStore,
    dispatch_registry: *kernel.DispatchRegistry,
    shape_cache: *std.StringHashMap(ShapeCacheEntry),
    compile_ctx: kernel.CompileContext,
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

        const desc = try kernel.describe_region(allocator, func, candidate.region);
        defer allocator.free(desc.inputs);
        defer allocator.free(desc.outputs);

        if (desc.outputs.len == 0) continue;

        const kernel_signature = try kernelize.compute_kernel_signature(allocator, desc);
        defer allocator.free(kernel_signature);

        // Already tuned (across functions)?
        if (store.get(kernel_signature) != null) {
            stats.dedup += 1;
            log.debug("dedup: region '{s}' shape already in store", .{candidate.region.name});
            continue;
        }

        // Check shape cache (same shape compiled earlier this session).
        if (shape_cache.get(kernel_signature)) |cached| {
            stats.dedup += 1;
            log.debug("dedup cache hit: region '{s}' reuses compiled artifact", .{candidate.region.name});

            try store.put_profitable(kernel_signature, .{
                .provider_name = cached.provider_name,
                .data = cached.data,
                .target_name = candidate.region.name,
                .workspace_bytes = cached.workspace_bytes,
            });

            // Register dispatch from provider (idempotent).
            try register_provider_dispatch(dispatch_registry, candidate.provider);
            continue;
        }

        // Invoke provider.
        var compiled = candidate.provider.compile(desc, compile_ctx, allocator) catch |err| switch (err) {
            error.Unsupported => {
                stats.negative += 1;
                log.debug("provider '{s}' cannot handle region '{s}', recording negative", .{
                    candidate.provider_name, candidate.region.name,
                });
                try store.put_negative(kernel_signature, "unsupported");
                continue;
            },
            else => {
                log.err("provider '{s}' failed to compile region '{s}': {s}", .{
                    candidate.provider_name, candidate.region.name, @errorName(err),
                });
                return err;
            },
        };
        defer compiled.deinit(allocator);

        stats.compiled += 1;

        // Populate shape cache.
        const cache_key = try allocator.dupe(u8, kernel_signature);
        errdefer allocator.free(cache_key);
        const cache_data = try allocator.dupe(u8, compiled.data);
        errdefer allocator.free(cache_data);
        try shape_cache.put(cache_key, .{
            .data = cache_data,
            .workspace_bytes = compiled.workspace_bytes,
            .provider_name = compiled.provider_name,
        });

        // Record profitable decision.
        try store.put_profitable(kernel_signature, .{
            .provider_name = compiled.provider_name,
            .data = compiled.data,
            .target_name = candidate.region.name,
            .workspace_bytes = compiled.workspace_bytes,
        });

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
    if (region.eqn_len == 0) return false;
    const start: usize = @intCast(region.eqn_start);
    const end = start + @as(usize, @intCast(region.eqn_len));
    for (candidates) |other| {
        if (other.region.eqn_len == 0) continue;
        const ostart: usize = @intCast(other.region.eqn_start);
        const oend = ostart + @as(usize, @intCast(other.region.eqn_len));
        if (ostart <= start and oend >= end and (ostart != start or oend != end)) return true;
    }
    return false;
}

fn dump_store_summary(store: *const kernel.KernelStore) void {
    var buf: [4096]u8 = undefined;
    var writer = std.fs.File.stdout().writer(&buf);
    const out = &writer.interface;

    out.writeAll("\n=== Tuning Summary ===\n") catch return;

    var it = store.decisions.iterator();
    while (it.next()) |entry| {
        switch (entry.value_ptr.*) {
            .profitable => |art| {
                out.print("  [+] {s} -> {s} ({d} bytes, ws={d})\n", .{
                    entry.key_ptr.*, art.target_name, art.data.len, art.workspace_bytes,
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

// ============================================================================
// Tests
// ============================================================================

test tune {
    const testing = std.testing;

    // Create a simple program with no regions -- tune should be a no-op.
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const func = try b.finish(&.{x});
    try program.add_function(func);

    var result = try tune(testing.allocator, &program, &.{}, .{});
    defer result.deinit();

    // No regions -> no decisions.
    try testing.expectEqual(@as(usize, 0), result.store.decisions.count());
    try testing.expectEqual(@as(usize, 0), result.dispatch_registry.entries.count());
}
