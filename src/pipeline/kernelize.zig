/// Kernelization Pass
///
/// PR -> PR pass that consults a pre-computed `KernelStore` to replace annotated
/// regions with `custom_call` ops. Never invokes providers or performs compilation.
///
/// For each region with a `kernelize` annotation, the pass:
/// 1. Computes a kernel signature from the region's equation signature.
/// 2. Looks up the signature in the store for a tuning decision.
/// 3. Profitable decisions: rewrites the region's equations into a single
///    `custom_call` op carrying kernel_key, provider_name, and carrier metadata.
/// 4. Negative or absent decisions: leaves the region unchanged for baseline lowering.
///
/// The store is populated externally by `tune()` (src/tune.zig). This pass is a
/// pure consumer -- it never modifies the store.
const std = @import("std");
const pr = @import("../pr/pr.zig");
const kernel = @import("../kernel.zig");
const pass_mod = @import("pass.zig");

const log = std.log.scoped(.@"zg/kernelize");
/// Temporary single custom_call target for kernelized dispatch.
const dispatcher_target_name = "zigrad.kernel.dispatch";

const KernelCandidate = struct {
    region: pr.Region,
    provider_name: []const u8,

    // Populated after successful store lookup (used for PR rewriting).
    inputs: []const pr.VarId = &.{},
    outputs: []const pr.VarId = &.{},
    kernel_key: []const u8 = &.{},
    carrier_hint: ?[]const u8 = null,
};

const KernelEntryOutcome = enum { compiled, dedup, fallback };

/// Diagnostic record for one region encountered during kernelization.
///
/// `name` and `provider` are non-owning slices valid for the duration of
/// `run_impl`. `ops` and `shape` are owned by the entries arena.
const KernelEntry = struct {
    name: []const u8,
    provider: []const u8,
    ops: []const u8,
    shape: []const u8,
    outcome: KernelEntryOutcome,
};

/// Kernelization pass state.
///
/// Consults a pre-computed `KernelStore` to decide which annotated regions
/// to replace with `custom_call` ops. The store is the sole data source --
/// this pass never invokes providers or performs any compilation.
///
/// Create this struct, then call `pass()` to get a pipeline-compatible
/// `pass_mod.Pass` value.
pub const KernelizePass = struct {
    /// Pre-computed tuning decisions. Populated by `tune()` before the pipeline runs.
    /// The pass borrows this; caller retains ownership.
    store: *const kernel.KernelStore,
    /// When false, the pass performs store lookup and diagnostics but does not
    /// rewrite regions into custom_call ops. Useful for dry-run / audit mode.
    rewrite_regions: bool = true,
    /// Print a summary table of compiled/dedup/fallback regions after the pass.
    dump_kernels: bool = false,

    pub fn pass(self: *KernelizePass) pass_mod.Pass {
        return .{
            .ptr = @ptrCast(self),
            .run_fn = run_impl,
            .name = "kernelize",
            .input_kind = .pr,
            .output_kind = .pr,
        };
    }

    fn run_impl(ptr: *anyopaque, artifact: *pass_mod.Artifact, ctx: *pass_mod.PassContext) pass_mod.PassError!void {
        const self: *KernelizePass = @ptrCast(@alignCast(ptr));
        if (artifact.kind() != .pr) return error.ArtifactKindMismatch;

        // Arena backing all diagnostic entries; freed after the table is printed.
        var entries_arena = std.heap.ArenaAllocator.init(ctx.allocator);
        defer entries_arena.deinit();
        const entries_alloc = entries_arena.allocator();
        var entries = std.ArrayList(KernelEntry).empty;

        const program = artifact.pr;
        for (program.functions, 0..) |func, idx| {
            const rewritten = self.kernelize_function(
                program,
                func,
                ctx.allocator,
                if (self.dump_kernels) &entries else null,
                entries_alloc,
            ) catch |err| {
                if (!@import("builtin").is_test)
                    log.err("kernelization failed for function '{s}': {}", .{ func.name, err });
                return err;
            };
            program.functions[idx] = rewritten;
        }

        if (self.dump_kernels and entries.items.len > 0) {
            var buf: [8192]u8 = undefined;
            var stdout_writer = std.fs.File.stdout().writer(&buf);
            const out = &stdout_writer.interface;
            dump_kernel_entries(out, entries.items) catch {};
            out.flush() catch {};
        }
    }

    fn kernelize_function(
        self: *KernelizePass,
        program: *pr.Program,
        func: pr.Function,
        temp_allocator: std.mem.Allocator,
        entries: ?*std.ArrayList(KernelEntry),
        entries_alloc: std.mem.Allocator,
    ) !pr.Function {
        if (func.regions.len == 0) return func;
        if (!self.rewrite_regions) return func;

        const store = self.store;

        // Collect all candidates with kernelize annotations.
        var candidates = try std.ArrayList(KernelCandidate).initCapacity(temp_allocator, func.regions.len);
        defer candidates.deinit(temp_allocator);
        for (func.regions) |region| {
            const provider_name = region.annotation.kernelize orelse continue;
            try candidates.append(temp_allocator, .{
                .region = region,
                .provider_name = provider_name,
            });
        }

        var rewrites = try std.ArrayList(KernelCandidate).initCapacity(temp_allocator, func.regions.len);
        defer {
            for (rewrites.items) |rewrite| {
                temp_allocator.free(rewrite.inputs);
                temp_allocator.free(rewrite.outputs);
                if (rewrite.kernel_key.len > 0) temp_allocator.free(rewrite.kernel_key);
            }
            rewrites.deinit(temp_allocator);
        }

        for (candidates.items) |candidate| {
            if (is_region_nested(candidate.region, candidates.items)) continue;

            const desc = try kernel.describe_region(temp_allocator, func, candidate.region);
            defer temp_allocator.free(desc.inputs);
            defer temp_allocator.free(desc.outputs);

            if (desc.outputs.len == 0) continue;

            const kernel_signature = try compute_kernel_signature(temp_allocator, desc);
            defer temp_allocator.free(kernel_signature);

            const decision = store.get(kernel_signature) orelse {
                log.debug("store: no decision for region '{s}' (shape '{s}'), skipping", .{ candidate.region.name, kernel_signature });
                if (entries) |e| {
                    const ops = build_ops_str(entries_alloc, desc) catch "";
                    const shape = build_shape_str(entries_alloc, desc) catch "";
                    e.append(entries_alloc, .{
                        .name = candidate.region.name,
                        .provider = candidate.provider_name,
                        .ops = ops,
                        .shape = shape,
                        .outcome = .fallback,
                    }) catch {};
                }
                continue;
            };

            switch (decision) {
                .profitable => |art| {
                    const carrier_hint: ?[]const u8 = if (is_attention_5op_region(func, candidate.region)) "attention_5op_v1" else null;

                    try rewrites.append(temp_allocator, .{
                        .region = candidate.region,
                        .provider_name = art.provider_name,
                        .inputs = try temp_allocator.dupe(pr.VarId, desc.inputs),
                        .outputs = try temp_allocator.dupe(pr.VarId, desc.outputs),
                        .kernel_key = try temp_allocator.dupe(u8, kernel_signature),
                        .carrier_hint = carrier_hint,
                    });
                    log.debug("store: profitable decision for region '{s}' -> '{s}'", .{ candidate.region.name, art.target_name });

                    if (entries) |e| {
                        const ops = build_ops_str(entries_alloc, desc) catch "";
                        const shape = build_shape_str(entries_alloc, desc) catch "";
                        e.append(entries_alloc, .{
                            .name = candidate.region.name,
                            .provider = art.provider_name,
                            .ops = ops,
                            .shape = shape,
                            .outcome = .compiled,
                        }) catch {};
                    }
                },
                .negative => |reason| {
                    log.debug("store: negative decision for region '{s}': {s}", .{ candidate.region.name, reason });
                    if (entries) |e| {
                        const ops = build_ops_str(entries_alloc, desc) catch "";
                        const shape = build_shape_str(entries_alloc, desc) catch "";
                        e.append(entries_alloc, .{
                            .name = candidate.region.name,
                            .provider = candidate.provider_name,
                            .ops = ops,
                            .shape = shape,
                            .outcome = .fallback,
                        }) catch {};
                    }
                },
            }
        }

        if (rewrites.items.len == 0) return func;

        const allocator = program.allocator();

        var eqns = try std.ArrayList(pr.Eqn).initCapacity(allocator, func.eqns.len);
        var varids_store = try std.ArrayList(pr.VarId).initCapacity(allocator, func.varids_store.len);
        var params_store = try std.ArrayList(pr.Param).initCapacity(allocator, func.params_store.len + rewrites.items.len * 7);

        var eqn_index: usize = 0;
        while (eqn_index < func.eqns.len) {
            const rewrite = find_rewrite_starting_at(rewrites.items, eqn_index);
            if (rewrite) |entry| {
                try append_custom_call_eqn(allocator, &eqns, &varids_store, &params_store, func, entry);
                eqn_index += @as(usize, @intCast(entry.region.eqn_len));
                continue;
            }

            const eqn = func.eqns[eqn_index];
            try append_existing_eqn(allocator, &eqns, &varids_store, &params_store, func, eqn);
            eqn_index += 1;
        }

        return .{
            .name = func.name,
            .params = func.params,
            .returns = func.returns,
            .avals = func.avals,
            .eqns = try eqns.toOwnedSlice(allocator),
            .varids_store = try varids_store.toOwnedSlice(allocator),
            .params_store = try params_store.toOwnedSlice(allocator),
            .regions = &.{},
        };
    }

    fn find_rewrite_starting_at(rewrites: []const KernelCandidate, eqn_start: usize) ?KernelCandidate {
        for (rewrites) |entry| {
            if (entry.region.eqn_len == 0) continue;
            if (@as(usize, @intCast(entry.region.eqn_start)) == eqn_start) return entry;
        }
        return null;
    }

    fn is_region_nested(region: pr.Region, candidates: []const KernelCandidate) bool {
        if (region.eqn_len == 0) return false;
        const start: usize = @intCast(region.eqn_start);
        const end: usize = start + @as(usize, @intCast(region.eqn_len));
        for (candidates) |other| {
            if (other.region.eqn_len == 0) continue;
            const other_start: usize = @intCast(other.region.eqn_start);
            const other_end: usize = other_start + @as(usize, @intCast(other.region.eqn_len));
            const contains = (other_start <= start) and (other_end >= end);
            const strictly_larger = (other_start < start) or (other_end > end);
            if (contains and strictly_larger) return true;
        }
        return false;
    }

    fn append_existing_eqn(
        allocator: std.mem.Allocator,
        eqns: *std.ArrayList(pr.Eqn),
        varids_store: *std.ArrayList(pr.VarId),
        params_store: *std.ArrayList(pr.Param),
        func: pr.Function,
        eqn: pr.Eqn,
    ) !void {
        const inputs = eqn.inputs.slice(pr.VarId, func.varids_store);
        const outputs = eqn.outputs.slice(pr.VarId, func.varids_store);
        const params = eqn.params.slice(pr.Param, func.params_store);

        const in_span = try append_varids(allocator, varids_store, inputs);
        const out_span = try append_varids(allocator, varids_store, outputs);
        const param_span = try append_params(allocator, params_store, params);

        try eqns.append(allocator, .{
            .prim = eqn.prim,
            .inputs = in_span,
            .outputs = out_span,
            .params = param_span,
        });
    }

    fn append_custom_call_eqn(
        allocator: std.mem.Allocator,
        eqns: *std.ArrayList(pr.Eqn),
        varids_store: *std.ArrayList(pr.VarId),
        params_store: *std.ArrayList(pr.Param),
        func: pr.Function,
        rewrite: KernelCandidate,
    ) !void {
        if (rewrite.outputs.len == 0) return error.InvalidRegion;

        const out_avals = try allocator.alloc(pr.Aval, rewrite.outputs.len);
        errdefer allocator.free(out_avals);
        for (rewrite.outputs, 0..) |out_id, idx| {
            out_avals[idx] = func.avals[@intCast(out_id)];
        }

        const kernel_key = try allocator.dupe(u8, rewrite.kernel_key);
        errdefer allocator.free(kernel_key);
        const provider_name = try allocator.dupe(u8, rewrite.provider_name);
        errdefer allocator.free(provider_name);
        const target_name = try allocator.dupe(u8, dispatcher_target_name);
        errdefer allocator.free(target_name);

        const carrier_hint = if (rewrite.carrier_hint) |hint|
            try allocator.dupe(u8, hint)
        else
            null;
        errdefer if (carrier_hint) |owned_hint| allocator.free(owned_hint);

        var params: [7]pr.Param = undefined;
        var param_count: usize = 0;

        params[param_count] = .{ .call_target_name = target_name };
        param_count += 1;
        params[param_count] = .{ .call_kernel_key = kernel_key };
        param_count += 1;
        params[param_count] = .{ .call_provider_name = provider_name };
        param_count += 1;
        if (carrier_hint) |owned_hint| {
            params[param_count] = .{ .call_carrier_hint = owned_hint };
            param_count += 1;
        }
        params[param_count] = .{ .has_side_effect = false };
        param_count += 1;
        params[param_count] = .{ .out_avals = out_avals };
        param_count += 1;

        const in_span = try append_varids(allocator, varids_store, rewrite.inputs);
        const out_span = try append_varids(allocator, varids_store, rewrite.outputs);
        const param_span = try append_params(allocator, params_store, params[0..param_count]);

        try eqns.append(allocator, .{
            .prim = .custom_call,
            .inputs = in_span,
            .outputs = out_span,
            .params = param_span,
        });
    }

    fn is_attention_5op_region(func: pr.Function, region: pr.Region) bool {
        if (region.eqn_len != 5) return false;

        const start: usize = @intCast(region.eqn_start);
        const end = start + @as(usize, @intCast(region.eqn_len));
        if (end > func.eqns.len) return false;

        const eqns = func.eqns[start..end];
        const first = eqns[0].prim;
        if (first != .dot and first != .dot_general) return false;

        return eqns[1].prim == .add and
            eqns[2].prim == .exp and
            eqns[3].prim == .multiply and
            eqns[4].prim == .log;
    }

    fn append_varids(
        allocator: std.mem.Allocator,
        store: *std.ArrayList(pr.VarId),
        values: []const pr.VarId,
    ) !pr.Span {
        const start: u32 = @intCast(store.items.len);
        try store.appendSlice(allocator, values);
        return .{ .start = start, .len = @intCast(values.len) };
    }

    fn append_params(
        allocator: std.mem.Allocator,
        store: *std.ArrayList(pr.Param),
        values: []const pr.Param,
    ) !pr.Span {
        const start: u32 = @intCast(store.items.len);
        try store.appendSlice(allocator, values);
        return .{ .start = start, .len = @intCast(values.len) };
    }
};

// ============================================================================
// Kernel Signature
// ============================================================================

/// Build a deterministic kernel signature string for a region descriptor.
///
/// The key encodes the region's equation sequence as:
///   `<prim>,<in0_aval><in1_aval>...-><out0_aval>...;<next_eqn>...`
///
/// Two regions produce the same key iff they have identical op sequences with
/// matching input/output dtypes and dims. This is the deduplication criterion
/// used by `tune()` and the lookup key consulted by `KernelizePass`.
pub fn compute_kernel_signature(allocator: std.mem.Allocator, desc: kernel.RegionDescriptor) ![]const u8 {
    var buf = try std.ArrayList(u8).initCapacity(allocator, 128);
    errdefer buf.deinit(allocator);
    const w = buf.writer(allocator);

    for (desc.eqns, 0..) |eqn, ei| {
        if (ei > 0) try w.writeByte(';');
        try w.writeAll(@tagName(eqn.prim));
        try w.writeByte(',');
        const ins = eqn.inputs.slice(pr.VarId, desc.varids_store);
        for (ins, 0..) |vid, i| {
            if (i > 0) try w.writeByte(',');
            try write_aval_key(w, desc.aval_of(vid));
        }
        try w.writeByte('>');
        const outs = eqn.outputs.slice(pr.VarId, desc.varids_store);
        for (outs, 0..) |vid, i| {
            if (i > 0) try w.writeByte(',');
            try write_aval_key(w, desc.aval_of(vid));
        }
    }

    return buf.toOwnedSlice(allocator);
}

fn write_aval_key(w: anytype, aval: ?pr.Aval) !void {
    const a = aval orelse return w.writeByte('?');
    switch (a) {
        .tensor => |t| {
            try w.writeAll(@tagName(t.dtype));
            try w.writeByte('[');
            for (t.shape.dims, 0..) |d, i| {
                if (i > 0) try w.writeByte(',');
                try w.print("{d}", .{d});
            }
            try w.writeByte(']');
        },
    }
}

// ============================================================================
// Kernel Dump Helpers
// ============================================================================

fn build_ops_str(allocator: std.mem.Allocator, desc: kernel.RegionDescriptor) ![]const u8 {
    var buf = try std.ArrayList(u8).initCapacity(allocator, 64);
    errdefer buf.deinit(allocator);
    const w = buf.writer(allocator);
    for (desc.eqns, 0..) |eqn, i| {
        if (i > 0) try w.writeByte('+');
        try w.writeAll(@tagName(eqn.prim));
    }
    return buf.toOwnedSlice(allocator);
}

fn build_shape_str(allocator: std.mem.Allocator, desc: kernel.RegionDescriptor) ![]const u8 {
    var buf = try std.ArrayList(u8).initCapacity(allocator, 64);
    errdefer buf.deinit(allocator);
    const w = buf.writer(allocator);
    for (desc.inputs, 0..) |vid, i| {
        if (i > 0) try w.writeByte('x');
        try write_aval_key(w, desc.aval_of(vid));
    }
    return buf.toOwnedSlice(allocator);
}

fn dump_kernel_entries(out: *std.Io.Writer, entries: []const KernelEntry) !void {
    var compiled: usize = 0;
    var dedup: usize = 0;
    var fallback: usize = 0;
    for (entries) |e| switch (e.outcome) {
        .compiled => compiled += 1,
        .dedup => dedup += 1,
        .fallback => fallback += 1,
    };
    try out.print("kernels: {d} compiled, {d} dedup, {d} fallback\n", .{ compiled, dedup, fallback });
    try out.print("  {s:<50} {s:<10} {s:<30} {s:<50} {s}\n", .{ "region", "provider", "ops", "shapes", "outcome" });
    try out.writeAll("  " ++ ("-" ** 150) ++ "\n");
    for (entries) |e| {
        try out.print("  {s:<50} {s:<10} {s:<30} {s:<50} {s}\n", .{
            e.name, e.provider, e.ops, e.shape, @tagName(e.outcome),
        });
    }
}

// ============================================================================
// Tests
// ============================================================================

test "kernelize pass rewrites profitable region from store" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});

    try b.push_region("test_region", .{ .kernelize = "mock" });
    const y = try b.emit(.exp, &.{x}, &.{});
    try b.pop_region();

    const func = try b.finish(&.{y});
    try program.add_function(func);

    // Pre-populate the store with a profitable decision for the kernel signature.
    // The kernel signature for a single exp(f32[2]) region is "exp,f32[2]>f32[2]".
    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    try store.put_profitable("exp,f32[2]>f32[2]", .{
        .provider_name = "mock",
        .data = "stored_kernel_data",
        .target_name = "test_region",
    });

    var kp = KernelizePass{
        .store = &store,
    };

    var artifact = pass_mod.Artifact{ .pr = &program };
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator };
    try kp.pass().run(&artifact, &ctx);

    // Region should be rewritten to custom_call.
    try testing.expectEqual(@as(usize, 1), program.functions[0].eqns.len);
    const rewritten = program.functions[0].eqns[0];
    try testing.expectEqual(pr.Prim.custom_call, rewritten.prim);

    const params = rewritten.params.slice(pr.Param, program.functions[0].params_store);
    try testing.expectEqualStrings(dispatcher_target_name, pr.param_call_target_name(params).?);
    // kernel_key is the kernel signature used for store lookup at dispatch time.
    try testing.expectEqualStrings("exp,f32[2]>f32[2]", pr.param_call_kernel_key(params).?);
    try testing.expectEqualStrings("mock", pr.param_call_provider_name(params).?);
}

test "kernelize pass skips negative decision" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});

    try b.push_region("test_region", .{ .kernelize = "mock" });
    const y = try b.emit(.exp, &.{x}, &.{});
    try b.pop_region();

    const func = try b.finish(&.{y});
    try program.add_function(func);

    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    try store.put_negative("exp,f32[2]>f32[2]", "unsupported");

    var kp = KernelizePass{
        .store = &store,
    };

    var artifact = pass_mod.Artifact{ .pr = &program };
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator };
    try kp.pass().run(&artifact, &ctx);

    // Region should be left unchanged (negative decision).
    try testing.expectEqual(@as(usize, 1), program.functions[0].eqns.len);
    try testing.expectEqual(pr.Prim.exp, program.functions[0].eqns[0].prim);
}

test "kernelize pass skips absent key" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});

    try b.push_region("test_region", .{ .kernelize = "mock" });
    const y = try b.emit(.exp, &.{x}, &.{});
    try b.pop_region();

    const func = try b.finish(&.{y});
    try program.add_function(func);

    // Empty store -- no decisions at all.
    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();

    var kp = KernelizePass{
        .store = &store,
    };

    var artifact = pass_mod.Artifact{ .pr = &program };
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator };
    try kp.pass().run(&artifact, &ctx);

    // Region should be left unchanged (absent key).
    try testing.expectEqual(@as(usize, 1), program.functions[0].eqns.len);
    try testing.expectEqual(pr.Prim.exp, program.functions[0].eqns[0].prim);
}

test "kernelize pass tags attention-like 5-op region with carrier metadata" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    const lhs = try b.param_tensor(.f32, &.{ 2, 3 });
    const rhs = try b.param_tensor(.f32, &.{ 3, 2 });
    const bias = try b.param_tensor(.f32, &.{ 2, 2 });
    const scale = try b.param_tensor(.f32, &.{ 2, 2 });

    try b.push_region("attention_like", .{ .kernelize = "mock" });
    const dot = try b.emit(.dot, &.{ lhs, rhs }, &.{});
    const sum = try b.emit(.add, &.{ dot, bias }, &.{});
    const exp = try b.emit(.exp, &.{sum}, &.{});
    const mul = try b.emit(.multiply, &.{ exp, scale }, &.{});
    const out = try b.emit(.log, &.{mul}, &.{});
    try b.pop_region();

    const func = try b.finish(&.{out});
    try program.add_function(func);

    // Compute the kernel signature for this 5-op region and put a profitable decision.
    const desc = try kernel.describe_region(testing.allocator, func, func.regions[0]);
    defer testing.allocator.free(desc.inputs);
    defer testing.allocator.free(desc.outputs);
    const kernel_signature = try compute_kernel_signature(testing.allocator, desc);
    defer testing.allocator.free(kernel_signature);

    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    try store.put_profitable(kernel_signature, .{
        .provider_name = "mock",
        .data = "mock_kernel_data",
        .target_name = "attention_like",
    });

    var kp = KernelizePass{
        .store = &store,
    };

    var artifact = pass_mod.Artifact{ .pr = &program };
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator };
    try kp.pass().run(&artifact, &ctx);

    const rewritten = program.functions[0].eqns[0];
    try testing.expectEqual(pr.Prim.custom_call, rewritten.prim);

    const params = rewritten.params.slice(pr.Param, program.functions[0].params_store);
    try testing.expectEqualStrings("attention_5op_v1", pr.param_call_carrier_hint(params).?);
}

test "kernelize pass rewrites multi-output region to custom_call" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});
    const y = try b.param_tensor(.f32, &.{2});

    try b.push_region("multi_out", .{ .kernelize = "mock" });
    const a = try b.emit(.exp, &.{x}, &.{});
    const b_out = try b.emit(.log, &.{y}, &.{});
    try b.pop_region();

    const func = try b.finish(&.{ a, b_out });
    try program.add_function(func);

    // Compute kernel signature and store a profitable decision.
    const desc = try kernel.describe_region(testing.allocator, func, func.regions[0]);
    defer testing.allocator.free(desc.inputs);
    defer testing.allocator.free(desc.outputs);
    const kernel_signature = try compute_kernel_signature(testing.allocator, desc);
    defer testing.allocator.free(kernel_signature);

    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    try store.put_profitable(kernel_signature, .{
        .provider_name = "mock",
        .data = "mock_kernel_data",
        .target_name = "multi_out",
    });

    var kp = KernelizePass{
        .store = &store,
    };

    var artifact = pass_mod.Artifact{ .pr = &program };
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator };
    try kp.pass().run(&artifact, &ctx);

    try testing.expectEqual(@as(usize, 1), program.functions[0].eqns.len);
    const rewritten = program.functions[0].eqns[0];
    try testing.expectEqual(pr.Prim.custom_call, rewritten.prim);

    const outputs = rewritten.outputs.slice(pr.VarId, program.functions[0].varids_store);
    try testing.expectEqual(@as(usize, 2), outputs.len);

    const params = rewritten.params.slice(pr.Param, program.functions[0].params_store);
    const maybe_out_avals = pr.param_out_avals(params);
    try testing.expect(maybe_out_avals != null);
    const out_avals = maybe_out_avals.?;
    try testing.expectEqual(@as(usize, 2), out_avals.len);
}

test "kernelize pass same-shape regions share store decision" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    // Two exp regions with identical f32[2] input/output -- same kernel signature.
    const x = try b.param_tensor(.f32, &.{2});
    const y = try b.param_tensor(.f32, &.{2});

    try b.push_region("region_a", .{ .kernelize = "mock" });
    const out_a = try b.emit(.exp, &.{x}, &.{});
    try b.pop_region();

    try b.push_region("region_b", .{ .kernelize = "mock" });
    const out_b = try b.emit(.exp, &.{y}, &.{});
    try b.pop_region();

    const func = try b.finish(&.{ out_a, out_b });
    try program.add_function(func);

    // One store entry covers both regions since they share the same kernel signature.
    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    try store.put_profitable("exp,f32[2]>f32[2]", .{
        .provider_name = "mock",
        .data = "payload",
        .target_name = "region_a",
    });

    var kp = KernelizePass{
        .store = &store,
    };

    var artifact = pass_mod.Artifact{ .pr = &program };
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator };
    try kp.pass().run(&artifact, &ctx);

    // Both regions must have been rewritten to custom_call.
    const eqns = program.functions[0].eqns;
    try testing.expectEqual(@as(usize, 2), eqns.len);
    try testing.expectEqual(pr.Prim.custom_call, eqns[0].prim);
    try testing.expectEqual(pr.Prim.custom_call, eqns[1].prim);
}

test "kernelize pass different-shape regions need separate decisions" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    // Two exp regions with DIFFERENT shapes -- distinct kernel signatures.
    const x = try b.param_tensor(.f32, &.{2});
    const y = try b.param_tensor(.f32, &.{4});

    try b.push_region("region_small", .{ .kernelize = "mock" });
    const out_small = try b.emit(.exp, &.{x}, &.{});
    try b.pop_region();

    try b.push_region("region_large", .{ .kernelize = "mock" });
    const out_large = try b.emit(.exp, &.{y}, &.{});
    try b.pop_region();

    const func = try b.finish(&.{ out_small, out_large });
    try program.add_function(func);

    // Only put a decision for the small shape.
    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    try store.put_profitable("exp,f32[2]>f32[2]", .{
        .provider_name = "mock",
        .data = "payload",
        .target_name = "region_small",
    });

    var kp = KernelizePass{
        .store = &store,
    };

    var artifact = pass_mod.Artifact{ .pr = &program };
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator };
    try kp.pass().run(&artifact, &ctx);

    // Only region_small should be rewritten; region_large has no store decision.
    const eqns = program.functions[0].eqns;
    try testing.expectEqual(@as(usize, 2), eqns.len);
    try testing.expectEqual(pr.Prim.custom_call, eqns[0].prim);
    try testing.expectEqual(pr.Prim.exp, eqns[1].prim);
}

test "kernelize pass skips rewriting when rewrite_regions is false" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});
    try b.push_region("test_region", .{ .kernelize = "mock" });
    const y = try b.emit(.exp, &.{x}, &.{});
    try b.pop_region();

    const func = try b.finish(&.{y});
    try program.add_function(func);

    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    try store.put_profitable("exp,f32[2]>f32[2]", .{
        .provider_name = "mock",
        .data = "mock_kernel_data",
        .target_name = "test_region",
    });

    var kp = KernelizePass{
        .store = &store,
        .rewrite_regions = false,
    };

    const before_eqn_prim = program.functions[0].eqns[0].prim;
    var artifact = pass_mod.Artifact{ .pr = &program };
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator };
    try kp.pass().run(&artifact, &ctx);

    // Equations should be unchanged when rewrite_regions is false.
    try testing.expectEqual(before_eqn_prim, program.functions[0].eqns[0].prim);
}
