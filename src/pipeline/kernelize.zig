/// Kernelization Pass
///
/// PR → PR pass that replaces annotated regions with custom_call ops,
/// compiling kernel artifacts via registered providers.
///
/// For each region with a `kernelize` annotation, the pass:
/// 1. Builds a RegionDescriptor from the function and region
/// 2. Finds the named provider
/// 3. Calls provider.compile() to produce a KernelArtifact
/// 4. Registers the KA in the registry under a target name
/// 5. Replaces the region's equations with a custom_call op
///
/// Current temporary execution strategy uses one dispatcher target
/// (`zigrad.kernel.dispatch`) for all kernelized regions. Per-region
/// dispatch identity is carried via custom_call params and lowering emits
/// typed-FFI backend_config attributes:
/// - `zigrad.kernel_key`
/// - `zigrad.provider`
///
/// If a provider cannot handle a region (returns Unsupported), the
/// region's equations are left unchanged — baseline lowering handles them.
const std = @import("std");
const pr = @import("../pr/pr.zig");
const kernel = @import("../kernel.zig");
const pass_mod = @import("pass.zig");

const log = std.log.scoped(.@"zg/kernelize");
/// Temporary single custom_call target for kernelized dispatch.
const dispatcher_target_name = "zigrad.kernel.dispatch";

const KernelCandidate = struct {
    region: pr.Region,
    provider: kernel.KernelProvider,
    provider_name: []const u8,

    // Populated after successful compilation (used for PR rewriting).
    inputs: []const pr.VarId = &.{},
    outputs: []const pr.VarId = &.{},
    kernel_key: []const u8 = &.{},
    kernel_id: ?u32 = null,
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

/// Cached artifact template keyed by region shape signature.
///
/// Stores enough information to clone a `KernelArtifact` for a new region
/// that has the same structural shape as a previously compiled one, avoiding
/// a redundant superoptimization + compile cycle.
const ShapeCacheEntry = struct {
    /// Copy of the compiled artifact bytes; owned by the cache.
    data: []const u8,
    workspace_bytes: usize,
    dispatch_fn: ?kernel.DispatchFn,
    /// Non-owning; points into the provider's own name storage.
    dispatch_ctx: ?*anyopaque,
    provider_name: []const u8,
};

pub const TargetNameMode = enum {
    region_name,
    outlined_eqn,
};

/// Kernelization pass state. Holds the registry and providers.
///
/// Create this struct, then call `pass()` to get a pipeline-compatible
/// `pass_mod.Pass` value.
pub const KernelizePass = struct {
    registry: *kernel.KernelRegistry,
    package: ?*kernel.KernelPackage = null,
    providers: []const kernel.KernelProvider,
    rewrite_regions: bool = true,
    target_name_mode: TargetNameMode = .region_name,
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

        // Shape cache spans all functions so that identical regions across
        // function boundaries also share a single compiled artifact.
        var shape_cache = std.StringHashMap(ShapeCacheEntry).init(ctx.allocator);
        defer {
            var it = shape_cache.iterator();
            while (it.next()) |entry| {
                ctx.allocator.free(entry.key_ptr.*);
                ctx.allocator.free(entry.value_ptr.data);
            }
            shape_cache.deinit();
        }

        // Arena backing all diagnostic entries; freed after the table is printed.
        var entries_arena = std.heap.ArenaAllocator.init(ctx.allocator);
        defer entries_arena.deinit();
        const entries_alloc = entries_arena.allocator();
        var entries = std.ArrayList(KernelEntry).empty;

        const program = artifact.pr;
        const functions: []pr.Function = @constCast(program.functions);
        for (program.functions, 0..) |func, idx| {
            const rewritten = self.kernelize_function(
                program,
                func,
                ctx.allocator,
                &shape_cache,
                if (self.dump_kernels) &entries else null,
                entries_alloc,
            ) catch |err| {
                log.err("kernelization failed for function '{s}': {}", .{ func.name, err });
                return err;
            };
            functions[idx] = rewritten;
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
        shape_cache: *std.StringHashMap(ShapeCacheEntry),
        entries: ?*std.ArrayList(KernelEntry),
        entries_alloc: std.mem.Allocator,
    ) !pr.Function {
        if (func.regions.len == 0) return func;

        var candidates = try std.ArrayList(KernelCandidate).initCapacity(temp_allocator, func.regions.len);
        defer candidates.deinit(temp_allocator);

        for (func.regions) |region| {
            const provider_name = region.annotation.kernelize orelse continue;
            const provider = self.find_provider(provider_name) orelse {
                log.debug("no provider named '{s}' for region '{s}', skipping", .{ provider_name, region.name });
                continue;
            };
            try candidates.append(temp_allocator, .{
                .region = region,
                .provider = provider,
                .provider_name = provider_name,
            });
        }

        var rewrites = try std.ArrayList(KernelCandidate).initCapacity(temp_allocator, func.regions.len);
        defer {
            for (rewrites.items) |rewrite| {
                temp_allocator.free(rewrite.inputs);
                temp_allocator.free(rewrite.outputs);
            }
            rewrites.deinit(temp_allocator);
        }

        for (candidates.items) |candidate| {
            if (is_region_nested(candidate.region, candidates.items)) {
                log.debug("region '{s}' is nested inside a larger kernelized region, skipping", .{candidate.region.name});
                continue;
            }

            const desc = try kernel.describe_region(temp_allocator, func, candidate.region);
            defer temp_allocator.free(desc.inputs);
            defer temp_allocator.free(desc.outputs);

            if (desc.outputs.len == 0) {
                log.debug("region '{s}' has no externally used outputs; skipping kernelization", .{candidate.region.name});
                continue;
            }

            // Check the shape cache before invoking the provider. Regions with
            // identical op sequences and tensor shapes (e.g. the same projection
            // in different LLaMA layers) share one compiled artifact, avoiding
            // redundant superoptimization passes and reducing unique .so files.
            const shape_key = try compute_shape_key(temp_allocator, desc);
            defer temp_allocator.free(shape_key);

            var was_dedup = false;
            var ka: kernel.KernelArtifact = if (shape_cache.get(shape_key)) |cached| hit: {
                was_dedup = true;
                log.debug("dedup cache hit: region '{s}' reuses compiled artifact", .{candidate.region.name});
                const reg_alloc = self.registry.allocator();
                const target_name = try reg_alloc.dupe(u8, candidate.region.name);
                errdefer reg_alloc.free(target_name);
                break :hit .{
                    .provider_name = cached.provider_name,
                    .data = try reg_alloc.dupe(u8, cached.data),
                    .target_name = target_name,
                    .workspace_bytes = cached.workspace_bytes,
                    .dispatch_fn = cached.dispatch_fn,
                    .dispatch_ctx = cached.dispatch_ctx,
                };
            } else miss: {
                const compiled = candidate.provider.compile(desc, self.registry.allocator()) catch |err| switch (err) {
                    error.Unsupported => {
                        log.debug("provider '{s}' cannot handle region '{s}', falling back to baseline", .{ candidate.provider_name, candidate.region.name });
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
                    },
                    else => {
                        log.err("provider '{s}' failed to compile region '{s}': {s}", .{ candidate.provider_name, candidate.region.name, @errorName(err) });
                        return err;
                    },
                };
                // Populate cache so subsequent same-shape regions skip compilation.
                const cache_key = try temp_allocator.dupe(u8, shape_key);
                errdefer temp_allocator.free(cache_key);
                const cache_data = try temp_allocator.dupe(u8, compiled.data);
                errdefer temp_allocator.free(cache_data);
                try shape_cache.put(cache_key, .{
                    .data = cache_data,
                    .workspace_bytes = compiled.workspace_bytes,
                    .dispatch_fn = compiled.dispatch_fn,
                    .dispatch_ctx = compiled.dispatch_ctx,
                    .provider_name = compiled.provider_name,
                });
                break :miss compiled;
            };

            try self.retarget_kernel_artifact(func, candidate.region, &ka);

            if (entries) |e| {
                const outcome: KernelEntryOutcome = if (was_dedup) .dedup else .compiled;
                const ops = build_ops_str(entries_alloc, desc) catch "";
                const shape = build_shape_str(entries_alloc, desc) catch "";
                e.append(entries_alloc, .{
                    .name = candidate.region.name,
                    .provider = candidate.provider_name,
                    .ops = ops,
                    .shape = shape,
                    .outcome = outcome,
                }) catch {};
            }

            self.registry.put(ka.target_name, ka) catch |err| switch (err) {
                error.DuplicateKey => {
                    var artifact = ka;
                    artifact.deinit(self.registry.allocator());
                    log.err("duplicate kernel key '{s}' for region '{s}'", .{ ka.target_name, candidate.region.name });
                    return error.DuplicateKey;
                },
                error.OutOfMemory => return error.OutOfMemory,
            };
            log.debug("compiled kernel '{s}' for region '{s}' via provider '{s}'", .{ ka.target_name, candidate.region.name, candidate.provider_name });

            const carrier_hint: ?[]const u8 = if (is_attention_5op_region(func, candidate.region)) "attention_5op_v1" else null;
            const kernel_id = kernel.kernel_id_from_key(ka.target_name);

            if (self.package) |pkg| {
                const pkg_artifact = try clone_artifact_for_package(pkg.allocator(), ka);
                pkg.put(kernel_id, pkg_artifact) catch |err| switch (err) {
                    error.DuplicateKey => {
                        var artifact = pkg_artifact;
                        artifact.deinit(pkg.allocator());
                        log.err("duplicate kernel id {d} for region '{s}'", .{ kernel_id, candidate.region.name });
                        return error.DuplicateKey;
                    },
                    error.OutOfMemory => return error.OutOfMemory,
                };
            }

            if (!self.rewrite_regions) {
                continue;
            }

            try rewrites.append(temp_allocator, .{
                .region = candidate.region,
                .provider = candidate.provider,
                .provider_name = ka.provider_name,
                .inputs = try temp_allocator.dupe(pr.VarId, desc.inputs),
                .outputs = try temp_allocator.dupe(pr.VarId, desc.outputs),
                .kernel_key = ka.target_name,
                .kernel_id = kernel_id,
                .carrier_hint = carrier_hint,
            });
        }

        if (!self.rewrite_regions or rewrites.items.len == 0) return func;

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

    fn find_provider(self: *KernelizePass, name: []const u8) ?kernel.KernelProvider {
        for (self.providers) |p| {
            if (std.mem.eql(u8, p.name, name)) return p;
        }
        return null;
    }

    fn retarget_kernel_artifact(self: *KernelizePass, func: pr.Function, region: pr.Region, artifact: *kernel.KernelArtifact) !void {
        switch (self.target_name_mode) {
            .region_name => {},
            .outlined_eqn => {
                if (region.eqn_len != 1) return error.InvalidRegion;

                const allocator = self.registry.allocator();
                const renamed = try std.fmt.allocPrint(allocator, "{s}_outlined_{d}", .{ func.name, region.eqn_start });
                allocator.free(artifact.target_name);
                artifact.target_name = renamed;
            },
        }
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
        if (rewrite.kernel_id) |kernel_id| {
            params[param_count] = .{ .call_kernel_id = kernel_id };
            param_count += 1;
        }
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

    fn clone_artifact_for_package(
        dst_allocator: std.mem.Allocator,
        artifact: kernel.KernelArtifact,
    ) !kernel.KernelArtifact {
        return .{
            .provider_name = artifact.provider_name,
            .data = try dst_allocator.dupe(u8, artifact.data),
            .target_name = artifact.target_name,
            .workspace_bytes = artifact.workspace_bytes,
            .dispatch_fn = artifact.dispatch_fn,
            .dispatch_ctx = artifact.dispatch_ctx,
        };
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
// Shape Key
// ============================================================================

/// Build a deterministic shape-signature string for a region descriptor.
///
/// The key encodes the region's equation sequence as:
///   `<prim>,<in0_aval><in1_aval>...-><out0_aval>...;<next_eqn>...`
///
/// Two regions produce the same key iff they have identical op sequences with
/// matching input/output dtypes and dims. This is the criterion for sharing a
/// compiled artifact via the shape cache in `run_impl`.
fn compute_shape_key(allocator: std.mem.Allocator, desc: kernel.RegionDescriptor) ![]const u8 {
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

test "kernelize pass skips regions with no matching provider" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});

    try b.push_region("r", .{ .kernelize = "nonexistent" });
    const y = try b.emit(.exp, &.{x}, &.{});
    try b.pop_region();

    const func = try b.finish(&.{y});
    try program.add_function(func);

    var registry = kernel.KernelRegistry.init(testing.allocator);
    defer registry.deinit();

    var kp = KernelizePass{
        .registry = &registry,
        .providers = &.{}, // no providers registered
    };

    var artifact = pass_mod.Artifact{ .pr = &program };
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator };
    try kp.pass().run(&artifact, &ctx);

    // Region equations should still be there (fallback).
    try testing.expectEqual(@as(usize, 1), program.functions[0].eqns.len);
    // Registry should be empty (nothing compiled).
    try testing.expect(registry.get("anything") == null);
}

test "kernelize pass calls provider and registers KA" {
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

    const MockProvider = struct {
        compiled: bool = false,

        fn compile(ptr: *anyopaque, desc: kernel.RegionDescriptor, allocator: std.mem.Allocator) kernel.CompileError!kernel.KernelArtifact {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.compiled = true;
            return .{
                .provider_name = "mock",
                .data = try allocator.dupe(u8, "mock_kernel_data"),
                .target_name = desc.name,
            };
        }
    };

    var mock = MockProvider{};
    const provider = kernel.KernelProvider{
        .name = "mock",
        .ptr = @ptrCast(&mock),
        .compile_fn = MockProvider.compile,
    };

    var registry = kernel.KernelRegistry.init(testing.allocator);
    defer registry.deinit();

    var kp = KernelizePass{
        .registry = &registry,
        .providers = &.{provider},
    };

    var artifact = pass_mod.Artifact{ .pr = &program };
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator };
    try kp.pass().run(&artifact, &ctx);

    try testing.expect(mock.compiled);
    const ka = registry.get("test_region");
    try testing.expect(ka != null);
    try testing.expectEqualStrings("mock_kernel_data", ka.?.data);

    const rewritten = program.functions[0].eqns[0];
    try testing.expectEqual(pr.Prim.custom_call, rewritten.prim);

    const params = rewritten.params.slice(pr.Param, program.functions[0].params_store);
    try testing.expectEqualStrings(dispatcher_target_name, pr.param_call_target_name(params).?);
    try testing.expectEqualStrings("test_region", pr.param_call_kernel_key(params).?);
    try testing.expectEqualStrings("mock", pr.param_call_provider_name(params).?);
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

    const MockProvider = struct {
        fn compile(_: *anyopaque, desc: kernel.RegionDescriptor, allocator: std.mem.Allocator) kernel.CompileError!kernel.KernelArtifact {
            return .{
                .provider_name = "mock",
                .data = try allocator.dupe(u8, "mock_kernel_data"),
                .target_name = try allocator.dupe(u8, desc.name),
            };
        }
    };

    const provider = kernel.KernelProvider{
        .name = "mock",
        .ptr = undefined,
        .compile_fn = MockProvider.compile,
    };

    var registry = kernel.KernelRegistry.init(testing.allocator);
    defer registry.deinit();

    var kp = KernelizePass{
        .registry = &registry,
        .providers = &.{provider},
    };

    var artifact = pass_mod.Artifact{ .pr = &program };
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator };
    try kp.pass().run(&artifact, &ctx);

    const rewritten = program.functions[0].eqns[0];
    try testing.expectEqual(pr.Prim.custom_call, rewritten.prim);

    const params = rewritten.params.slice(pr.Param, program.functions[0].params_store);
    try testing.expectEqualStrings("attention_5op_v1", pr.param_call_carrier_hint(params).?);
    try testing.expect(pr.param_call_kernel_id(params) != null);
}

test "kernelize pass populates kernel package for rewritten region" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});

    try b.push_region("generic_region", .{ .kernelize = "mock" });
    const y = try b.emit(.exp, &.{x}, &.{});
    try b.pop_region();

    const func = try b.finish(&.{y});
    try program.add_function(func);

    const MockProvider = struct {
        fn compile(_: *anyopaque, desc: kernel.RegionDescriptor, allocator: std.mem.Allocator) kernel.CompileError!kernel.KernelArtifact {
            return .{
                .provider_name = "mock",
                .data = try allocator.dupe(u8, "mock_kernel_data"),
                .target_name = try allocator.dupe(u8, desc.name),
            };
        }
    };

    const provider = kernel.KernelProvider{
        .name = "mock",
        .ptr = undefined,
        .compile_fn = MockProvider.compile,
    };

    var registry = kernel.KernelRegistry.init(testing.allocator);
    defer registry.deinit();
    var package = kernel.KernelPackage.init(testing.allocator);
    defer package.deinit();

    var kp = KernelizePass{
        .registry = &registry,
        .package = &package,
        .providers = &.{provider},
    };

    var artifact = pass_mod.Artifact{ .pr = &program };
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator };
    try kp.pass().run(&artifact, &ctx);

    const rewritten = program.functions[0].eqns[0];
    const params = rewritten.params.slice(pr.Param, program.functions[0].params_store);
    try testing.expect(pr.param_call_kernel_id(params) != null);
    const kernel_id = pr.param_call_kernel_id(params).?;

    const pkg_artifact = package.get(kernel_id);
    try testing.expect(pkg_artifact != null);
    try testing.expectEqualStrings("mock_kernel_data", pkg_artifact.?.data);
}

test "kernelize pass falls back when provider returns Unsupported" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});

    try b.push_region("unsupported_region", .{ .kernelize = "mirage" });
    const y = try b.emit(.exp, &.{x}, &.{});
    try b.pop_region();

    const func = try b.finish(&.{y});
    try program.add_function(func);

    const StubUnsupportedProvider = struct {
        fn compile(_: *anyopaque, _: kernel.RegionDescriptor, _: std.mem.Allocator) kernel.CompileError!kernel.KernelArtifact {
            return error.Unsupported;
        }
    };

    const provider = kernel.KernelProvider{
        .name = "mirage",
        .ptr = undefined,
        .compile_fn = StubUnsupportedProvider.compile,
    };

    var registry = kernel.KernelRegistry.init(testing.allocator);
    defer registry.deinit();

    var kp = KernelizePass{
        .registry = &registry,
        .providers = &.{provider},
    };

    var artifact = pass_mod.Artifact{ .pr = &program };
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator };
    try kp.pass().run(&artifact, &ctx);

    try testing.expectEqual(@as(usize, 1), program.functions[0].eqns.len);
    try testing.expectEqual(pr.Prim.exp, program.functions[0].eqns[0].prim);
    try testing.expect(registry.get("unsupported_region") == null);
}

test "kernelize pass propagates provider errors other than Unsupported" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});

    try b.push_region("broken_region", .{ .kernelize = "mirage" });
    const y = try b.emit(.exp, &.{x}, &.{});
    try b.pop_region();

    const func = try b.finish(&.{y});
    try program.add_function(func);

    const StubProvider = struct {
        fn compile(_: *anyopaque, _: kernel.RegionDescriptor, _: std.mem.Allocator) kernel.CompileError!kernel.KernelArtifact {
            return error.MirageApiUnsupported;
        }
    };

    const provider = kernel.KernelProvider{
        .name = "mirage",
        .ptr = undefined,
        .compile_fn = StubProvider.compile,
    };

    var registry = kernel.KernelRegistry.init(testing.allocator);
    defer registry.deinit();

    var kp = KernelizePass{
        .registry = &registry,
        .providers = &.{provider},
    };

    var artifact = pass_mod.Artifact{ .pr = &program };
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator };
    try testing.expectError(error.MirageApiUnsupported, kp.pass().run(&artifact, &ctx));

    try testing.expectEqual(pr.Prim.exp, program.functions[0].eqns[0].prim);
    try testing.expect(registry.get("broken_region") == null);
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

    const MockProvider = struct {
        fn compile(_: *anyopaque, desc: kernel.RegionDescriptor, allocator: std.mem.Allocator) kernel.CompileError!kernel.KernelArtifact {
            return .{
                .provider_name = "mock",
                .data = try allocator.dupe(u8, "mock_kernel_data"),
                .target_name = try allocator.dupe(u8, desc.name),
            };
        }
    };

    const provider = kernel.KernelProvider{
        .name = "mock",
        .ptr = undefined,
        .compile_fn = MockProvider.compile,
    };

    var registry = kernel.KernelRegistry.init(testing.allocator);
    defer registry.deinit();

    var kp = KernelizePass{
        .registry = &registry,
        .providers = &.{provider},
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

test "kernelize pass can materialize without rewriting PR" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 2 });
    try b.push_region("matmul_region", .{ .kernelize = "mock" });
    const y = try b.emit(.exp, &.{x}, &.{});
    try b.pop_region();

    const func = try b.finish(&.{y});
    try program.add_function(func);

    const MockProvider = struct {
        fn compile(_: *anyopaque, desc: kernel.RegionDescriptor, allocator: std.mem.Allocator) kernel.CompileError!kernel.KernelArtifact {
            return .{
                .provider_name = "mock",
                .data = try allocator.dupe(u8, "mock_kernel_data"),
                .target_name = try allocator.dupe(u8, desc.name),
            };
        }
    };

    const provider = kernel.KernelProvider{
        .name = "mock",
        .ptr = undefined,
        .compile_fn = MockProvider.compile,
    };

    var registry = kernel.KernelRegistry.init(testing.allocator);
    defer registry.deinit();

    var kp = KernelizePass{
        .registry = &registry,
        .providers = &.{provider},
        .rewrite_regions = false,
    };

    const before_eqn_prim = program.functions[0].eqns[0].prim;
    var artifact = pass_mod.Artifact{ .pr = &program };
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator };
    try kp.pass().run(&artifact, &ctx);

    try testing.expectEqual(before_eqn_prim, program.functions[0].eqns[0].prim);
    try testing.expect(registry.get("matmul_region") != null);
}

test "kernelize pass outlined_eqn target naming mode" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});
    try b.push_region("region_name_unused", .{ .kernelize = "mock" });
    const y = try b.emit(.exp, &.{x}, &.{});
    try b.pop_region();

    const func = try b.finish(&.{y});
    try program.add_function(func);

    const MockProvider = struct {
        fn compile(_: *anyopaque, desc: kernel.RegionDescriptor, allocator: std.mem.Allocator) kernel.CompileError!kernel.KernelArtifact {
            return .{
                .provider_name = "mock",
                .data = try allocator.dupe(u8, "mock_kernel_data"),
                .target_name = try allocator.dupe(u8, desc.name),
            };
        }
    };

    const provider = kernel.KernelProvider{
        .name = "mock",
        .ptr = undefined,
        .compile_fn = MockProvider.compile,
    };

    var registry = kernel.KernelRegistry.init(testing.allocator);
    defer registry.deinit();

    var kp = KernelizePass{
        .registry = &registry,
        .providers = &.{provider},
        .rewrite_regions = false,
        .target_name_mode = .outlined_eqn,
    };

    var artifact = pass_mod.Artifact{ .pr = &program };
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator };
    try kp.pass().run(&artifact, &ctx);

    try testing.expect(registry.get("main_outlined_0") != null);
}

test "kernelize pass deduplicates same-shape regions across a function" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    // Two exp regions with identical f32[2] input/output — same shape signature.
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

    const MockProvider = struct {
        compile_count: usize = 0,

        fn compile(ptr: *anyopaque, desc: kernel.RegionDescriptor, allocator: std.mem.Allocator) kernel.CompileError!kernel.KernelArtifact {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.compile_count += 1;
            return .{
                .provider_name = "mock",
                .data = try allocator.dupe(u8, "payload"),
                .target_name = try allocator.dupe(u8, desc.name),
            };
        }
    };

    var mock = MockProvider{};
    const provider = kernel.KernelProvider{
        .name = "mock",
        .ptr = @ptrCast(&mock),
        .compile_fn = MockProvider.compile,
    };

    var registry = kernel.KernelRegistry.init(testing.allocator);
    defer registry.deinit();

    var kp = KernelizePass{
        .registry = &registry,
        .providers = &.{provider},
    };

    var artifact = pass_mod.Artifact{ .pr = &program };
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator };
    try kp.pass().run(&artifact, &ctx);

    // Both regions must be registered under their own target names.
    try testing.expect(registry.get("region_a") != null);
    try testing.expect(registry.get("region_b") != null);

    // Provider was only called once — region_b reused the cached artifact.
    try testing.expectEqual(@as(usize, 1), mock.compile_count);

    // Both regions must have been rewritten to custom_call.
    const eqns = program.functions[0].eqns;
    try testing.expectEqual(@as(usize, 2), eqns.len);
    try testing.expectEqual(pr.Prim.custom_call, eqns[0].prim);
    try testing.expectEqual(pr.Prim.custom_call, eqns[1].prim);
}

test "kernelize pass does not deduplicate regions with different shapes" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    // Two exp regions with DIFFERENT shapes — distinct signatures.
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

    const MockProvider = struct {
        compile_count: usize = 0,

        fn compile(ptr: *anyopaque, desc: kernel.RegionDescriptor, allocator: std.mem.Allocator) kernel.CompileError!kernel.KernelArtifact {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.compile_count += 1;
            return .{
                .provider_name = "mock",
                .data = try allocator.dupe(u8, "payload"),
                .target_name = try allocator.dupe(u8, desc.name),
            };
        }
    };

    var mock = MockProvider{};
    const provider = kernel.KernelProvider{
        .name = "mock",
        .ptr = @ptrCast(&mock),
        .compile_fn = MockProvider.compile,
    };

    var registry = kernel.KernelRegistry.init(testing.allocator);
    defer registry.deinit();

    var kp = KernelizePass{
        .registry = &registry,
        .providers = &.{provider},
    };

    var artifact = pass_mod.Artifact{ .pr = &program };
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator };
    try kp.pass().run(&artifact, &ctx);

    // Different shapes — provider must be called for each region.
    try testing.expectEqual(@as(usize, 2), mock.compile_count);
}
