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

const RewriteCandidate = struct {
    region: pr.Region,
    inputs: []const pr.VarId,
    outputs: []const pr.VarId,
    kernel_key: []const u8,
    provider_name: []const u8,
};

const RegionCandidate = struct {
    region: pr.Region,
    provider: kernel.KernelProvider,
    provider_name: []const u8,
};

/// Kernelization pass state. Holds the registry and providers.
///
/// Create this struct, then call `pass()` to get a pipeline-compatible
/// `pass_mod.Pass` value.
pub const KernelizePass = struct {
    registry: *kernel.KernelRegistry,
    providers: []const kernel.KernelProvider,

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

        const program = artifact.pr;
        const functions: []pr.Function = @constCast(program.functions);
        for (program.functions, 0..) |func, idx| {
            const rewritten = self.kernelize_function(program, func, ctx.allocator) catch |err| {
                log.err("kernelization failed for function '{s}': {}", .{ func.name, err });
                return err;
            };
            functions[idx] = rewritten;
        }
    }

    fn kernelize_function(self: *KernelizePass, program: *pr.Program, func: pr.Function, temp_allocator: std.mem.Allocator) !pr.Function {
        if (func.regions.len == 0) return func;

        var candidates = try std.ArrayList(RegionCandidate).initCapacity(temp_allocator, func.regions.len);
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

        var rewrites = try std.ArrayList(RewriteCandidate).initCapacity(temp_allocator, func.regions.len);
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

            const ka = candidate.provider.compile(desc, self.registry.allocator()) catch |err| switch (err) {
                error.Unsupported => {
                    log.debug("provider '{s}' cannot handle region '{s}', falling back to baseline", .{ candidate.provider_name, candidate.region.name });
                    continue;
                },
                else => {
                    log.err("provider '{s}' failed to compile region '{s}': {s}", .{ candidate.provider_name, candidate.region.name, @errorName(err) });
                    return err;
                },
            };

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

            try rewrites.append(temp_allocator, .{
                .region = candidate.region,
                .inputs = try temp_allocator.dupe(pr.VarId, desc.inputs),
                .outputs = try temp_allocator.dupe(pr.VarId, desc.outputs),
                .kernel_key = ka.target_name,
                .provider_name = ka.provider_name,
            });
        }

        if (rewrites.items.len == 0) return func;

        const allocator = program.allocator();

        var eqns = try std.ArrayList(pr.Eqn).initCapacity(allocator, func.eqns.len);
        var varids_store = try std.ArrayList(pr.VarId).initCapacity(allocator, func.varids_store.len);
        var params_store = try std.ArrayList(pr.Param).initCapacity(allocator, func.params_store.len + rewrites.items.len * 5);

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

    fn find_rewrite_starting_at(rewrites: []const RewriteCandidate, eqn_start: usize) ?RewriteCandidate {
        for (rewrites) |entry| {
            if (entry.region.eqn_len == 0) continue;
            if (@as(usize, @intCast(entry.region.eqn_start)) == eqn_start) return entry;
        }
        return null;
    }

    fn is_region_nested(region: pr.Region, candidates: []const RegionCandidate) bool {
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
        rewrite: RewriteCandidate,
    ) !void {
        if (rewrite.outputs.len == 0) return error.InvalidRegion;

        const out_avals = try allocator.alloc(pr.Aval, rewrite.outputs.len);
        for (rewrite.outputs, 0..) |out_id, idx| {
            out_avals[idx] = func.avals[@intCast(out_id)];
        }

        const kernel_key = try allocator.dupe(u8, rewrite.kernel_key);
        const provider_name = try allocator.dupe(u8, rewrite.provider_name);
        const target_name = try allocator.dupe(u8, dispatcher_target_name);

        const params = [_]pr.Param{
            .{ .call_target_name = target_name },
            .{ .call_kernel_key = kernel_key },
            .{ .call_provider_name = provider_name },
            .{ .has_side_effect = false },
            .{ .out_avals = out_avals },
        };

        const in_span = try append_varids(allocator, varids_store, rewrite.inputs);
        const out_span = try append_varids(allocator, varids_store, rewrite.outputs);
        const param_span = try append_params(allocator, params_store, params[0..]);

        try eqns.append(allocator, .{
            .prim = .custom_call,
            .inputs = in_span,
            .outputs = out_span,
            .params = param_span,
        });
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
