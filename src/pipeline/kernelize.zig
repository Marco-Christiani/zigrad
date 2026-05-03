//! Kernelization Pass
//!
//! PR -> PR pass that consults a pre-computed `KernelStore` to replace annotated
//!  regions with `custom_call` ops. Never invokes providers or performs compilation.
//!
//! For each region with a `kernelize` annotation, the pass:
//!  1. Computes a kernel signature from the region's op signature.
//!  2. Looks up the signature in the store for a tuning decision.
//!  3. Profitable decisions: rewrites the region's ops into a single
//!      `custom_call` op carrying kernel_key, provider_name, and carrier metadata.
//!  4. Negative or absent decisions: leaves the region unchanged for baseline lowering.
//!
//! The store is populated externally by `tune()` (src/tune.zig). This pass is a
//!  pure consumer, it does not modifies the store.
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
    inputs: []const *pr.Var = &.{},
    outputs: []const *pr.Var = &.{},
    kernel_key: []const u8 = &.{},
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
            var stdout_writer = std.Io.File.stdout().writer(ctx.io, &buf);
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
                    const ops_str = build_ops_str(entries_alloc, desc) catch "";
                    const shape = build_shape_str(entries_alloc, desc) catch "";
                    e.append(entries_alloc, .{
                        .name = candidate.region.name,
                        .provider = candidate.provider_name,
                        .ops = ops_str,
                        .shape = shape,
                        .outcome = .fallback,
                    }) catch {};
                }
                continue;
            };

            switch (decision) {
                .profitable => |art| {
                    try rewrites.append(temp_allocator, .{
                        .region = candidate.region,
                        .provider_name = art.provider_name,
                        .inputs = try temp_allocator.dupe(*pr.Var, desc.inputs),
                        .outputs = try temp_allocator.dupe(*pr.Var, desc.outputs),
                        .kernel_key = try temp_allocator.dupe(u8, kernel_signature),
                    });
                    log.debug("store: profitable decision for region '{s}' -> '{s}'", .{ candidate.region.name, art.target_name });

                    if (entries) |e| {
                        const ops_str = build_ops_str(entries_alloc, desc) catch "";
                        const shape = build_shape_str(entries_alloc, desc) catch "";
                        e.append(entries_alloc, .{
                            .name = candidate.region.name,
                            .provider = art.provider_name,
                            .ops = ops_str,
                            .shape = shape,
                            .outcome = .compiled,
                        }) catch {};
                    }
                },
                .negative => |reason| {
                    log.debug("store: negative decision for region '{s}': {s}", .{ candidate.region.name, reason });
                    if (entries) |e| {
                        const ops_str = build_ops_str(entries_alloc, desc) catch "";
                        const shape = build_shape_str(entries_alloc, desc) catch "";
                        e.append(entries_alloc, .{
                            .name = candidate.region.name,
                            .provider = candidate.provider_name,
                            .ops = ops_str,
                            .shape = shape,
                            .outcome = .fallback,
                        }) catch {};
                    }
                },
            }
        }

        if (rewrites.items.len == 0) return func;

        const arena = program.allocator();

        var new_ops = try std.ArrayList(*pr.Op).initCapacity(arena, func.ops.len);
        var op_index: usize = 0;
        while (op_index < func.ops.len) {
            if (find_rewrite_starting_at(rewrites.items, op_index)) |rewrite| {
                const new_op = try build_custom_call_op(arena, rewrite);
                try new_ops.append(arena, new_op);
                op_index += @as(usize, @intCast(rewrite.region.op_len));
                continue;
            }
            try new_ops.append(arena, func.ops[op_index]);
            op_index += 1;
        }

        return .{
            .name = func.name,
            .params = func.params,
            .returns = func.returns,
            .ops = try new_ops.toOwnedSlice(arena),
            .regions = &.{},
            .var_count = func.var_count,
        };
    }

    fn find_rewrite_starting_at(rewrites: []const KernelCandidate, op_start: usize) ?KernelCandidate {
        for (rewrites) |entry| {
            if (entry.region.op_len == 0) continue;
            if (@as(usize, @intCast(entry.region.op_start)) == op_start) return entry;
        }
        return null;
    }

    fn is_region_nested(region: pr.Region, candidates: []const KernelCandidate) bool {
        if (region.op_len == 0) return false;
        const start: usize = @intCast(region.op_start);
        const end: usize = start + @as(usize, @intCast(region.op_len));
        for (candidates) |other| {
            if (other.region.op_len == 0) continue;
            const other_start: usize = @intCast(other.region.op_start);
            const other_end: usize = other_start + @as(usize, @intCast(other.region.op_len));
            const contains = (other_start <= start) and (other_end >= end);
            const strictly_larger = (other_start < start) or (other_end > end);
            if (contains and strictly_larger) return true;
        }
        return false;
    }

    /// Build a new custom_call Op from a rewrite candidate.
    fn build_custom_call_op(arena: std.mem.Allocator, rewrite: KernelCandidate) !*pr.Op {
        const out_avals = try arena.alloc(pr.Aval, rewrite.outputs.len);
        for (rewrite.outputs, 0..) |out_var, idx| {
            out_avals[idx] = out_var.aval;
        }

        const operands = try arena.alloc(pr.Operand, rewrite.inputs.len);
        const new_op = try arena.create(pr.Op);

        for (rewrite.inputs, 0..) |in_var, i| {
            operands[i] = .{
                .value = in_var,
                .owner = new_op,
                .index = @intCast(i),
            };
            // Wire into the Var's use-list.
            operands[i].next = in_var.first_use;
            if (in_var.first_use) |head| head.prev = &operands[i];
            in_var.first_use = &operands[i];
        }

        const out_vars = try arena.alloc(*pr.Var, rewrite.outputs.len);
        for (rewrite.outputs, 0..) |out_var, i| {
            out_vars[i] = out_var;
            out_var.defining_op = new_op;
        }

        new_op.* = .{
            .inputs = operands,
            .outputs = out_vars,
            .params = .{ .custom_call = .{
                .target_name = try arena.dupe(u8, dispatcher_target_name),
                .has_side_effect = false,
                .out_avals = out_avals,
                .kernel_key = try arena.dupe(u8, rewrite.kernel_key),
                .provider_name = try arena.dupe(u8, rewrite.provider_name),
            } },
        };

        return new_op;
    }
};

// ============================================================================
// Kernel Signature
// ============================================================================

/// Build a deterministic kernel signature string for a region descriptor.
///
/// The key encodes the region's op sequence as:
///   `<prim>,<in0_aval><in1_aval>...-><out0_aval>...;<next_op>...`
///
/// Two regions produce the same key iff they have identical op sequences with
/// matching input/output dtypes and dims. This is the deduplication criterion
/// used by `tune()` and the lookup key consulted by `KernelizePass`.
pub fn compute_kernel_signature(allocator: std.mem.Allocator, desc: kernel.RegionDescriptor) ![]const u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;

    for (desc.ops, 0..) |op, oi| {
        if (oi > 0) try w.writeByte(';');
        try w.writeAll(@tagName(op.prim()));
        try w.writeByte(',');
        for (op.inputs, 0..) |operand, i| {
            if (i > 0) try w.writeByte(',');
            try write_aval_key(w, operand.value.aval);
        }
        try w.writeByte('>');
        for (op.outputs, 0..) |out_var, i| {
            if (i > 0) try w.writeByte(',');
            try write_aval_key(w, out_var.aval);
        }
    }

    return try aw.toOwnedSlice();
}

fn write_aval_key(w: anytype, aval: pr.Aval) !void {
    switch (aval) {
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
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    for (desc.ops, 0..) |op, i| {
        if (i > 0) try w.writeByte('+');
        try w.writeAll(@tagName(op.prim()));
    }
    return try aw.toOwnedSlice();
}

fn build_shape_str(allocator: std.mem.Allocator, desc: kernel.RegionDescriptor) ![]const u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    for (desc.inputs, 0..) |in_var, i| {
        if (i > 0) try w.writeByte('x');
        try write_aval_key(w, in_var.aval);
    }
    return try aw.toOwnedSlice();
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
    const y = try b.emit(.{ .exp = {} }, &.{x});
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
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator, .io = std.testing.io };
    try kp.pass().run(&artifact, &ctx);

    // Region should be rewritten to custom_call.
    try testing.expectEqual(@as(usize, 1), program.functions[0].ops.len);
    const rewritten = program.functions[0].ops[0];
    try testing.expectEqual(pr.Prim.custom_call, rewritten.prim());

    try testing.expectEqualStrings(dispatcher_target_name, rewritten.params.custom_call.target_name);
    // kernel_key is the kernel signature used for store lookup at dispatch time.
    try testing.expectEqualStrings("exp,f32[2]>f32[2]", rewritten.params.custom_call.kernel_key.?);
    try testing.expectEqualStrings("mock", rewritten.params.custom_call.provider_name.?);
}

test "kernelize pass skips negative decision" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});

    try b.push_region("test_region", .{ .kernelize = "mock" });
    const y = try b.emit(.{ .exp = {} }, &.{x});
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
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator, .io = std.testing.io };
    try kp.pass().run(&artifact, &ctx);

    // Region should be left unchanged (negative decision).
    try testing.expectEqual(@as(usize, 1), program.functions[0].ops.len);
    try testing.expectEqual(pr.Prim.exp, program.functions[0].ops[0].prim());
}

test "kernelize pass skips absent key" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});

    try b.push_region("test_region", .{ .kernelize = "mock" });
    const y = try b.emit(.{ .exp = {} }, &.{x});
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
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator, .io = std.testing.io };
    try kp.pass().run(&artifact, &ctx);

    // Region should be left unchanged (absent key).
    try testing.expectEqual(@as(usize, 1), program.functions[0].ops.len);
    try testing.expectEqual(pr.Prim.exp, program.functions[0].ops[0].prim());
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
    const a = try b.emit(.{ .exp = {} }, &.{x});
    const b_out = try b.emit(.{ .log = {} }, &.{y});
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
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator, .io = std.testing.io };
    try kp.pass().run(&artifact, &ctx);

    try testing.expectEqual(@as(usize, 1), program.functions[0].ops.len);
    const rewritten = program.functions[0].ops[0];
    try testing.expectEqual(pr.Prim.custom_call, rewritten.prim());

    try testing.expectEqual(@as(usize, 2), rewritten.outputs.len);

    const out_avals = rewritten.params.custom_call.out_avals;
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
    const out_a = try b.emit(.{ .exp = {} }, &.{x});
    try b.pop_region();

    try b.push_region("region_b", .{ .kernelize = "mock" });
    const out_b = try b.emit(.{ .exp = {} }, &.{y});
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
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator, .io = std.testing.io };
    try kp.pass().run(&artifact, &ctx);

    // Both regions must have been rewritten to custom_call.
    const ops = program.functions[0].ops;
    try testing.expectEqual(@as(usize, 2), ops.len);
    try testing.expectEqual(pr.Prim.custom_call, ops[0].prim());
    try testing.expectEqual(pr.Prim.custom_call, ops[1].prim());
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
    const out_small = try b.emit(.{ .exp = {} }, &.{x});
    try b.pop_region();

    try b.push_region("region_large", .{ .kernelize = "mock" });
    const out_large = try b.emit(.{ .exp = {} }, &.{y});
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
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator, .io = std.testing.io };
    try kp.pass().run(&artifact, &ctx);

    // Only region_small should be rewritten; region_large has no store decision.
    const ops = program.functions[0].ops;
    try testing.expectEqual(@as(usize, 2), ops.len);
    try testing.expectEqual(pr.Prim.custom_call, ops[0].prim());
    try testing.expectEqual(pr.Prim.exp, ops[1].prim());
}

test "kernelize pass skips rewriting when rewrite_regions is false" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});
    try b.push_region("test_region", .{ .kernelize = "mock" });
    const y = try b.emit(.{ .exp = {} }, &.{x});
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

    const before_op_prim = program.functions[0].ops[0].prim();
    var artifact = pass_mod.Artifact{ .pr = &program };
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator, .io = std.testing.io };
    try kp.pass().run(&artifact, &ctx);

    // Ops should be unchanged when rewrite_regions is false.
    try testing.expectEqual(before_op_prim, program.functions[0].ops[0].prim());
}
