//! PR kernel-provider substitution.
//!
//! `KernelizePass` queries a populated `KernelStore` and replaces annotated
//!  regions when a decision supplies an artifact.
//!
//! Missing and negative decisions leave the region unchanged. Provider
//!  compilation and tuning run before this transform.
const std = @import("std");
const compilation = @import("../compilation.zig");
const device = @import("../device.zig");
const region_view = @import("region_view.zig");
const pr = @import("pr.zig");
const kernel = @import("kernel.zig");

const log = std.log.scoped(.@"zg/kernelize");

const KernelCandidate = struct {
    region: pr.Region,
    provider_name: []const u8,

    inputs: []const *pr.Var = &.{},
    outputs: []const *pr.Var = &.{},
    kernel_key: []const u8 = &.{},
};

const KernelEntryOutcome = enum { compiled, dedup, fallback };

/// Diagnostic record for one region encountered during kernelization.
///
/// `name` and `provider` borrow storage that must outlive the report.
///
/// The report arena stores `ops` and `shape` until `Report.deinit`.
const KernelEntry = struct {
    name: []const u8,
    provider: []const u8,
    ops: []const u8,
    shape: []const u8,
    outcome: KernelEntryOutcome,
};

/// Diagnostics collected while kernelizing a program.
pub const Report = struct {
    arena: std.heap.ArenaAllocator,
    entries: std.ArrayList(KernelEntry) = .empty,

    pub fn init(allocator: std.mem.Allocator) Report {
        return .{ .arena = .init(allocator) };
    }

    pub fn deinit(self: *Report) void {
        self.entries.deinit(self.arena.allocator());
        self.arena.deinit();
        self.* = undefined;
    }
};

/// Emits diagnostics collected by a preceding kernelization operation.
pub const DumpKernels = struct {
    pub const Input = *pr.Program;
    pub const Output = *pr.Program;

    report: *const Report,

    pub fn run(self: DumpKernels, program: Input, ctx: *compilation.Context) !Output {
        if (self.report.entries.items.len == 0) return program;

        var buffer: [8192]u8 = undefined;
        var stdout_writer = std.Io.File.stdout().writer(ctx.io, &buffer);
        const out = &stdout_writer.interface;
        try dump_kernel_entries(out, self.report.entries.items);
        try out.flush();
        return program;
    }
};

/// Kernelization pass state.
///
/// Consults a pre-computed `KernelStore` to decide which annotated regions
///  to replace with `custom_call` ops. The store is the sole data source.
///
/// This operation never invokes providers or performs compilation.
pub const KernelizePass = struct {
    pub const Input = *pr.Program;
    pub const Output = *pr.Program;

    /// Pre-computed tuning decisions populated before this operation runs.
    ///
    /// The store must outlive this operation.
    store: *const kernel.KernelStore,

    /// Device used when the decision store was populated.
    device: device.Device,

    /// Optional destination for diagnostics consumed by `DumpKernels`.
    report: ?*Report = null,

    pub fn run(
        self: *KernelizePass,
        program: Input,
        ctx: *compilation.Context,
    ) !Output {
        try self.run_program(program, ctx.allocator);
        return program;
    }

    fn run_program(
        self: *KernelizePass,
        program: *pr.Program,
        allocator: std.mem.Allocator,
    ) !void {
        const entries = if (self.report) |report| &report.entries else null;
        const entries_allocator = if (self.report) |report|
            report.arena.allocator()
        else
            allocator;

        for (program.functions, 0..) |func, idx| {
            const rewritten = self.kernelize_function(
                program,
                func,
                allocator,
                entries,
                entries_allocator,
            ) catch |err| {
                if (!@import("builtin").is_test)
                    log.err("kernelization failed for function '{s}': {}", .{ func.name, err });
                return err;
            };
            program.functions[idx] = rewritten;
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

            const desc = try region_view.describe(temp_allocator, func, candidate.region);
            defer desc.deinit(temp_allocator);

            if (desc.outputs.len == 0) continue;

            const region_signature = try kernel.compute_region_signature(temp_allocator, desc);
            defer temp_allocator.free(region_signature.bytes);
            const decision_key = try kernel.make_decision_key(
                temp_allocator,
                candidate.provider_name,
                self.device,
                region_signature,
            );
            defer temp_allocator.free(decision_key.bytes);

            const decision = store.get(decision_key) orelse {
                log.debug("store: no decision for provider '{s}' region '{s}' (shape '{s}'), skipping", .{
                    candidate.provider_name,
                    candidate.region.name,
                    region_signature.bytes,
                });
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
                        .kernel_key = try temp_allocator.dupe(u8, decision_key.bytes),
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
            if (find_rewrite_starting_at(func, rewrites.items, op_index)) |rewrite| {
                const replacement_id = func.ops[op_index].id;
                const new_op = try build_custom_call_op(arena, rewrite, replacement_id);
                try new_ops.append(arena, new_op);
                op_index += rewrite.region.op_ids.len;
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

    fn find_rewrite_starting_at(func: pr.Function, rewrites: []const KernelCandidate, op_start: usize) ?KernelCandidate {
        for (rewrites) |entry| {
            if (entry.region.op_ids.len == 0) continue;
            const start = func.op_index_by_id(entry.region.op_ids[0]) orelse continue;
            if (start == op_start) return entry;
        }
        return null;
    }

    fn is_region_nested(region: pr.Region, candidates: []const KernelCandidate) bool {
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

    /// Build a new custom_call Op from a rewrite candidate.
    fn build_custom_call_op(arena: std.mem.Allocator, rewrite: KernelCandidate, replacement_id: u32) !*pr.Op {
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
            .id = replacement_id,
            .inputs = operands,
            .outputs = out_vars,
            .params = .{ .custom_call = .{
                .target_name = try arena.dupe(u8, kernel.dispatch_target_name),
                .has_side_effect = false,
                .out_avals = out_avals,
                .kernel_key = try arena.dupe(u8, rewrite.kernel_key),
                .provider_name = try arena.dupe(u8, rewrite.provider_name),
            } },
        };

        return new_op;
    }
};

// Kernel Dump Helpers

fn build_ops_str(allocator: std.mem.Allocator, desc: region_view.RegionView) ![]const u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    for (desc.ops, 0..) |op, i| {
        if (i > 0) try w.writeByte('+');
        try w.writeAll(@tagName(op.prim()));
    }
    return try aw.toOwnedSlice();
}

fn build_shape_str(allocator: std.mem.Allocator, desc: region_view.RegionView) ![]const u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    for (desc.inputs, 0..) |in_var, i| {
        if (i > 0) try w.writeByte('x');
        try kernel.write_aval_signature(w, in_var.aval);
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

// Tests.

const test_device: device.Device = .{ .platform = .cpu };

fn make_test_decision_key(
    allocator: std.mem.Allocator,
    provider_name: []const u8,
    region_signature: []const u8,
) !kernel.DecisionKey {
    return try kernel.make_decision_key(
        allocator,
        provider_name,
        test_device,
        .{ .bytes = region_signature },
    );
}

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

    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    const decision_key = try make_test_decision_key(testing.allocator, "mock", "exp,f32[2]>f32[2]");
    defer testing.allocator.free(decision_key.bytes);
    try store.put_profitable(decision_key, .{
        .provider_name = "mock",
        .data = "stored_kernel_data",
        .target_name = "test_region",
    });

    var kp = KernelizePass{
        .store = &store,
        .device = test_device,
    };

    var ctx = compilation.Context{ .allocator = testing.allocator, .io = std.testing.io };
    _ = try kp.run(&program, &ctx);

    // Region should be rewritten to custom_call.
    try testing.expectEqual(@as(usize, 1), program.functions[0].ops.len);
    const rewritten = program.functions[0].ops[0];
    try testing.expectEqual(pr.Prim.custom_call, rewritten.prim());

    try testing.expectEqualStrings(kernel.dispatch_target_name, rewritten.params.custom_call.target_name);
    try testing.expectEqualStrings(decision_key.bytes, rewritten.params.custom_call.kernel_key.?);
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
    const decision_key = try make_test_decision_key(testing.allocator, "mock", "exp,f32[2]>f32[2]");
    defer testing.allocator.free(decision_key.bytes);
    try store.put_negative(decision_key, "unsupported");

    var kp = KernelizePass{
        .store = &store,
        .device = test_device,
    };

    var ctx = compilation.Context{ .allocator = testing.allocator, .io = std.testing.io };
    _ = try kp.run(&program, &ctx);

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

    // An empty store leaves every region unchanged.
    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();

    var kp = KernelizePass{
        .store = &store,
        .device = test_device,
    };

    var ctx = compilation.Context{ .allocator = testing.allocator, .io = std.testing.io };
    _ = try kp.run(&program, &ctx);

    // Region should be left unchanged (absent key).
    try testing.expectEqual(@as(usize, 1), program.functions[0].ops.len);
    try testing.expectEqual(pr.Prim.exp, program.functions[0].ops[0].prim());
}

test "kernelize pass does not reuse another provider decision" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var builder = try pr.FunctionBuilder.init(&program, "test");
    defer builder.deinit();
    const input = try builder.param_tensor(.f32, &.{2});
    try builder.push_region("test_region", .{ .kernelize = "requested" });
    const output = try builder.emit(.{ .exp = {} }, &.{input});
    try builder.pop_region();
    const function = try builder.finish(&.{output});
    try program.add_function(function);

    const other_key = try make_test_decision_key(testing.allocator, "other", "exp,f32[2]>f32[2]");
    defer testing.allocator.free(other_key.bytes);
    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    try store.put_profitable(other_key, .{
        .provider_name = "other",
        .data = "payload",
        .target_name = "other_artifact",
    });

    var kernelize = KernelizePass{
        .store = &store,
        .device = test_device,
    };
    var context = compilation.Context{ .allocator = testing.allocator, .io = testing.io };
    _ = try kernelize.run(&program, &context);

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

    const desc = try region_view.describe(testing.allocator, func, func.regions[0]);
    defer desc.deinit(testing.allocator);
    const region_signature = try kernel.compute_region_signature(testing.allocator, desc);
    defer testing.allocator.free(region_signature.bytes);
    const decision_key = try kernel.make_decision_key(
        testing.allocator,
        "mock",
        test_device,
        region_signature,
    );
    defer testing.allocator.free(decision_key.bytes);

    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    try store.put_profitable(decision_key, .{
        .provider_name = "mock",
        .data = "mock_kernel_data",
        .target_name = "multi_out",
    });

    var kp = KernelizePass{
        .store = &store,
        .device = test_device,
    };

    var ctx = compilation.Context{ .allocator = testing.allocator, .io = std.testing.io };
    _ = try kp.run(&program, &ctx);

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

    // Both regions have the same signature because their shapes and operations match.
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

    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    const decision_key = try make_test_decision_key(testing.allocator, "mock", "exp,f32[2]>f32[2]");
    defer testing.allocator.free(decision_key.bytes);
    try store.put_profitable(decision_key, .{
        .provider_name = "mock",
        .data = "payload",
        .target_name = "region_a",
    });

    var kp = KernelizePass{
        .store = &store,
        .device = test_device,
    };

    var ctx = compilation.Context{ .allocator = testing.allocator, .io = std.testing.io };
    _ = try kp.run(&program, &ctx);

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

    // The different shapes produce distinct kernel signatures.
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

    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    const decision_key = try make_test_decision_key(testing.allocator, "mock", "exp,f32[2]>f32[2]");
    defer testing.allocator.free(decision_key.bytes);
    try store.put_profitable(decision_key, .{
        .provider_name = "mock",
        .data = "payload",
        .target_name = "region_small",
    });

    var kp = KernelizePass{
        .store = &store,
        .device = test_device,
    };

    var ctx = compilation.Context{ .allocator = testing.allocator, .io = std.testing.io };
    _ = try kp.run(&program, &ctx);

    // Only region_small has a store decision, so region_large remains unchanged.
    const ops = program.functions[0].ops;
    try testing.expectEqual(@as(usize, 2), ops.len);
    try testing.expectEqual(pr.Prim.custom_call, ops[0].prim());
    try testing.expectEqual(pr.Prim.exp, ops[1].prim());
}
