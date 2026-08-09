//! PR kernel-provider substitution.
//!
//! `OutlineCandidates` turns provider regions into callable PR functions.
//! `KernelizePass` queries a populated `KernelStore` and replaces calls when a
//!  decision supplies an artifact.
//!
//! Missing and negative decisions leave the ordinary function call in place.
//!
//! Provider compilation and tuning run after outlining and before substitution.
const std = @import("std");
const compilation = @import("../compilation.zig");
const device = @import("../device.zig");
const output_mod = @import("../output.zig");
const fingerprint = @import("fingerprint.zig");
const outline = @import("outline.zig");
const pr = @import("pr.zig");
const kernel = @import("kernel.zig");

const log = std.log.scoped(.@"zg/kernelize");

const KernelEntryOutcome = enum { compiled, fallback };

/// Diagnostic record for one function encountered during kernelization.
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
    target: output_mod.Target,

    pub fn run(self: DumpKernels, program: Input, ctx: *compilation.Context) !Output {
        if (self.report.entries.items.len == 0) return program;

        const task = struct {
            entries: []const KernelEntry,

            pub fn emit(value: @This(), writer: *std.Io.Writer) !void {
                try dump_kernel_entries(writer, value.entries);
            }
        }{ .entries = self.report.entries.items };
        try output_mod.write(ctx.io, self.target, task);
        return program;
    }
};

/// Outlines provider-request regions into callable PR functions.
///
/// Outermost requests take precedence when regions are nested. Their function
///  retains the provider request, and nested provider requests are consumed.
pub const OutlineCandidates = struct {
    pub const Input = *pr.Program;
    pub const Output = *pr.Program;

    pub fn run(_: OutlineCandidates, program: Input, ctx: *compilation.Context) !Output {
        while (try find_outermost_request(program)) |request| {
            const result = try outline.apply(
                program,
                ctx.allocator,
                request.function_index,
                request.region.id,
                .{ .function_annotations = request.region.annotations },
            );
            try consume_nested_requests(program, result.function_index);
        }
        return program;
    }
};

const OutlineRequest = struct {
    function_index: usize,
    region: pr.Region,
};

fn find_outermost_request(program: *const pr.Program) kernel.AnnotationError!?OutlineRequest {
    var found: ?OutlineRequest = null;
    for (program.functions, 0..) |func, function_index| {
        if (try kernel.requested_provider(func) != null) continue;
        for (func.regions) |region| {
            if (try kernel.requested_provider(region) == null) continue;
            if (found == null or region.op_ids.len > found.?.region.op_ids.len) {
                found = .{ .function_index = function_index, .region = region };
            }
        }
    }
    return found;
}

fn consume_nested_requests(program: *pr.Program, function_index: usize) std.mem.Allocator.Error!void {
    const arena = program.allocator();
    const func = &program.functions[function_index];
    const has_nested_request = for (func.regions) |region| {
        if (region.find_annotation(kernel.provider_annotation_name) != null) break true;
    } else false;
    if (!has_nested_request) return;

    const regions = try arena.alloc(pr.Region, func.regions.len);
    for (func.regions, regions) |region, *updated| {
        var annotations = try std.ArrayList(pr.Annotation).initCapacity(arena, region.annotations.len);
        for (region.annotations) |annotation| {
            if (!std.mem.eql(u8, annotation.name, kernel.provider_annotation_name)) {
                try annotations.append(arena, annotation);
            }
        }
        updated.* = region;
        updated.annotations = try annotations.toOwnedSlice(arena);
    }
    func.regions = regions;
}

/// Kernelization pass state.
///
/// Consults a pre-computed `KernelStore` to decide which calls to provider
///  functions become `custom_call` ops.
pub const KernelizePass = struct {
    pub const Input = *pr.Program;
    pub const Output = *pr.Program;

    /// Pre-computed tuning decisions populated before this operation runs, borrowed.
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
        try kernel.require_outlined_requests(program);

        const entries = if (self.report) |report| &report.entries else null;
        const entries_allocator = if (self.report) |report|
            report.arena.allocator()
        else
            allocator;

        for (program.functions) |func| {
            self.kernelize_function(program, func, allocator, entries, entries_allocator) catch |err| {
                if (!@import("builtin").is_test)
                    log.err("kernelization failed for function '{s}': {}", .{ func.name, err });
                return err;
            };
        }
    }

    fn kernelize_function(
        self: *KernelizePass,
        program: *pr.Program,
        func: pr.Function,
        temp_allocator: std.mem.Allocator,
        entries: ?*std.ArrayList(KernelEntry),
        entries_alloc: std.mem.Allocator,
    ) !void {
        for (func.ops) |op| {
            if (op.prim() != .call) continue;
            const candidate = program.get_function(op.params.call.callee) orelse return error.CallUnresolvedCallee;
            const provider_name = (try kernel.requested_provider(candidate)) orelse continue;
            const function_fingerprint = try fingerprint.function(temp_allocator, candidate);
            const decision_key = try kernel.make_decision_key(
                temp_allocator,
                provider_name,
                self.device,
                function_fingerprint,
            );
            defer temp_allocator.free(decision_key.bytes);

            const decision = self.store.get(decision_key);
            const outcome: KernelEntryOutcome = if (decision) |selected| switch (selected) {
                .profitable => |stored| outcome: {
                    try rewrite_call(
                        program.allocator(),
                        op,
                        decision_key.bytes,
                        pr.function_may_have_side_effects(program, candidate),
                    );
                    log.debug("selected provider '{s}' for function '{s}'", .{ stored.provider_name, candidate.name });
                    break :outcome .compiled;
                },
                .negative => |reason| outcome: {
                    log.debug("provider '{s}' declined function '{s}': {s}", .{ provider_name, candidate.name, reason });
                    break :outcome .fallback;
                },
            } else .fallback;

            if (entries) |list| {
                try list.append(entries_alloc, .{
                    .name = candidate.name,
                    .provider = provider_name,
                    .ops = try build_ops_str(entries_alloc, candidate),
                    .shape = try build_shape_str(entries_alloc, candidate),
                    .outcome = outcome,
                });
            }
        }
    }

    fn rewrite_call(
        arena: std.mem.Allocator,
        op: *pr.Op,
        decision_key: []const u8,
        has_side_effect: bool,
    ) std.mem.Allocator.Error!void {
        op.params = .{ .custom_call = .{
            .target_name = try arena.dupe(u8, kernel.dispatch_target_name),
            .has_side_effect = has_side_effect,
            .payload = try arena.dupe(u8, decision_key),
        } };
    }
};

// Kernel Dump Helpers

fn build_ops_str(allocator: std.mem.Allocator, func: pr.Function) ![]const u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    for (func.ops, 0..) |op, i| {
        if (i > 0) try w.writeByte('+');
        try w.writeAll(@tagName(op.prim()));
    }
    return try aw.toOwnedSlice();
}

fn build_shape_str(allocator: std.mem.Allocator, func: pr.Function) ![]const u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    for (func.params, 0..) |in_var, i| {
        if (i > 0) try w.writeByte('x');
        try kernel.write_aval_signature(w, in_var.aval);
    }
    return try aw.toOwnedSlice();
}

fn dump_kernel_entries(out: *std.Io.Writer, entries: []const KernelEntry) !void {
    var compiled: usize = 0;
    var fallback: usize = 0;
    for (entries) |e| switch (e.outcome) {
        .compiled => compiled += 1,
        .fallback => fallback += 1,
    };
    try out.print("kernels: {d} compiled, {d} fallback\n", .{ compiled, fallback });
    try out.print("  {s:<50} {s:<10} {s:<30} {s:<50} {s}\n", .{ "function", "provider", "ops", "shapes", "outcome" });
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
    func: pr.Function,
) !kernel.DecisionKey {
    const function_fingerprint = try fingerprint.function(allocator, func);
    return try kernel.make_decision_key(
        allocator,
        provider_name,
        test_device,
        function_fingerprint,
    );
}

fn outline_test_program(program: *pr.Program) !void {
    var ctx = compilation.Context{ .allocator = std.testing.allocator, .io = std.testing.io };
    _ = try (OutlineCandidates{}).run(program, &ctx);
}

test "OutlineCandidates gives outer provider requests precedence" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();

    const lhs = try builder.param_tensor(.f32, &.{2});
    const rhs = try builder.param_tensor(.f32, &.{2});
    try builder.push_region("outer", &.{kernel.provider_annotation("outer_provider")});
    const sum = try builder.add(lhs, rhs);
    try builder.push_region("inner", &.{kernel.provider_annotation("inner_provider")});
    const product = try builder.multiply(sum, rhs);
    try builder.pop_region();
    try builder.pop_region();
    try program.add_function(try builder.finish(&.{product}));

    try outline_test_program(&program);

    try testing.expectEqual(@as(usize, 2), program.functions.len);
    try testing.expectEqual(pr.Prim.call, program.functions[0].ops[0].prim());
    try testing.expectEqualStrings(
        "outer_provider",
        (try kernel.requested_provider(program.functions[1])).?,
    );
    try testing.expectEqual(@as(usize, 1), program.functions[1].regions.len);
    try testing.expect((try kernel.requested_provider(program.functions[1].regions[0])) == null);
}

test "kernelize pass rewrites a selected function call" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});

    try b.push_region("test_region", &.{kernel.provider_annotation("mock")});
    const y = try b.emit(.{ .exp = {} }, &.{x});
    try b.pop_region();

    const func = try b.finish(&.{y});
    try program.add_function(func);
    try outline_test_program(&program);

    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    const decision_key = try make_test_decision_key(testing.allocator, "mock", program.functions[1]);
    defer testing.allocator.free(decision_key.bytes);
    try store.put_profitable(decision_key, "mock", .{
        .data = "stored_kernel_data",
    }, .copy);

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
    try testing.expectEqualStrings(decision_key.bytes, rewritten.params.custom_call.payload);
    try testing.expect(!rewritten.params.custom_call.has_side_effect);
}

test "kernelize pass preserves observable side effects" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var builder = try pr.FunctionBuilder.init(&program, "test");
    defer builder.deinit();
    const input = try builder.param_tensor(.f32, &.{2});
    try builder.push_region("effectful", &.{kernel.provider_annotation("mock")});
    const outputs = try builder.custom_call(.{
        .target_name = "test.effectful",
        .has_side_effect = true,
    }, &.{input}, &.{input.aval});
    try builder.pop_region();
    try program.add_function(try builder.finish(outputs));
    try outline_test_program(&program);

    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    const decision_key = try make_test_decision_key(testing.allocator, "mock", program.functions[1]);
    defer testing.allocator.free(decision_key.bytes);
    try store.put_profitable(decision_key, "mock", .{ .data = "payload" }, .copy);

    var kernelize = KernelizePass{ .store = &store, .device = test_device };
    var ctx = compilation.Context{ .allocator = testing.allocator, .io = testing.io };
    _ = try kernelize.run(&program, &ctx);

    const rewritten = program.functions[0].ops[0];
    try testing.expectEqual(pr.Prim.custom_call, rewritten.prim());
    try testing.expect(rewritten.params.custom_call.has_side_effect);
}

test "kernelize pass retains a declined function call" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});

    try b.push_region("test_region", &.{kernel.provider_annotation("mock")});
    const y = try b.emit(.{ .exp = {} }, &.{x});
    try b.pop_region();

    const func = try b.finish(&.{y});
    try program.add_function(func);
    try outline_test_program(&program);

    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    const decision_key = try make_test_decision_key(testing.allocator, "mock", program.functions[1]);
    defer testing.allocator.free(decision_key.bytes);
    try store.put_negative(decision_key, "unsupported");

    var kp = KernelizePass{
        .store = &store,
        .device = test_device,
    };

    var ctx = compilation.Context{ .allocator = testing.allocator, .io = std.testing.io };
    _ = try kp.run(&program, &ctx);

    // The fallback remains a normal PR call.
    try testing.expectEqual(@as(usize, 1), program.functions[0].ops.len);
    try testing.expectEqual(pr.Prim.call, program.functions[0].ops[0].prim());
}

test "kernelize pass retains a call without a decision" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});

    try b.push_region("test_region", &.{kernel.provider_annotation("mock")});
    const y = try b.emit(.{ .exp = {} }, &.{x});
    try b.pop_region();

    const func = try b.finish(&.{y});
    try program.add_function(func);
    try outline_test_program(&program);

    // An empty store leaves every call unchanged.
    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();

    var kp = KernelizePass{
        .store = &store,
        .device = test_device,
    };

    var ctx = compilation.Context{ .allocator = testing.allocator, .io = std.testing.io };
    _ = try kp.run(&program, &ctx);

    // The fallback remains a normal PR call.
    try testing.expectEqual(@as(usize, 1), program.functions[0].ops.len);
    try testing.expectEqual(pr.Prim.call, program.functions[0].ops[0].prim());
}

test "kernelize pass does not reuse another provider decision" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var builder = try pr.FunctionBuilder.init(&program, "test");
    defer builder.deinit();
    const input = try builder.param_tensor(.f32, &.{2});
    try builder.push_region("test_region", &.{kernel.provider_annotation("requested")});
    const output = try builder.emit(.{ .exp = {} }, &.{input});
    try builder.pop_region();
    const function = try builder.finish(&.{output});
    try program.add_function(function);
    try outline_test_program(&program);

    const other_key = try make_test_decision_key(testing.allocator, "other", program.functions[1]);
    defer testing.allocator.free(other_key.bytes);
    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    try store.put_profitable(other_key, "other", .{
        .data = "payload",
    }, .copy);

    var kernelize = KernelizePass{
        .store = &store,
        .device = test_device,
    };
    var context = compilation.Context{ .allocator = testing.allocator, .io = testing.io };
    _ = try kernelize.run(&program, &context);

    try testing.expectEqual(pr.Prim.call, program.functions[0].ops[0].prim());
}

test "kernelize pass rewrites a multi-output function call" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});
    const y = try b.param_tensor(.f32, &.{2});

    try b.push_region("multi_out", &.{kernel.provider_annotation("mock")});
    const a = try b.emit(.{ .exp = {} }, &.{x});
    const b_out = try b.emit(.{ .log = {} }, &.{y});
    try b.pop_region();

    const func = try b.finish(&.{ a, b_out });
    try program.add_function(func);
    try outline_test_program(&program);
    const call_outputs = program.functions[0].ops[0].outputs;

    const decision_key = try make_test_decision_key(testing.allocator, "mock", program.functions[1]);
    defer testing.allocator.free(decision_key.bytes);

    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    try store.put_profitable(decision_key, "mock", .{
        .data = "mock_kernel_data",
    }, .copy);

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

    try testing.expect(rewritten.outputs[0] == call_outputs[0]);
    try testing.expect(rewritten.outputs[1] == call_outputs[1]);
}

test "kernelize pass shares decisions for equal functions" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    // Both candidates have the same semantics.
    const x = try b.param_tensor(.f32, &.{2});
    const y = try b.param_tensor(.f32, &.{2});

    try b.push_region("region_a", &.{kernel.provider_annotation("mock")});
    const out_a = try b.emit(.{ .exp = {} }, &.{x});
    try b.pop_region();

    try b.push_region("region_b", &.{kernel.provider_annotation("mock")});
    const out_b = try b.emit(.{ .exp = {} }, &.{y});
    try b.pop_region();

    const func = try b.finish(&.{ out_a, out_b });
    try program.add_function(func);
    try outline_test_program(&program);

    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    const decision_key = try make_test_decision_key(testing.allocator, "mock", program.functions[1]);
    defer testing.allocator.free(decision_key.bytes);
    try store.put_profitable(decision_key, "mock", .{
        .data = "payload",
    }, .copy);

    var kp = KernelizePass{
        .store = &store,
        .device = test_device,
    };

    var ctx = compilation.Context{ .allocator = testing.allocator, .io = std.testing.io };
    _ = try kp.run(&program, &ctx);

    // Both calls use the shared decision.
    const ops = program.functions[0].ops;
    try testing.expectEqual(@as(usize, 2), ops.len);
    try testing.expectEqual(pr.Prim.custom_call, ops[0].prim());
    try testing.expectEqual(pr.Prim.custom_call, ops[1].prim());
}

test "kernelize pass separates functions with different shapes" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    // The different shapes produce distinct kernel signatures.
    const x = try b.param_tensor(.f32, &.{2});
    const y = try b.param_tensor(.f32, &.{4});

    try b.push_region("region_small", &.{kernel.provider_annotation("mock")});
    const out_small = try b.emit(.{ .exp = {} }, &.{x});
    try b.pop_region();

    try b.push_region("region_large", &.{kernel.provider_annotation("mock")});
    const out_large = try b.emit(.{ .exp = {} }, &.{y});
    try b.pop_region();

    const func = try b.finish(&.{ out_small, out_large });
    try program.add_function(func);
    try outline_test_program(&program);

    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    const decision_key = try make_test_decision_key(testing.allocator, "mock", program.functions[1]);
    defer testing.allocator.free(decision_key.bytes);
    try store.put_profitable(decision_key, "mock", .{
        .data = "payload",
    }, .copy);

    var kp = KernelizePass{
        .store = &store,
        .device = test_device,
    };

    var ctx = compilation.Context{ .allocator = testing.allocator, .io = std.testing.io };
    _ = try kp.run(&program, &ctx);

    // Only region_small has a store decision, so region_large remains a call.
    const ops = program.functions[0].ops;
    try testing.expectEqual(@as(usize, 2), ops.len);
    try testing.expectEqual(pr.Prim.custom_call, ops[0].prim());
    try testing.expectEqual(pr.Prim.call, ops[1].prim());
}
