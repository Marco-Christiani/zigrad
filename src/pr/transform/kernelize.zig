//! PR kernel-provider substitution.
//!
//! `OutlineCandidates` turns provider regions into callable PR functions.
//! `KernelizePass` queries a populated `KernelStore` and replaces calls when a
//!  selection supplies a provider artifact.
//!
//! Selecting the original candidate leaves the ordinary function call in place.
//!
//! Provider compilation and tuning run after outlining and before substitution.
const std = @import("std");
const compilation = @import("../../compilation.zig");
const device = @import("../../device.zig");
const output_mod = @import("../../output.zig");
const effects = @import("../analysis/effects.zig");
const fingerprint = @import("../analysis/fingerprint.zig");
const outline = @import("outline.zig");
const pr = @import("../pr.zig");
const kernel = @import("../../kernel.zig");

const log = std.log.scoped(.@"zg/kernelize");

const KernelEntryOutcome = enum { provider, original };

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

/// Discovers provider-supported PR ranges and records them as requests.
///
/// Explicit provider regions take precedence. Providers recognizing the same
///  range become candidates in one request. Overlapping discoveries with
///  different boundaries are rejected because the resident selector compares
///  implementations of one callable boundary.
pub const DiscoverCandidates = struct {
    pub const Input = *pr.Program;
    pub const Output = *pr.Program;

    providers: []const kernel.KernelProvider,

    pub fn run(self: DiscoverCandidates, program: Input, ctx: *compilation.Context) !Output {
        for (program.functions) |*func| {
            try discover_function(program.allocator(), ctx.allocator, func, self.providers);
        }
        return program;
    }
};

const DiscoveredRange = struct {
    start: usize,
    op_count: usize,
    providers: std.ArrayList([]const u8) = .empty,
};

fn discover_function(
    arena: std.mem.Allocator,
    scratch: std.mem.Allocator,
    func: *pr.Function,
    providers: []const kernel.KernelProvider,
) !void {
    try kernel.validate_providers(providers);

    var ranges = std.ArrayList(DiscoveredRange).empty;
    defer {
        for (ranges.items) |*range| range.providers.deinit(scratch);
        ranges.deinit(scratch);
    }

    for (providers) |provider| {
        for (0..func.ops.len) |start| {
            const matched = provider.match(func.*, start) orelse continue;
            if (matched.op_count == 0 or start + matched.op_count > func.ops.len) {
                return error.InvalidKernelMatch;
            }
            if (overlaps_explicit_request(func.*, start, matched)) continue;

            var same_range: ?*DiscoveredRange = null;
            for (ranges.items) |*existing| {
                if (existing.start == start and existing.op_count == matched.op_count) {
                    same_range = existing;
                    break;
                }
                if (ranges_overlap(existing.*, start, matched)) return error.OverlappingKernelMatches;
            }

            if (same_range) |existing| {
                try existing.providers.append(scratch, provider.name);
            } else {
                var discovered = DiscoveredRange{
                    .start = start,
                    .op_count = matched.op_count,
                };
                errdefer discovered.providers.deinit(scratch);
                try discovered.providers.append(scratch, provider.name);
                try ranges.append(scratch, discovered);
            }
        }
    }

    if (ranges.items.len == 0) return;
    const existing_len = func.regions.len;
    const updated = try arena.alloc(pr.Region, existing_len + ranges.items.len);
    @memcpy(updated[0..existing_len], func.regions);
    var next_id: u32 = 0;
    for (func.regions) |region| next_id = @max(next_id, region.id + 1);

    for (ranges.items, updated[existing_len..]) |range, *destination| {
        const op_ids = try arena.alloc(u32, range.op_count);
        for (func.ops[range.start..][0..range.op_count], op_ids) |op, *op_id| op_id.* = op.id;
        const provider_names = try arena.alloc([]const u8, range.providers.items.len);
        for (range.providers.items, provider_names) |name, *owned| owned.* = try arena.dupe(u8, name);
        destination.* = .{
            .id = next_id,
            .name = try std.fmt.allocPrint(arena, "kernel_candidate_{d}", .{next_id}),
            .annotations = try pr.dupe_annotations(arena, &.{kernel.providers_annotation(provider_names)}),
            .op_ids = op_ids,
        };
        next_id += 1;
    }
    func.regions = updated;
}

fn overlaps_explicit_request(func: pr.Function, start: usize, matched: kernel.Match) bool {
    for (func.regions) |region| {
        if (region.find_annotation(kernel.provider_annotation_name) == null) continue;
        for (func.ops[start..][0..matched.op_count]) |op| {
            for (region.op_ids) |op_id| if (op.id == op_id) return true;
        }
    }
    return false;
}

fn ranges_overlap(existing: DiscoveredRange, start: usize, matched: kernel.Match) bool {
    const existing_end = existing.start + existing.op_count;
    const matched_end = start + matched.op_count;
    return existing.start < matched_end and start < existing_end;
}

const OutlineRequest = struct {
    function_index: usize,
    region: pr.Region,
};

fn find_outermost_request(program: *const pr.Program) kernel.AnnotationError!?OutlineRequest {
    var found: ?OutlineRequest = null;
    for (program.functions, 0..) |func, function_index| {
        if (try kernel.requested_providers(func) != null) continue;
        for (func.regions) |region| {
            if (try kernel.requested_providers(region) == null) continue;
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

    /// Pre-computed selections populated before this operation runs, borrowed.
    store: *const kernel.KernelStore,

    /// Device used when the selection store was populated.
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
            const provider_request = (try kernel.requested_providers(candidate)) orelse continue;
            const function_fingerprint = try fingerprint.function(temp_allocator, candidate);
            const selection_key = try kernel.make_selection_key(
                temp_allocator,
                provider_request,
                self.device,
                function_fingerprint,
            );
            defer temp_allocator.free(selection_key.bytes);

            const selection = self.store.get(selection_key);
            var selected_provider: ?[]const u8 = null;
            const outcome: KernelEntryOutcome = if (selection) |selected| switch (selected.candidate) {
                .provider => |stored| outcome: {
                    selected_provider = stored.provider_name;
                    try rewrite_call(
                        program.allocator(),
                        op,
                        selection_key.bytes,
                        effects.function_may_have_side_effects(program, candidate),
                    );
                    log.debug("selected provider '{s}' for function '{s}'", .{ stored.provider_name, candidate.name });
                    break :outcome .provider;
                },
                .original => outcome: {
                    log.debug("selected original function '{s}': {s}", .{ candidate.name, selected.reason });
                    break :outcome .original;
                },
            } else .original;

            if (entries) |list| {
                try list.append(entries_alloc, .{
                    .name = candidate.name,
                    .provider = selected_provider orelse try build_providers_str(entries_alloc, provider_request),
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
        selection_key: []const u8,
        has_side_effect: bool,
    ) std.mem.Allocator.Error!void {
        op.params = .{ .custom_call = .{
            .target_name = try arena.dupe(u8, kernel.dispatch_target_name),
            .has_side_effect = has_side_effect,
            .payload = try arena.dupe(u8, selection_key),
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

fn build_providers_str(allocator: std.mem.Allocator, request: kernel.ProviderRequest) ![]const u8 {
    var output: std.Io.Writer.Allocating = .init(allocator);
    errdefer output.deinit();
    for (0..request.len()) |index| {
        if (index > 0) try output.writer.writeByte('|');
        try output.writer.writeAll(request.at(index));
    }
    return try output.toOwnedSlice();
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
    var provider: usize = 0;
    var original: usize = 0;
    for (entries) |e| switch (e.outcome) {
        .provider => provider += 1,
        .original => original += 1,
    };
    try out.print("kernels: {d} provider, {d} original\n", .{ provider, original });
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

fn make_test_selection_key(
    allocator: std.mem.Allocator,
    provider_name: []const u8,
    func: pr.Function,
) !kernel.SelectionKey {
    const function_fingerprint = try fingerprint.function(allocator, func);
    return try kernel.make_selection_key(
        allocator,
        .{ .one = provider_name },
        test_device,
        function_fingerprint,
    );
}

fn select_provider_for_test(
    store: *kernel.KernelStore,
    selection_key: kernel.SelectionKey,
    provider_name: []const u8,
    data: []const u8,
) !void {
    try store.put(selection_key, .{
        .candidate = .{ .provider = .{
            .provider_name = provider_name,
            .artifact = .{ .data = try store.allocator().dupe(u8, data) },
        } },
        .reason = "available",
    });
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
        (try kernel.requested_providers(program.functions[1])).?.at(0),
    );
    try testing.expectEqual(@as(usize, 1), program.functions[1].regions.len);
    try testing.expect((try kernel.requested_providers(program.functions[1].regions[0])) == null);
}

test "DiscoverCandidates groups providers matching the same PR range" {
    const testing = std.testing;
    const Matcher = struct {
        name: []const u8,

        fn provider(self: *@This()) kernel.KernelProvider {
            return .{
                .name = self.name,
                .ptr = @ptrCast(self),
                .compile_fn = undefined,
                .match_fn = match,
            };
        }

        fn match(_: *anyopaque, func: pr.Function, start: usize) ?kernel.Match {
            if (start >= func.ops.len or func.ops[start].prim() != .dot) return null;
            return .{ .op_count = 1 };
        }
    };

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const lhs = try builder.param_tensor(.f32, &.{ 2, 3 });
    const rhs = try builder.param_tensor(.f32, &.{ 3, 2 });
    const result = try builder.dot(lhs, rhs);
    try program.add_function(try builder.finish(&.{result}));

    var first = Matcher{ .name = "first" };
    var second = Matcher{ .name = "second" };
    var providers = [_]kernel.KernelProvider{ first.provider(), second.provider() };
    var ctx = compilation.Context{ .allocator = testing.allocator, .io = testing.io };
    _ = try (DiscoverCandidates{ .providers = &providers }).run(&program, &ctx);

    try testing.expectEqual(@as(usize, 1), program.functions[0].regions.len);
    const request = (try kernel.requested_providers(program.functions[0].regions[0])).?;
    try testing.expectEqual(@as(usize, 2), request.len());
    try testing.expectEqualStrings("first", request.at(0));
    try testing.expectEqualStrings("second", request.at(1));
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
    const selection_key = try make_test_selection_key(testing.allocator, "mock", program.functions[1]);
    defer testing.allocator.free(selection_key.bytes);
    try select_provider_for_test(&store, selection_key, "mock", "stored_kernel_data");

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
    try testing.expectEqualStrings(selection_key.bytes, rewritten.params.custom_call.payload);
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
    const selection_key = try make_test_selection_key(testing.allocator, "mock", program.functions[1]);
    defer testing.allocator.free(selection_key.bytes);
    try select_provider_for_test(&store, selection_key, "mock", "payload");

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
    const selection_key = try make_test_selection_key(testing.allocator, "mock", program.functions[1]);
    defer testing.allocator.free(selection_key.bytes);
    try store.put(selection_key, .{
        .candidate = .original,
        .reason = "unsupported",
    });

    var kp = KernelizePass{
        .store = &store,
        .device = test_device,
    };

    var ctx = compilation.Context{ .allocator = testing.allocator, .io = std.testing.io };
    _ = try kp.run(&program, &ctx);

    // The original candidate remains a normal PR call.
    try testing.expectEqual(@as(usize, 1), program.functions[0].ops.len);
    try testing.expectEqual(pr.Prim.call, program.functions[0].ops[0].prim());
}

test "kernelize pass retains a call without a selection" {
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

    // The original candidate remains a normal PR call.
    try testing.expectEqual(@as(usize, 1), program.functions[0].ops.len);
    try testing.expectEqual(pr.Prim.call, program.functions[0].ops[0].prim());
}

test "kernelize pass does not reuse another provider selection" {
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

    const other_key = try make_test_selection_key(testing.allocator, "other", program.functions[1]);
    defer testing.allocator.free(other_key.bytes);
    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    try select_provider_for_test(&store, other_key, "other", "payload");

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

    const selection_key = try make_test_selection_key(testing.allocator, "mock", program.functions[1]);
    defer testing.allocator.free(selection_key.bytes);

    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    try select_provider_for_test(&store, selection_key, "mock", "mock_kernel_data");

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

test "kernelize pass shares selections for equal functions" {
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
    const selection_key = try make_test_selection_key(testing.allocator, "mock", program.functions[1]);
    defer testing.allocator.free(selection_key.bytes);
    try select_provider_for_test(&store, selection_key, "mock", "payload");

    var kp = KernelizePass{
        .store = &store,
        .device = test_device,
    };

    var ctx = compilation.Context{ .allocator = testing.allocator, .io = std.testing.io };
    _ = try kp.run(&program, &ctx);

    // Both calls use the shared selection.
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
    const selection_key = try make_test_selection_key(testing.allocator, "mock", program.functions[1]);
    defer testing.allocator.free(selection_key.bytes);
    try select_provider_for_test(&store, selection_key, "mock", "payload");

    var kp = KernelizePass{
        .store = &store,
        .device = test_device,
    };

    var ctx = compilation.Context{ .allocator = testing.allocator, .io = std.testing.io };
    _ = try kp.run(&program, &ctx);

    // Only region_small has a store selection, so region_large remains a call.
    const ops = program.functions[0].ops;
    try testing.expectEqual(@as(usize, 2), ops.len);
    try testing.expectEqual(pr.Prim.custom_call, ops[0].prim());
    try testing.expectEqual(pr.Prim.call, ops[1].prim());
}
