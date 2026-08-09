//! PR region outlining.
const std = @import("std");
const compilation = @import("../compilation.zig");
const pr = @import("pr.zig");
const region_view = @import("region_view.zig");

const Allocator = std.mem.Allocator;

/// Region annotation requesting PR function outlining.
pub const annotation_name = "zigrad.outline";

/// Unit-valued outline request for region builders.
pub const annotation: pr.Annotation = .{
    .name = annotation_name,
    .value = .unit,
};

/// Invalid payloads for the outline annotation contract.
pub const AnnotationError = error{InvalidOutlineAnnotation};

/// Return whether a region requests outlining.
pub fn is_requested(region: pr.Region) AnnotationError!bool {
    const found = region.find_annotation(annotation_name) orelse return false;
    return switch (found.value) {
        .unit => true,
        .boolean => |value| value,
        else => error.InvalidOutlineAnnotation,
    };
}

/// Configuration for outlining one region.
pub const ApplyOptions = struct {
    /// Unique name assigned to the outlined function.
    ///
    /// A null value derives a name from the caller and region id.
    function_name: ?[]const u8 = null,
    /// Annotations attached to the outlined function.
    function_annotations: []const pr.Annotation = &.{},
};

/// Location of the function and call produced by outlining.
pub const ApplyResult = struct {
    /// Index of the appended outlined function.
    function_index: usize,
    /// Operation id of the replacement call in the caller.
    call_op_id: u32,
};

/// Structural and allocation failures produced by region outlining.
pub const ApplyError = pr.BuildError || error{
    FunctionIndexOutOfRange,
    RegionNotFound,
    EmptyRegion,
    UnknownRegionOperation,
    NonContiguousRegion,
    PartialRegionOverlap,
    MissingValueMapping,
};

const OpRange = struct {
    start: usize,
    end: usize,
};

/// Outline one region into a PR function and replace its operations with a call.
///
/// `scratch` backs temporary maps only. The program arena owns the resulting
///  function, call, annotations, and remapped regions.
pub fn apply(
    program: *pr.Program,
    scratch: Allocator,
    function_index: usize,
    region_id: u32,
    opts: ApplyOptions,
) ApplyError!ApplyResult {
    if (function_index >= program.functions.len) return error.FunctionIndexOutOfRange;

    const source = program.functions[function_index];
    const target = find_region(source, region_id) orelse return error.RegionNotFound;
    const generated_name = if (opts.function_name == null)
        try unique_function_name(scratch, program, function_index, target)
    else
        null;
    defer if (generated_name) |name| scratch.free(name);
    const function_name = opts.function_name orelse generated_name.?;
    if (program.get_function(function_name) != null) return error.DuplicateFunctionName;
    const target_range = try region_range(source, target);

    for (source.regions) |region| {
        const candidate_range = try region_range(source, region);
        if (region.id == target.id or ranges_disjoint(target_range, candidate_range)) continue;
        if (range_contains(target_range, candidate_range)) continue;
        if (range_contains(candidate_range, target_range)) continue;
        return error.PartialRegionOverlap;
    }

    const desc = try region_view.describe(scratch, source, target);
    defer desc.deinit(scratch);

    const callee_op_ids = try scratch.alloc(?u32, source.ops.len);
    defer scratch.free(callee_op_ids);
    @memset(callee_op_ids, null);

    var outlined = try build_outlined_function(
        program,
        scratch,
        source,
        desc,
        target_range,
        function_name,
        opts.function_annotations,
        callee_op_ids,
    );
    outlined.regions = try build_outlined_regions(
        program.allocator(),
        source,
        target,
        target_range,
        callee_op_ids,
    );

    const caller_op_ids = try scratch.alloc(?u32, source.ops.len);
    defer scratch.free(caller_op_ids);
    @memset(caller_op_ids, null);

    var caller = try build_caller_function(
        program,
        scratch,
        source,
        desc,
        target_range,
        function_name,
        caller_op_ids,
    );
    const call_op_id = caller_op_ids[target_range.start].?;
    caller.annotations = source.annotations;
    caller.regions = try build_caller_regions(
        program.allocator(),
        source,
        target,
        target_range,
        call_op_id,
        caller_op_ids,
    );

    const functions = try program.allocator().alloc(pr.Function, program.functions.len + 1);
    @memcpy(functions[0..program.functions.len], program.functions);
    functions[function_index] = caller;
    functions[program.functions.len] = outlined;
    const previous_functions = program.functions;
    program.functions = functions;
    pr.validate_program(program) catch |err| {
        program.functions = previous_functions;
        return err;
    };
    return .{
        .function_index = functions.len - 1,
        .call_op_id = call_op_id,
    };
}

/// Compilation operation that consumes explicit outline annotations.
pub const Pass = struct {
    pub const Input = *pr.Program;
    pub const Output = *pr.Program;

    pub fn run(_: Pass, program: Input, ctx: *compilation.Context) !Output {
        while (try find_innermost_request(program)) |request| {
            const remaining = try annotations_without(
                ctx.allocator,
                request.region.annotations,
                annotation_name,
            );
            defer ctx.allocator.free(remaining);

            const function_name = try unique_function_name(
                ctx.allocator,
                program,
                request.function_index,
                request.region,
            );
            defer ctx.allocator.free(function_name);

            _ = try apply(
                program,
                ctx.allocator,
                request.function_index,
                request.region.id,
                .{
                    .function_name = function_name,
                    .function_annotations = remaining,
                },
            );
        }
        return program;
    }
};

const Request = struct {
    function_index: usize,
    region: pr.Region,
};

fn find_innermost_request(program: *const pr.Program) AnnotationError!?Request {
    var found: ?Request = null;
    for (program.functions, 0..) |func, function_index| {
        for (func.regions) |region| {
            if (!try is_requested(region)) continue;
            if (found == null or region.op_ids.len < found.?.region.op_ids.len) {
                found = .{ .function_index = function_index, .region = region };
            }
        }
    }
    return found;
}

fn annotations_without(
    allocator: Allocator,
    annotations: []const pr.Annotation,
    name: []const u8,
) Allocator.Error![]const pr.Annotation {
    var result = try std.ArrayList(pr.Annotation).initCapacity(allocator, annotations.len);
    defer result.deinit(allocator);
    for (annotations) |item| {
        if (!std.mem.eql(u8, item.name, name)) try result.append(allocator, item);
    }
    return try result.toOwnedSlice(allocator);
}

fn unique_function_name(
    allocator: Allocator,
    program: *const pr.Program,
    function_index: usize,
    region: pr.Region,
) Allocator.Error![]u8 {
    const parent_name = program.functions[function_index].name;
    var suffix: usize = 0;
    while (true) : (suffix += 1) {
        const name = if (suffix == 0)
            try std.fmt.allocPrint(allocator, "{s}_outlined_{d}", .{ parent_name, region.id })
        else
            try std.fmt.allocPrint(allocator, "{s}_outlined_{d}_{d}", .{ parent_name, region.id, suffix });
        if (program.get_function(name) == null) return name;
        allocator.free(name);
    }
}

fn find_region(func: pr.Function, region_id: u32) ?pr.Region {
    for (func.regions) |region| {
        if (region.id == region_id) return region;
    }
    return null;
}

fn region_range(func: pr.Function, region: pr.Region) ApplyError!OpRange {
    if (region.op_ids.len == 0) return error.EmptyRegion;
    const start = func.op_index_by_id(region.op_ids[0]) orelse return error.UnknownRegionOperation;
    const end = start + region.op_ids.len;
    if (end > func.ops.len) return error.NonContiguousRegion;
    for (region.op_ids, 0..) |op_id, offset| {
        if (func.ops[start + offset].id != op_id) return error.NonContiguousRegion;
    }
    return .{ .start = start, .end = end };
}

fn ranges_disjoint(lhs: OpRange, rhs: OpRange) bool {
    return lhs.end <= rhs.start or rhs.end <= lhs.start;
}

fn range_contains(outer: OpRange, inner: OpRange) bool {
    return outer.start <= inner.start and outer.end >= inner.end;
}

fn build_outlined_function(
    program: *pr.Program,
    scratch: Allocator,
    source: pr.Function,
    desc: region_view.RegionView,
    target_range: OpRange,
    function_name: []const u8,
    function_annotations: []const pr.Annotation,
    op_ids: []?u32,
) ApplyError!pr.Function {
    var builder = try pr.FunctionBuilder.init(program, function_name);
    defer builder.deinit();

    const values = try scratch.alloc(?*pr.Var, source.var_count);
    defer scratch.free(values);
    @memset(values, null);

    for (desc.inputs) |input| {
        const tensor = input.aval.as_tensor();
        values[input.id] = try builder.param_tensor(tensor.dtype, tensor.shape.dims);
    }

    for (source.ops[target_range.start..target_range.end], target_range.start..) |op, source_index| {
        const outputs = try clone_op(&builder, scratch, op, values);
        op_ids[source_index] = @intCast(source_index - target_range.start);
        for (op.outputs, outputs) |old, new| values[old.id] = new;
    }

    const returns = try mapped_values(scratch, desc.outputs, values);
    defer scratch.free(returns);
    var outlined = try builder.finish(returns);
    outlined.annotations = try pr.dupe_annotations(program.allocator(), function_annotations);
    return outlined;
}

fn build_caller_function(
    program: *pr.Program,
    scratch: Allocator,
    source: pr.Function,
    desc: region_view.RegionView,
    target_range: OpRange,
    function_name: []const u8,
    op_ids: []?u32,
) ApplyError!pr.Function {
    var builder = try pr.FunctionBuilder.init(program, source.name);
    defer builder.deinit();

    const values = try scratch.alloc(?*pr.Var, source.var_count);
    defer scratch.free(values);
    @memset(values, null);

    for (source.params) |param| {
        const tensor = param.aval.as_tensor();
        values[param.id] = try builder.param_tensor(tensor.dtype, tensor.shape.dims);
    }

    var source_index: usize = 0;
    var next_op_id: u32 = 0;
    while (source_index < source.ops.len) {
        if (source_index == target_range.start) {
            const inputs = try mapped_values(scratch, desc.inputs, values);
            defer scratch.free(inputs);
            const out_avals = try avals(scratch, desc.outputs);
            defer scratch.free(out_avals);
            const callee_name = try program.allocator().dupe(u8, function_name);
            const outputs = try builder.emit_outputs(
                .{ .call = .{ .callee = callee_name } },
                inputs,
                out_avals,
            );
            for (desc.outputs, outputs) |old, new| values[old.id] = new;
            op_ids[source_index] = next_op_id;
            next_op_id += 1;
            source_index = target_range.end;
            continue;
        }

        const op = source.ops[source_index];
        const outputs = try clone_op(&builder, scratch, op, values);
        op_ids[source_index] = next_op_id;
        next_op_id += 1;
        for (op.outputs, outputs) |old, new| values[old.id] = new;
        source_index += 1;
    }

    const returns = try mapped_values(scratch, source.returns, values);
    defer scratch.free(returns);
    return try builder.finish(returns);
}

fn clone_op(
    builder: *pr.FunctionBuilder,
    scratch: Allocator,
    op: *const pr.Op,
    values: []const ?*pr.Var,
) ApplyError![]*pr.Var {
    const inputs = try scratch.alloc(*pr.Var, op.inputs.len);
    defer scratch.free(inputs);
    for (op.inputs, 0..) |operand, index| {
        inputs[index] = values[operand.value.id] orelse return error.MissingValueMapping;
    }

    const out_avals = try scratch.alloc(pr.Aval, op.outputs.len);
    defer scratch.free(out_avals);
    for (op.outputs, 0..) |output, index| out_avals[index] = output.aval;
    return try builder.emit_outputs(op.params, inputs, out_avals);
}

fn mapped_values(
    scratch: Allocator,
    source: []const *pr.Var,
    values: []const ?*pr.Var,
) ApplyError![]*pr.Var {
    const mapped = try scratch.alloc(*pr.Var, source.len);
    errdefer scratch.free(mapped);
    for (source, 0..) |value, index| {
        mapped[index] = values[value.id] orelse return error.MissingValueMapping;
    }
    return mapped;
}

fn avals(scratch: Allocator, values: []const *pr.Var) Allocator.Error![]pr.Aval {
    const result = try scratch.alloc(pr.Aval, values.len);
    for (values, 0..) |value, index| result[index] = value.aval;
    return result;
}

fn build_outlined_regions(
    arena: Allocator,
    source: pr.Function,
    target: pr.Region,
    target_range: OpRange,
    op_ids: []const ?u32,
) ApplyError![]const pr.Region {
    var regions = std.ArrayList(pr.Region).empty;
    for (source.regions) |region| {
        if (region.id == target.id) continue;
        const candidate_range = try region_range(source, region);
        if (!range_contains(target_range, candidate_range)) continue;
        try regions.append(arena, try remap_region(arena, source, region, op_ids, null));
    }
    return try regions.toOwnedSlice(arena);
}

fn build_caller_regions(
    arena: Allocator,
    source: pr.Function,
    target: pr.Region,
    target_range: OpRange,
    call_op_id: u32,
    op_ids: []const ?u32,
) ApplyError![]const pr.Region {
    var regions = std.ArrayList(pr.Region).empty;
    for (source.regions) |region| {
        if (region.id == target.id) continue;
        const candidate_range = try region_range(source, region);
        if (range_contains(target_range, candidate_range)) continue;
        const replacement = if (range_contains(candidate_range, target_range)) call_op_id else null;
        try regions.append(arena, try remap_region(arena, source, region, op_ids, replacement));
    }
    return try regions.toOwnedSlice(arena);
}

fn remap_region(
    arena: Allocator,
    source: pr.Function,
    region: pr.Region,
    op_ids: []const ?u32,
    replacement: ?u32,
) ApplyError!pr.Region {
    var mapped = std.ArrayList(u32).empty;
    var wrote_replacement = false;
    for (region.op_ids) |op_id| {
        const index = source.op_index_by_id(op_id) orelse return error.UnknownRegionOperation;
        if (op_ids[index]) |new_id| {
            try mapped.append(arena, new_id);
            if (replacement != null and new_id == replacement.?) wrote_replacement = true;
        } else if (!wrote_replacement) {
            try mapped.append(arena, replacement orelse return error.MissingValueMapping);
            wrote_replacement = true;
        }
    }
    return .{
        .id = region.id,
        .name = region.name,
        .annotations = region.annotations,
        .op_ids = try mapped.toOwnedSlice(arena),
    };
}

test is_requested {
    const requested = pr.Region{
        .id = 0,
        .name = "requested",
        .annotations = &.{annotation},
        .op_ids = &.{},
    };
    try @import("std").testing.expect(try is_requested(requested));
}

test "apply outlines a nested region and preserves surrounding regions" {
    const testing = std.testing;
    const retained = pr.Annotation{
        .name = "example.payload.v1",
        .value = .{ .bytes = &.{ 1, 2, 3 } },
    };

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();

    const x = try builder.param_tensor(.f32, &.{ 2, 2 });
    const y = try builder.param_tensor(.f32, &.{ 2, 2 });
    try builder.push_region("outer", &.{.{ .name = "example.outer", .value = .unit }});
    const sum = try builder.add(x, y);
    try builder.push_region("target", &.{ annotation, retained });
    try builder.push_region("inner", &.{.{ .name = "example.inner", .value = .unit }});
    const product = try builder.multiply(sum, y);
    try builder.pop_region();
    const logged = try builder.log(product);
    try builder.pop_region();
    const result = try builder.add(logged, x);
    try builder.pop_region();

    const main = try builder.finish(&.{result});
    try program.add_function(main);

    const target = for (main.regions) |region| {
        if (std.mem.eql(u8, region.name, "target")) break region;
    } else unreachable;
    const outlined = try apply(
        &program,
        testing.allocator,
        0,
        target.id,
        .{
            .function_name = "main_target",
            .function_annotations = &.{retained},
        },
    );

    try testing.expectEqual(@as(usize, 1), outlined.function_index);
    try testing.expectEqual(@as(u32, 1), outlined.call_op_id);
    try testing.expectEqual(@as(usize, 2), program.functions.len);

    const caller = program.functions[0];
    try testing.expectEqual(@as(usize, 3), caller.ops.len);
    try testing.expectEqual(pr.Prim.call, caller.ops[1].prim());
    try testing.expectEqualStrings("main_target", caller.ops[1].params.call.callee);
    try testing.expectEqual(@as(usize, 2), caller.ops[1].inputs.len);
    try testing.expectEqual(@as(usize, 1), caller.ops[1].outputs.len);
    try testing.expectEqual(@as(usize, 1), caller.regions.len);
    try testing.expectEqualStrings("outer", caller.regions[0].name);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, caller.regions[0].op_ids);

    const callee = program.functions[1];
    try testing.expectEqualStrings("main_target", callee.name);
    try testing.expectEqualSlices(
        u8,
        &.{ 1, 2, 3 },
        callee.find_annotation("example.payload.v1").?.value.bytes,
    );
    try testing.expectEqual(@as(usize, 2), callee.params.len);
    try testing.expectEqual(@as(usize, 2), callee.ops.len);
    try testing.expectEqual(@as(usize, 1), callee.returns.len);
    try testing.expectEqual(@as(usize, 1), callee.regions.len);
    try testing.expectEqualStrings("inner", callee.regions[0].name);
    try testing.expectEqualSlices(u32, &.{0}, callee.regions[0].op_ids);
}

test "Pass consumes outline and transfers independent annotations" {
    const kernel = @import("kernel.zig");
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();

    const lhs = try builder.param_tensor(.f32, &.{ 2, 3 });
    const rhs = try builder.param_tensor(.f32, &.{ 3, 2 });
    try builder.push_region("matmul", &.{
        annotation,
        .{ .name = "example.payload.v1", .value = .{ .bytes = &.{ 4, 5, 6 } } },
        kernel.provider_annotation("tvm"),
    });
    const result = try builder.dot(lhs, rhs);
    try builder.pop_region();
    try program.add_function(try builder.finish(&.{result}));

    var ctx = compilation.Context{ .allocator = testing.allocator, .io = testing.io };
    _ = try (Pass{}).run(&program, &ctx);

    try testing.expectEqual(@as(usize, 2), program.functions.len);
    try testing.expectEqual(@as(usize, 0), program.functions[0].regions.len);
    const callee = program.functions[1];
    try testing.expect(callee.find_annotation(annotation_name) == null);
    try testing.expectEqualSlices(
        u8,
        &.{ 4, 5, 6 },
        callee.find_annotation("example.payload.v1").?.value.bytes,
    );
    try testing.expectEqualStrings(
        "tvm",
        callee.find_annotation(kernel.provider_annotation_name).?.value.as_string().?,
    );
}

test "apply preserves multiple region outputs" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();

    const x = try builder.param_tensor(.f32, &.{2});
    const y = try builder.param_tensor(.f32, &.{2});
    try builder.push_region("pair", &.{annotation});
    const exponential = try builder.exp(x);
    const logarithm = try builder.log(y);
    try builder.pop_region();

    const main = try builder.finish(&.{ exponential, logarithm });
    try program.add_function(main);
    _ = try apply(&program, testing.allocator, 0, main.regions[0].id, .{
        .function_name = "main_pair",
    });

    try testing.expectEqual(@as(usize, 1), program.functions[0].ops.len);
    try testing.expectEqual(@as(usize, 2), program.functions[0].ops[0].outputs.len);
    try testing.expectEqual(@as(usize, 2), program.functions[1].returns.len);
}
