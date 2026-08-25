//! Helpers for attaching compiler annotations to PR operation ranges.
const std = @import("std");
const pattern = @import("../analysis/pattern.zig");
const pr = @import("../pr.zig");

/// Failures while adding an annotated operation range.
pub const Error = pr.BuildError || error{InvalidOperationRange};

/// Append one annotated range to `func` and return its stable region id.
///
/// The program arena owns the copied name, annotations, operation ids, and
///  updated region slice. An empty or out-of-bounds range returns
///  `InvalidOperationRange` without mutating `func`.
pub fn range(
    /// Program arena that owns every allocation referenced by `func`.
    arena: std.mem.Allocator,
    /// Function that receives the appended region.
    func: *pr.Function,
    /// Nonempty in-bounds operation range to annotate.
    matched: pattern.Range,
    /// Region name copied into `arena`.
    name: []const u8,
    /// Region annotations deep-copied into `arena`.
    annotations: []const pr.Annotation,
) Error!u32 {
    if (matched.start >= matched.end or matched.end > func.ops.len)
        return error.InvalidOperationRange;

    var next_id: u32 = 0;
    for (func.regions) |region| next_id = @max(next_id, region.id + 1);

    const op_ids = try arena.alloc(u32, matched.len());
    for (func.ops[matched.start..matched.end], op_ids) |op, *op_id| op_id.* = op.id;

    const updated = try arena.alloc(pr.Region, func.regions.len + 1);
    @memcpy(updated[0..func.regions.len], func.regions);
    updated[func.regions.len] = .{
        .id = next_id,
        .name = try arena.dupe(u8, name),
        .annotations = try pr.dupe_annotations(arena, annotations),
        .op_ids = op_ids,
    };
    func.regions = updated;
    return next_id;
}

test range {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const input = try builder.param_tensor(.f32, &.{4});
    const logged = try builder.log(input);
    const output = try builder.exp(logged);
    var func = try builder.finish(.{ .returns = &.{output} });

    const id = try range(
        program.allocator(),
        &func,
        .{ .start = 0, .end = 2 },
        "transcendentals",
        &.{.{ .name = "test.annotation", .value = .unit }},
    );
    try testing.expectEqual(@as(u32, 0), id);
    try testing.expectEqual(@as(usize, 1), func.regions.len);
    try testing.expectEqualSlices(u32, &.{ 0, 1 }, func.regions[0].op_ids);
    try testing.expect(func.regions[0].find_annotation("test.annotation") != null);
}
