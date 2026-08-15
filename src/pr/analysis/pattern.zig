//! Structural matching over PR functions.
const pr = @import("../pr.zig");

/// Half-open contiguous operation range in one function.
///
/// `start` must not exceed `end`. Functions in this module construct valid
///  ranges.
pub const Range = struct {
    /// Index of the first matched operation.
    start: usize,
    /// Index immediately after the last matched operation.
    end: usize,

    /// Returns the number of operations in the range.
    pub fn len(self: Range) usize {
        return self.end - self.start;
    }

    /// Returns whether this range contains `operation_index`.
    pub fn contains(self: Range, operation_index: usize) bool {
        return operation_index >= self.start and operation_index < self.end;
    }
};

/// Predicate used to classify one operation during structural matching.
pub const OpPredicate = *const fn (op: *const pr.Op) bool;

/// Common local constraints for one operation.
pub const Operation = struct {
    /// Required operation primitive.
    primitive: ?pr.Prim = null,
    /// Required operand count.
    input_count: ?usize = null,
    /// Required result count.
    output_count: ?usize = null,
    /// Required data type of the first result.
    first_output_dtype: ?pr.DType = null,
    /// Required rank of the first result.
    first_output_rank: ?usize = null,

    /// Returns whether an operation satisfies every specified constraint.
    pub fn matches(
        self: Operation,
        /// Borrowed operation to test. This function stores no pointers.
        op: *const pr.Op,
    ) bool {
        if (self.primitive) |primitive| if (op.prim() != primitive) return false;
        if (self.input_count) |count| if (op.inputs.len != count) return false;
        if (self.output_count) |count| if (op.outputs.len != count) return false;
        if (self.first_output_dtype) |dtype| {
            if (op.outputs.len == 0 or op.outputs[0].as_tensor().dtype != dtype) return false;
        }
        if (self.first_output_rank) |rank| {
            if (op.outputs.len == 0 or op.outputs[0].as_tensor().shape.rank() != rank) return false;
        }
        return true;
    }
};

/// Constraints for a connected operation-range match.
pub const ConnectedOptions = struct {
    /// Every operation in the result satisfies this predicate.
    accepts: OpPredicate,
    /// At least one operation in the result satisfies this predicate.
    contains: OpPredicate,
    /// Maximum number of operations in the result.
    max_ops: usize,
};

/// Matches the maximal supported connected range beginning at one operation.
///
/// Every operation after the first consumes a value produced inside the
///  range. A start that continues the preceding supported range is rejected so
///  repeated calls at successive operation indices return each maximal range
///  once. Returns null when the first operation is rejected, the range lacks the
///  required operation, or `start` is outside the function.
pub fn connected_range(
    /// Function whose source-ordered operation list is searched.
    func: pr.Function,
    /// Index of the first operation eligible for the result.
    start: usize,
    /// Operation predicates and maximum result size.
    options: ConnectedOptions,
) ?Range {
    if (options.max_ops == 0 or start >= func.ops.len) return null;
    if (!options.accepts(func.ops[start])) return null;

    if (start > 0 and options.accepts(func.ops[start - 1]) and
        consumes_range(func, func.ops[start], .{ .start = start - 1, .end = start }))
    {
        return null;
    }

    var end = start + 1;
    var contains_required = options.contains(func.ops[start]);
    while (end < func.ops.len and end - start < options.max_ops) : (end += 1) {
        const op = func.ops[end];
        const current = Range{ .start = start, .end = end };
        if (!options.accepts(op) or !consumes_range(func, op, current)) break;
        contains_required = contains_required or options.contains(op);
    }
    if (!contains_required) return null;
    return .{ .start = start, .end = end };
}

/// Matches an exact source-ordered primitive sequence.
///
/// Dataflow connectivity is not required. Returns null for an empty sequence,
///  an out-of-bounds range, or a primitive mismatch.
pub fn sequence(
    /// Function whose source-ordered operation list is searched.
    func: pr.Function,
    /// Index corresponding to `expected[0]`.
    start: usize,
    /// Nonempty primitive sequence to match.
    expected: []const pr.Prim,
) ?Range {
    if (expected.len == 0 or start > func.ops.len or
        expected.len > func.ops.len - start) return null;
    for (func.ops[start..][0..expected.len], expected) |op, primitive| {
        if (op.prim() != primitive) return null;
    }
    return .{ .start = start, .end = start + expected.len };
}

/// Returns whether an operation consumes a value defined inside a range.
pub fn consumes_range(
    /// Function that owns both the operation and range.
    func: pr.Function,
    /// Operation whose operands are inspected.
    op: *const pr.Op,
    /// Half-open operation range containing candidate definitions.
    range: Range,
) bool {
    for (op.inputs) |operand| {
        const defining = operand.value.defining_op orelse continue;
        const defining_index = func.op_index_by_id(defining.id) orelse continue;
        if (range.contains(defining_index)) return true;
    }
    return false;
}

test connected_range {
    const std = @import("std");
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const lhs = try builder.param_tensor(.f32, &.{ 4, 8 });
    const rhs = try builder.param_tensor(.f32, &.{ 8, 2 });
    const bias = try builder.param_tensor(.f32, &.{ 4, 2 });
    const dot = try builder.dot(lhs, rhs);
    const sum = try builder.add(dot, bias);
    const output = try builder.exp(sum);
    const func = try builder.finish(&.{output});

    const options = ConnectedOptions{
        .accepts = accept_dot_or_pointwise,
        .contains = accept_dot,
        .max_ops = 5,
    };
    const matched = connected_range(func, 0, options) orelse
        return error.TestUnexpectedResult;
    try testing.expectEqual(Range{ .start = 0, .end = 3 }, matched);
    try testing.expect(connected_range(func, 1, options) == null);
}

test "connected_range stops before an unrelated supported operation" {
    const std = @import("std");
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const lhs = try builder.param_tensor(.f32, &.{ 4, 8 });
    const rhs = try builder.param_tensor(.f32, &.{ 8, 2 });
    const independent = try builder.param_tensor(.f32, &.{ 4, 2 });
    const dot = try builder.dot(lhs, rhs);
    const unrelated = try builder.exp(independent);
    const output = try builder.add(dot, unrelated);
    const func = try builder.finish(&.{output});

    const matched = connected_range(func, 0, .{
        .accepts = accept_dot_or_pointwise,
        .contains = accept_dot,
        .max_ops = 5,
    }) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(Range{ .start = 0, .end = 1 }, matched);
}

test sequence {
    const std = @import("std");
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const input = try builder.param_tensor(.f32, &.{4});
    const logged = try builder.log(input);
    const output = try builder.exp(logged);
    const func = try builder.finish(&.{output});

    try testing.expectEqual(
        Range{ .start = 0, .end = 2 },
        sequence(func, 0, &.{ .log, .exp }).?,
    );
    try testing.expect(sequence(func, 0, &.{ .exp, .log }) == null);
}

test Operation {
    const std = @import("std");
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const lhs = try builder.param_tensor(.f32, &.{ 4, 8 });
    const rhs = try builder.param_tensor(.f32, &.{ 8, 2 });
    const output = try builder.dot(lhs, rhs);
    const func = try builder.finish(&.{output});

    try testing.expect((Operation{
        .primitive = .dot,
        .input_count = 2,
        .output_count = 1,
        .first_output_dtype = .f32,
        .first_output_rank = 2,
    }).matches(func.ops[0]));
    try testing.expect(!(Operation{ .primitive = .add }).matches(func.ops[0]));
}

fn accept_dot(op: *const pr.Op) bool {
    return op.prim() == .dot;
}

fn accept_dot_or_pointwise(op: *const pr.Op) bool {
    return switch (op.prim()) {
        .dot, .add, .exp => true,
        else => false,
    };
}
