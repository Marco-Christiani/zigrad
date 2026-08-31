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

    /// Return the number of operations in the range.
    pub fn len(self: Range) usize {
        return self.end - self.start;
    }

    /// Return whether this range contains `operation_index`.
    pub fn contains(self: Range, operation_index: usize) bool {
        return operation_index >= self.start and operation_index < self.end;
    }
};

/// Predicate used to classify one operation during structural matching.
pub const OpPredicate = *const fn (op: *const pr.Op) bool;

/// Operand ordering used when matching a binary operation.
pub const BinaryOperandOrder = enum {
    /// Operands must appear as `left`, then `right`.
    ordered,
    /// Either operand order is accepted.
    unordered,
};

/// One operand relationship in a rooted dataflow pattern.
pub const GraphInput = union(enum) {
    /// Accept any value at this operand position.
    any,
    /// Require one result of another pattern node.
    node_output: struct {
        /// Earlier node in the pattern.
        node: usize,
        /// Result position produced by `node`.
        result: usize = 0,
    },
};

/// Operation and input relationships for one rooted dataflow-pattern node.
pub const GraphNode = struct {
    /// Local operation constraints.
    operation: Operation,
    /// Operand relationships. An empty slice leaves operands unconstrained.
    inputs: []const GraphInput = &.{},
    /// Operand ordering. Unordered matching requires exactly two inputs.
    input_order: BinaryOperandOrder = .ordered,
};

/// Rooted dataflow pattern with caller-provided capture storage.
///
/// Node references point to earlier nodes, making the pattern acyclic by
///  construction. Matching follows operands from `root` and does not depend on
///  source operation order. The matcher recursively follows use-to-definition
///  edges and backtracks only when trying both orders of a binary node. Graphs
///  contain at most 64 nodes.
pub const Graph = struct {
    /// Pattern nodes in dependency order.
    nodes: []const GraphNode,
    /// Node matched against the supplied root operation.
    root: usize,

    /// Match `root_op` and write one captured operation per pattern node.
    ///
    /// `captures` must have `nodes.len` elements. It is cleared when matching
    ///  fails.
    pub fn match(
        self: Graph,
        /// Operation matched against the root node.
        root_op: *const pr.Op,
        /// Storage for one captured operation per node.
        captures: []?*const pr.Op,
    ) bool {
        if (self.nodes.len == 0 or self.nodes.len > max_graph_nodes or
            self.root >= self.nodes.len or captures.len != self.nodes.len)
        {
            return false;
        }
        @memset(captures, null);
        if (!match_graph_node(self, self.root, root_op, captures)) {
            @memset(captures, null);
            return false;
        }
        for (captures) |captured| if (captured == null) {
            @memset(captures, null);
            return false;
        };
        return true;
    }
};

const max_graph_nodes = 64;

fn match_graph_node(
    graph: Graph,
    node_index: usize,
    op: *const pr.Op,
    captures: []?*const pr.Op,
) bool {
    if (captures[node_index]) |captured| return captured == op;
    const node = graph.nodes[node_index];
    if (!node.operation.matches(op)) return false;
    if (node.inputs.len != 0 and node.inputs.len != op.inputs.len) return false;

    captures[node_index] = op;
    if (node.inputs.len == 0) return true;

    return switch (node.input_order) {
        .ordered => match_graph_inputs(graph, node_index, op, node.inputs, captures),
        .unordered => blk: {
            if (node.inputs.len != 2) break :blk false;
            var saved: [max_graph_nodes]?*const pr.Op = undefined;
            @memcpy(saved[0..graph.nodes.len], captures);
            if (match_graph_inputs(graph, node_index, op, node.inputs, captures))
                break :blk true;

            @memcpy(captures, saved[0..graph.nodes.len]);
            const swapped = [_]GraphInput{ node.inputs[1], node.inputs[0] };
            break :blk match_graph_inputs(graph, node_index, op, &swapped, captures);
        },
    };
}

fn match_graph_inputs(
    graph: Graph,
    owner_index: usize,
    op: *const pr.Op,
    expected: []const GraphInput,
    captures: []?*const pr.Op,
) bool {
    for (op.inputs, expected) |operand, input| switch (input) {
        .any => {},
        .node_output => |reference| {
            if (reference.node >= owner_index) return false;
            const producer = operand.value.defining_op orelse return false;
            if (!match_graph_node(graph, reference.node, producer, captures)) return false;
            if (reference.result >= producer.outputs.len or
                producer.outputs[reference.result] != operand.value) return false;
        },
    };
    return true;
}

/// Return the contiguous range containing exactly the captured operations.
///
/// Returns null when a capture is missing, duplicated, absent from `func`, or
///  separated by an operation outside the match.
pub fn dense_range(
    /// Function expected to contain every captured operation.
    func: pr.Function,
    /// One captured operation per graph node.
    captures: []const ?*const pr.Op,
) ?Range {
    if (captures.len == 0) return null;
    var start = func.ops.len;
    var end: usize = 0;
    for (captures, 0..) |maybe_op, capture_index| {
        const op = maybe_op orelse return null;
        const index = func.op_index_by_id(op.id) orelse return null;
        for (captures[0..capture_index]) |prior| {
            if (prior.? == op) return null;
        }
        start = @min(start, index);
        end = @max(end, index + 1);
    }
    if (end - start != captures.len) return null;
    return .{ .start = start, .end = end };
}

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

    /// Return whether an operation satisfies every specified constraint.
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

/// Matches an exact source-ordered operation sequence.
///
/// Dataflow connectivity is not required. Returns null for an empty sequence,
///  an out-of-bounds range, or an operation mismatch.
pub fn sequence(
    /// Function whose source-ordered operation list is searched.
    func: pr.Function,
    /// Index corresponding to `expected[0]`.
    start: usize,
    /// Nonempty operation-constraint sequence to match.
    expected: []const Operation,
) ?Range {
    if (expected.len == 0 or start > func.ops.len or
        expected.len > func.ops.len - start) return null;
    for (func.ops[start..][0..expected.len], expected) |op, operation| {
        if (!operation.matches(op)) return null;
    }
    return .{ .start = start, .end = start + expected.len };
}

/// Return whether a binary operation consumes two expected values.
pub fn binary_operands(
    /// Operation to inspect. Operations with another arity return false.
    op: *const pr.Op,
    /// Expected left operand in ordered mode.
    left: *const pr.Var,
    /// Expected right operand in ordered mode.
    right: *const pr.Var,
    /// Operand comparison mode.
    order: BinaryOperandOrder,
) bool {
    if (op.inputs.len != 2) return false;
    const actual_left = op.inputs[0].value;
    const actual_right = op.inputs[1].value;
    return switch (order) {
        .ordered => actual_left == left and actual_right == right,
        .unordered => (actual_left == left and actual_right == right) or
            (actual_left == right and actual_right == left),
    };
}

/// Return whether an operation consumes a value defined inside a range.
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
    const mm = try builder.mm(lhs, rhs);
    const sum = try builder.add(mm, bias);
    const output = try builder.exp(sum);
    const func = try builder.finish(.{ .returns = &.{output} });

    const options = ConnectedOptions{
        .accepts = accept_mm_or_pointwise,
        .contains = accept_mm,
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
    const mm = try builder.mm(lhs, rhs);
    const unrelated = try builder.exp(independent);
    const output = try builder.add(mm, unrelated);
    const func = try builder.finish(.{ .returns = &.{output} });

    const matched = connected_range(func, 0, .{
        .accepts = accept_mm_or_pointwise,
        .contains = accept_mm,
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
    const func = try builder.finish(.{ .returns = &.{output} });

    try testing.expectEqual(
        Range{ .start = 0, .end = 2 },
        sequence(func, 0, &.{
            .{ .primitive = .log, .input_count = 1 },
            .{ .primitive = .exp, .output_count = 1 },
        }).?,
    );
    try testing.expect(sequence(func, 0, &.{
        .{ .primitive = .exp },
        .{ .primitive = .log },
    }) == null);
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
    const output = try builder.mm(lhs, rhs);
    const func = try builder.finish(.{ .returns = &.{output} });

    try testing.expect((Operation{
        .primitive = .mm,
        .input_count = 2,
        .output_count = 1,
        .first_output_dtype = .f32,
        .first_output_rank = 2,
    }).matches(func.ops[0]));
    try testing.expect(!(Operation{ .primitive = .add }).matches(func.ops[0]));
}

test "Graph matches branch dataflow independently of source order" {
    const std = @import("std");
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "gated");
    defer builder.deinit();
    const input = try builder.param_tensor(.f16, &.{ 8, 32 });
    const gate_weight = try builder.param_tensor(.f16, &.{ 32, 64 });
    const up_weight = try builder.param_tensor(.f16, &.{ 32, 64 });
    const up = try builder.mm(input, up_weight);
    const gate = try builder.mm(input, gate_weight);
    const activation = try builder.logistic(gate);
    const silu = try builder.multiply(activation, gate);
    const output = try builder.multiply(up, silu);
    const func = try builder.finish(.{ .returns = &.{output} });

    const graph = Graph{
        .nodes = &.{
            .{ .operation = .{ .primitive = .mm, .input_count = 2, .output_count = 1 } },
            .{ .operation = .{ .primitive = .mm, .input_count = 2, .output_count = 1 } },
            .{
                .operation = .{ .primitive = .logistic, .input_count = 1, .output_count = 1 },
                .inputs = &.{.{ .node_output = .{ .node = 0 } }},
            },
            .{
                .operation = .{ .primitive = .multiply, .input_count = 2, .output_count = 1 },
                .inputs = &.{
                    .{ .node_output = .{ .node = 0 } },
                    .{ .node_output = .{ .node = 2 } },
                },
                .input_order = .unordered,
            },
            .{
                .operation = .{ .primitive = .multiply, .input_count = 2, .output_count = 1 },
                .inputs = &.{
                    .{ .node_output = .{ .node = 3 } },
                    .{ .node_output = .{ .node = 1 } },
                },
                .input_order = .unordered,
            },
        },
        .root = 4,
    };
    var captures: [5]?*const pr.Op = undefined;
    try testing.expect(graph.match(output.defining_op.?, &captures));
    try testing.expect(captures[0] == gate.defining_op.?);
    try testing.expect(captures[1] == up.defining_op.?);
    try testing.expectEqual(Range{ .start = 0, .end = 5 }, dense_range(func, &captures).?);
}

test "dense_range rejects interleaved operations" {
    const std = @import("std");
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "interleaved");
    defer builder.deinit();
    const input = try builder.param_tensor(.f32, &.{4});
    const first = try builder.exp(input);
    _ = try builder.log(input);
    const second = try builder.add(first, input);
    const func = try builder.finish(.{ .returns = &.{second} });

    const captures = [_]?*const pr.Op{ first.defining_op.?, second.defining_op.? };
    try testing.expect(dense_range(func, &captures) == null);
}

test "value-use and operand relationships" {
    const std = @import("std");
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const input = try builder.param_tensor(.f32, &.{4});
    const activation = try builder.logistic(input);
    const output = try builder.multiply(input, activation);
    const func = try builder.finish(.{ .returns = &.{output} });

    try testing.expect(activation.only_user() == func.ops[1]);
    try testing.expect(binary_operands(func.ops[1], activation, input, .unordered));
    try testing.expect(!binary_operands(func.ops[1], activation, input, .ordered));
}

fn accept_mm(op: *const pr.Op) bool {
    return op.prim() == .mm;
}

fn accept_mm_or_pointwise(op: *const pr.Op) bool {
    return switch (op.prim()) {
        .mm, .add, .exp => true,
        else => false,
    };
}
