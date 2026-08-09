//! Canonical semantic fingerprints for callable PR functions.

const std = @import("std");
const pr = @import("../pr.zig");
const serialize = @import("../serialize.zig");

const Allocator = std.mem.Allocator;
const Blake3 = std.crypto.hash.Blake3;
const Writer = std.Io.Writer;

/// Version of the canonical function fingerprint encoding.
pub const encoding_version: u32 = 2;

/// Semantic identity of a callable PR function.
pub const Function = struct {
    bytes: [Blake3.digest_length]u8,

    pub fn eql(lhs: Function, rhs: Function) bool {
        return std.mem.eql(u8, &lhs.bytes, &rhs.bytes);
    }

    /// Write the digest as lowercase hexadecimal bytes.
    pub fn write_hex(self: Function, writer: *Writer) Writer.Error!void {
        try writer.print("{x}", .{self.bytes});
    }
};

/// Failures produced while fingerprinting a PR function.
pub const Error = Allocator.Error || serialize.EmitError || error{
    DuplicateValue,
    UnknownValue,
    UnsupportedCall,
};

/// Compute the semantic fingerprint of a callable PR function.
///
/// Names, annotations, regions, and stored IR ids do not affect the result.
/// PR calls are not yet supported because their identity must incorporate the
///  callee semantics instead of its symbol name.
pub fn function(allocator: Allocator, func: pr.Function) Error!Function {
    var buffer: [1024]u8 = undefined;
    var hashing: Writer.Hashing(Blake3) = .init(&buffer);
    const writer = &hashing.writer;

    var value_ids = std.AutoHashMap(*const pr.Var, u32).init(allocator);
    defer value_ids.deinit();

    try writer.writeAll("zigrad.pr.function");
    try writer.writeInt(u32, encoding_version, .little);

    try serialize.write_length(writer, func.params.len);
    var next_value_id: u32 = 0;
    for (func.params) |param| {
        try put_value(&value_ids, param, next_value_id);
        next_value_id += 1;
        try serialize.write_value(pr.Aval, writer, param.aval);
    }

    try serialize.write_length(writer, func.ops.len);
    for (func.ops) |op| {
        if (op.prim() == .call) return error.UnsupportedCall;

        try serialize.write_value(pr.Params, writer, op.params);
        try serialize.write_length(writer, op.inputs.len);
        for (op.inputs) |operand| {
            const value_id = value_ids.get(operand.value) orelse return error.UnknownValue;
            try writer.writeInt(u32, value_id, .little);
        }

        try serialize.write_length(writer, op.outputs.len);
        for (op.outputs) |output| {
            try put_value(&value_ids, output, next_value_id);
            next_value_id += 1;
            try serialize.write_value(pr.Aval, writer, output.aval);
        }
    }

    try serialize.write_length(writer, func.returns.len);
    for (func.returns) |return_var| {
        const value_id = value_ids.get(return_var) orelse return error.UnknownValue;
        try writer.writeInt(u32, value_id, .little);
    }

    try writer.flush();
    var result: Function = undefined;
    hashing.hasher.final(&result.bytes);
    return result;
}

fn put_value(
    value_ids: *std.AutoHashMap(*const pr.Var, u32),
    value: *const pr.Var,
    id: u32,
) Error!void {
    const result = try value_ids.getOrPut(value);
    if (result.found_existing) return error.DuplicateValue;
    result.value_ptr.* = id;
}

test "function ignores names, annotations, and stored ids" {
    var first_program = pr.Program.init(std.testing.allocator);
    defer first_program.deinit();
    var first_builder = try pr.FunctionBuilder.init(&first_program, "first");
    defer first_builder.deinit();

    const first_lhs = try first_builder.param_tensor(.f32, &.{2});
    const first_rhs = try first_builder.param_tensor(.f32, &.{2});
    const first_result = try first_builder.add(first_lhs, first_rhs);
    var first = try first_builder.finish(&.{first_result});
    first.params[0].id = 91;
    first.params[1].id = 17;
    first.ops[0].id = 43;
    first_result.id = 8;

    var second_program = pr.Program.init(std.testing.allocator);
    defer second_program.deinit();
    var second_builder = try pr.FunctionBuilder.init(&second_program, "second");
    defer second_builder.deinit();

    const annotation = pr.Annotation{ .name = "example.hint", .value = .unit };
    const second_lhs = try second_builder.param_tensor(.f32, &.{2});
    const second_rhs = try second_builder.param_tensor(.f32, &.{2});
    const second_result = try second_builder.add(second_lhs, second_rhs);
    var second = try second_builder.finish(&.{second_result});
    second.annotations = &.{annotation};

    const first_fingerprint = try function(std.testing.allocator, first);
    const second_fingerprint = try function(std.testing.allocator, second);
    try std.testing.expect(first_fingerprint.eql(second_fingerprint));
}

test "function includes parameters and literal values" {
    var first_program = pr.Program.init(std.testing.allocator);
    defer first_program.deinit();
    var first_builder = try pr.FunctionBuilder.init(&first_program, "first");
    defer first_builder.deinit();
    const one = try first_builder.literal_scalar(.{ .f32 = 1.0 });
    const first = try first_builder.finish(&.{one});

    var second_program = pr.Program.init(std.testing.allocator);
    defer second_program.deinit();
    var second_builder = try pr.FunctionBuilder.init(&second_program, "second");
    defer second_builder.deinit();
    const two = try second_builder.literal_scalar(.{ .f32 = 2.0 });
    const second = try second_builder.finish(&.{two});

    const first_fingerprint = try function(std.testing.allocator, first);
    const second_fingerprint = try function(std.testing.allocator, second);
    try std.testing.expect(!first_fingerprint.eql(second_fingerprint));
}

test "function includes custom-call payload" {
    var first_program = pr.Program.init(std.testing.allocator);
    defer first_program.deinit();
    var first_builder = try pr.FunctionBuilder.init(&first_program, "first");
    defer first_builder.deinit();
    const first_input = try first_builder.param_tensor(.f32, &.{2});
    const first_outputs = try first_builder.custom_call(.{
        .target_name = "example.dispatch",
        .has_side_effect = false,
        .payload = &.{1},
    }, &.{first_input}, &.{first_input.aval});
    const first = try first_builder.finish(first_outputs);

    var second_program = pr.Program.init(std.testing.allocator);
    defer second_program.deinit();
    var second_builder = try pr.FunctionBuilder.init(&second_program, "second");
    defer second_builder.deinit();
    const second_input = try second_builder.param_tensor(.f32, &.{2});
    const second_outputs = try second_builder.custom_call(.{
        .target_name = "example.dispatch",
        .has_side_effect = false,
        .payload = &.{2},
    }, &.{second_input}, &.{second_input.aval});
    const second = try second_builder.finish(second_outputs);

    const first_fingerprint = try function(std.testing.allocator, first);
    const second_fingerprint = try function(std.testing.allocator, second);
    try std.testing.expect(!first_fingerprint.eql(second_fingerprint));
}

test "function includes graph wiring" {
    var first_program = pr.Program.init(std.testing.allocator);
    defer first_program.deinit();
    var first_builder = try pr.FunctionBuilder.init(&first_program, "first");
    defer first_builder.deinit();
    const first_lhs = try first_builder.param_tensor(.f32, &.{2});
    const first_rhs = try first_builder.param_tensor(.f32, &.{2});
    const first_sum = try first_builder.add(first_lhs, first_rhs);
    const first_product = try first_builder.multiply(first_sum, first_lhs);
    const first = try first_builder.finish(&.{first_product});

    var second_program = pr.Program.init(std.testing.allocator);
    defer second_program.deinit();
    var second_builder = try pr.FunctionBuilder.init(&second_program, "second");
    defer second_builder.deinit();
    const second_lhs = try second_builder.param_tensor(.f32, &.{2});
    const second_rhs = try second_builder.param_tensor(.f32, &.{2});
    const second_sum = try second_builder.add(second_lhs, second_rhs);
    const second_product = try second_builder.multiply(second_sum, second_rhs);
    const second = try second_builder.finish(&.{second_product});

    const first_fingerprint = try function(std.testing.allocator, first);
    const second_fingerprint = try function(std.testing.allocator, second);
    try std.testing.expect(!first_fingerprint.eql(second_fingerprint));
}

test "function includes return order" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "returns");
    defer builder.deinit();
    const lhs = try builder.param_tensor(.f32, &.{2});
    const rhs = try builder.param_tensor(.f32, &.{2});
    const sum = try builder.add(lhs, rhs);
    const product = try builder.multiply(lhs, rhs);
    const forward = try builder.finish(&.{ sum, product });
    var reverse = forward;
    var reversed_returns = [_]*pr.Var{ product, sum };
    reverse.returns = &reversed_returns;

    const forward_fingerprint = try function(std.testing.allocator, forward);
    const reverse_fingerprint = try function(std.testing.allocator, reverse);
    try std.testing.expect(!forward_fingerprint.eql(reverse_fingerprint));
}

test "function rejects unresolved calls" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var callee_builder = try pr.FunctionBuilder.init(&program, "callee");
    defer callee_builder.deinit();
    const callee_input = try callee_builder.param_tensor(.f32, &.{2});
    const callee = try callee_builder.finish(&.{callee_input});
    try program.add_function(callee);

    var builder = try pr.FunctionBuilder.init(&program, "caller");
    defer builder.deinit();
    const input = try builder.param_tensor(.f32, &.{2});
    const result = try builder.call("callee", &.{input});
    const func = try builder.finish(result);

    try std.testing.expectError(error.UnsupportedCall, function(std.testing.allocator, func));
}
