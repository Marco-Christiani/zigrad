const std = @import("std");
const zg = @import("zigrad");

/// Construct the computation shared by the deployment targets.
pub fn build(allocator: std.mem.Allocator) !zg.pr.Program {
    var program = zg.pr.Program.init(allocator);
    errdefer program.deinit();

    var builder = try zg.pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();

    const lhs = try builder.param_tensor(.f32, &.{ 2, 3 });
    const rhs = try builder.param_tensor(.f32, &.{ 3, 2 });
    const scale = try builder.param_tensor(.f32, &.{ 2, 2 });
    const product = try builder.dot(lhs, rhs);
    const shifted = try builder.add(product, scale);
    const result = try builder.multiply(shifted, scale);

    const function = try builder.finish(&.{result});
    try program.add_function(function);
    return program;
}
