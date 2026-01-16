const std = @import("std");
const pr = @import("../pr/pr.zig");

pub fn buildDemoProgram(allocator: std.mem.Allocator) !pr.Program {
    var program = pr.Program.init(allocator);
    errdefer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a_id = try b.paramTensor(.f32, &.{ 2, 3 });
    const b_id = try b.paramTensor(.f32, &.{ 3, 2 });
    const c_id = try b.paramTensor(.f32, &.{ 2, 2 });

    const dot_id = try b.dot(a_id, b_id);
    const add_id = try b.add(dot_id, c_id);
    const out_id = try b.multiply(add_id, c_id);

    const func = try b.finish(&.{out_id});
    try program.addFunction(func);

    return program;
}
