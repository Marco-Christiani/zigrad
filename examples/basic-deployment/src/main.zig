//! Execute the packaged model through the selected IREE runtime.

const std = @import("std");
const zg = @import("zigrad");
const build_options = @import("build_options");

const vmfb align(16) = @embedFile("model.vmfb").*;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const device_construction: zg.iree.DeviceConstruction = switch (build_options.runtime) {
        .embedded_elf_sync => .embedded_elf_sync,
        .registered => .registered,
    };
    var runtime = try zg.iree.Runtime.init(
        allocator,
        device_construction,
        .{ .driver = build_options.driver },
    );
    defer runtime.deinit();

    var bytecode: zg.iree.Bytecode = .{ .borrowed = vmfb[0..] };
    var executable = try runtime.load(
        allocator,
        &bytecode,
        "module.main",
    );
    defer executable.deinit();

    const lhs = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const rhs = [_]f32{ 7, 8, 9, 10, 11, 12 };
    const scale = [_]f32{ 2, 2, 2, 2 };
    var inputs = [_]zg.iree.Buffer{
        try runtime.create_buffer(std.mem.sliceAsBytes(&lhs), .f32, &.{ 2, 3 }),
        try runtime.create_buffer(std.mem.sliceAsBytes(&rhs), .f32, &.{ 3, 2 }),
        try runtime.create_buffer(std.mem.sliceAsBytes(&scale), .f32, &.{ 2, 2 }),
    };
    defer for (&inputs) |*input| input.deinit();

    var invocation = try executable.invoke(allocator, &inputs);
    defer invocation.deinit();
    if (invocation.outputs.len != 1) return error.UnexpectedOutputCount;

    var output: [4]f32 = undefined;
    try runtime.read_buffer(&invocation.outputs[0], std.mem.sliceAsBytes(&output));
    const expected = [_]f32{ 120, 132, 282, 312 };
    for (output, expected) |actual, wanted|
        if (@abs(actual - wanted) > 1e-4) return error.NumericalMismatch;
}
