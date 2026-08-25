const std = @import("std");

const pr = @import("../pr/pr.zig");
const pr_eval = @import("../pr/tests/eval.zig");
const Tensor = @import("../tensor.zig");
const trace = @import("../trace.zig").trace;
const transforms = @import("../transforms.zig");

fn square(x: Tensor) !Tensor {
    return try x.mul(x);
}

fn cube(x: Tensor) !Tensor {
    return try (try x.mul(x)).mul(x);
}

test "second derivative executes through generated calls" {
    const first = comptime transforms.make_grad(square, .{});
    const second = comptime transforms.make_grad(first, .{});
    const specs = .{Tensor.abstract(.f32, &.{})};

    var traced = try trace(second, std.testing.allocator, specs, .{ .name = "second_derivative" });
    defer traced.deinit();
    try pr.validate_program(&traced.program);

    var input = try pr_eval.HostTensor.init_with_data(std.testing.allocator, &.{}, &.{3.0});
    defer input.deinit();
    const entry = traced.program.get_function_by_id(try traced.program.resolve_entry()).?;
    const results = try pr_eval.eval(std.testing.allocator, &traced.program, entry, &.{input});
    defer {
        for (results) |*result| result.deinit();
        std.testing.allocator.free(results);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), results[0].data[0], 1e-6);
}

test "generated gradients compose recursively" {
    const first = comptime transforms.make_grad(cube, .{});
    const second = comptime transforms.make_grad(first, .{});
    const third = comptime transforms.make_grad(second, .{});
    const specs = .{Tensor.abstract(.f32, &.{})};

    var traced = try trace(third, std.testing.allocator, specs, .{ .name = "third_derivative" });
    defer traced.deinit();
    try pr.validate_program(&traced.program);

    var input = try pr_eval.HostTensor.init_with_data(std.testing.allocator, &.{}, &.{3.0});
    defer input.deinit();
    const entry = traced.program.get_function_by_id(try traced.program.resolve_entry()).?;
    const results = try pr_eval.eval(std.testing.allocator, &traced.program, entry, &.{input});
    defer {
        for (results) |*result| result.deinit();
        std.testing.allocator.free(results);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), results[0].data[0], 1e-5);
}
