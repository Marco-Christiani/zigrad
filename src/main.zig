const std = @import("std");

const zg = @import("zigrad");

const Tensor = zg.NDTensor(f32);
const Array = zg.NDArray(f32);

const TestOpts: zg.device.HostDevice.Options = .{
    .max_cache_size = zg.constants.@"1Mb" / 2,
};

pub fn main() !void {
    const T = f32;

    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();

    var graph = zg.Graph.init(std.heap.smp_allocator, .{});
    defer graph.deinit();

    {
        const x = try Tensor.from_slice(cpu.reference(), &.{ -2.0, -0.5, 0.5, 2.0 }, &.{ 2, 2 }, .{
            .requires_grad = true,
            .graph = &graph,
        });

        const y = try x.clamp(-1.0, 1.0);

        try y.backward();
        const expected_output: []const f32 = &.{ -1.0, -0.5, 0.5, 1.0 };
        const expected_grad: []const f32 = &.{ 0.0, 1.0, 1.0, 0.0 };

        try std.testing.expectEqualSlices(T, expected_output, y.get_data());
        try std.testing.expectEqualSlices(T, expected_grad, x.assume_grad_data());

        x.deinit();
        y.deinit();
    }
}
