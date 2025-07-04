const std = @import("std");

const zg = @import("zigrad");

const T = f32;
const Tensor = zg.NDTensor(T);
const Array = zg.NDArray(T);

const TestOpts: zg.device.HostDevice.Options = .{
    .max_cache_size = zg.constants.@"1Mb" / 2,
};

pub fn main() !void {
    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();

    {
        var graph = zg.Graph.init(std.heap.smp_allocator, .{});
        defer graph.deinit();
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

    if (zg.has_cuda) {
        var graph = zg.Graph.init(std.heap.smp_allocator, .{});
        defer graph.deinit();
        var gpu = zg.device.CudaDevice.init(0);
        defer gpu.deinit();

        const x = try Tensor.from_slice(gpu.reference(), &.{ -2.0, -0.5, 0.5, 2.0 }, &.{ 2, 2 }, .{
            .requires_grad = true,
            .graph = &graph,
        });
        const x2 = try Tensor.from_slice(gpu.reference(), &.{ -2.0, -0.5, 0.5, 2.0 }, &.{ 2, 2 }, .{
            .requires_grad = true,
            .graph = &graph,
        });

        const y = try x.add(x2);
        // NOTE: add_scalar needs tweaking for cuda
        // const y = try x.add_scalar(1.0);

        // try y.backward();

        x.deinit();
        x2.deinit();
        y.deinit();
    }
}
