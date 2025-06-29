const std = @import("std");

const zg = @import("zigrad");

const Tensor = zg.NDTensor(f32);
const Array = zg.NDArray(f32);

const TestOpts: zg.device.HostDevice.Options = .{
    .max_cache_size = zg.constants.@"1Mb" / 2,
};

pub fn main() !void {
    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();

    var graph = zg.Graph.init(std.heap.smp_allocator, .{});
    defer graph.deinit();

    {
        cpu.capture.open();

        const x = try Tensor.from_slice(cpu.reference(), &.{ -2.0, -0.5, 0.5, 2.0 }, &.{ 2, 2 }, .{
            .requires_grad = true,
            .graph = &graph,
        });
        defer x.deinit();

        const y = try x.clamp(-1.0, 1.0);
        defer y.deinit();

        y.print(); // still all zeroes/undefined

        const seg = cpu.capture.close();
        seg.run();

        y.print(); // now contains results
    }
}
