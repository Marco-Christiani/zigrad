const std = @import("std");
const zg = @import("zigrad");

const Pointwise = zg.conv.Pointwise;

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    // use global graph for project
    zg.global_graph_init(allocator, .{
        .eager_teardown = false,
    });
    defer zg.global_graph_deinit();

    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();

    const device = cpu.reference();

    var pw = try Pointwise(f32, .{
        .layout = .NHWC,
        .channels = 3,
        .kernels = 3,
    }).init(device, .normal, .{
        .requires_grad = true,
        .acquired = true,
        .label = "pw-conv",
    });

    const A = try zg.NDTensor(f32).ones(device, &.{ 3, 3, 3 }, .{
        .requires_grad = true,
        .acquired = true,
    });

    const y = try pw.apply(A);

    try y.backward();

    pw.filter.print();
}
