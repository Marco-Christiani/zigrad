// To demonstrate basic functionality, we will do the following in Zigrad:
//
// import torch
// from torch import nn
//
// lin = nn.Linear(in_features=2, out_features=1, bias=True, dtype=torch.float32)
// lin.load_state_dict({"weight": torch.tensor([[2, 2]]), "bias": torch.tensor([0])})
// inp = torch.tensor([[3, 3]], requires_grad=True, dtype=torch.float32)
// out = lin(inp)
// out.backward()
// print(out, inp.grad)

const std = @import("std");
const zg = @import("zigrad");

pub fn main() !void {
    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);
    const stdout = bw.writer();

    const T = f32;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer switch (gpa.deinit()) {
        .ok => {},
        .leak => @panic("Leaked"),
    };

    var cpu = zg.device.HostDevice.init(std.heap.raw_c_allocator);
    defer cpu.deinit();
    const device = cpu.reference();

    try stdout.print("Creating layer\n", .{});

    // 2 -> 1
    var layer = try zg.layer.LinearLayer(T).init(device, 2, 1);
    layer.weights.fill(2);

    // 1x2 input
    const input = try zg.NDTensor(T).init(&.{ 3, 3 }, &.{ 1, 2 }, true, device);

    try stdout.print("Forward pass\n", .{});
    const output = try layer.forward(input);
    try output.setup_grad(0);

    var gm = zg.GraphManager(zg.NDTensor(T)).init(gpa.allocator(), .{});
    defer gm.deinit();
    try stdout.print("Backward pass\n", .{});
    try gm.backward(output);

    std.debug.assert(12 == output.get(0));
    try output.print_to_writer(stdout);
    try input.print_to_writer(stdout);

    try stdout.print("Success\n", .{});

    try bw.flush();
}
