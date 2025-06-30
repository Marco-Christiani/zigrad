const std = @import("std");
const zg = @import("zigrad");
const ParamTree = @import("utils/param_tree.zig").ParamTree;

pub fn main() !void {
    const stderr = std.io.getStderr();
    const ttyconf = std.io.tty.detectConfig(stderr);

    std.fs.cwd().deleteFile("model.safetensors") catch {};
    std.debug.print("\nDone.\n", .{});
    try ttyconf.setColor(stderr, .reset);
}

test "visitors" {
    const allocator = std.testing.allocator;

    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();

    var graph = zg.Graph.init(allocator, .{});
    defer graph.deinit();

    const device = cpu.reference();
    const tree = try ParamTree(zg.NDTensor(f32)).create(allocator);
    defer tree.deinit();

    const tensor = try zg.NDTensor(f32).from_slice(
        device,
        &[_]f32{ 1.0, 2.0, 3.0, 4.0 },
        &.{ 2, 2 },
        .{ .requires_grad = true, .graph = &graph },
    );
    defer tensor.deinit();

    try tree.put("test", tensor);

    const Counter = struct {
        count: u32 = 0,

        pub fn visit(self: *@This(), path: []const u8, tensor_param: *zg.NDTensor(f32)) !void {
            _ = path;
            _ = tensor_param;
            self.count += 1;
        }
    };

    var counter = Counter{};
    try tree.for_each(&counter);

    try std.testing.expectEqual(@as(u32, 1), counter.count);
}
