const std = @import("std");
const zg = @import("zigrad");
//const ParamTree = @import("utils/param_tree.zig").ParamTree;
const LayerMap = zg.LayerMap;

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

    const x = try zg.NDTensor(f32).empty(cpu.reference(), &.{ 2, 2 }, .{
        .label = "x: f32",
        .graph = &graph,
    });

    const y = try zg.NDTensor(f64).empty(cpu.reference(), &.{ 2, 2 }, .{
        .label = "y: f64",
        .graph = &graph,
    });

    var lmap = LayerMap.init(allocator);
    defer lmap.deinit();

    try lmap.put("layer.x", x, .{ .owned = true });
    try lmap.put("layer.y", y, .{ .owned = true });

    lmap.for_all(struct { // target every node in the graph (auto-cast)
        pub fn call(_: @This(), key: []const u8, t: anytype) void {
            std.debug.print("LABEL (ALL): {?s}, key: {s}\n", .{ t.get_label(), key });
        }
    }{});

    lmap.for_all_type(struct { // only target NDTensor(f32)
        pub fn call(_: @This(), key: []const u8, t: *zg.NDTensor(f32)) void {
            std.debug.print("LABEL (f32): {?s}, key: {s}\n", .{ t.get_label(), key });
        }
    }{});

    lmap.for_all_type(struct { // only target NDTensor(f64)
        pub fn call(_: @This(), key: []const u8, t: *zg.NDTensor(f64)) void {
            std.debug.print("LABEL (f64): {?s}, key: {s}\n", .{ t.get_label(), key });
        }
    }{});
}
