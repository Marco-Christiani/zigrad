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

    const x = try zg.NDTensor(f32).empty(cpu.reference(), &.{ 2, 6 }, .{
        .label = "x: f32",
        .graph = &graph,
    });
    defer x.deinit();

    const y = try zg.NDTensor(f64).empty(cpu.reference(), &.{ 2, 2 }, .{
        .label = "y: f64",
        .graph = &graph,
    });
    defer y.deinit();

    var lmap = LayerMap.init(allocator);
    defer lmap.deinit();

    try lmap.put("layer_a.foo.bar.weights", x, .{ .owned = false });
    try lmap.put("layer_a.foo.bar.bias", y, .{ .owned = false });
    try lmap.put("layer_a.foo.baz.weights", x, .{ .owned = false });
    try lmap.put("layer_a.foo.baz.bias", y, .{ .owned = false });

    try lmap.put("layer_b.foo.bar.weights", x, .{ .owned = false });
    try lmap.put("layer_b.foo.bar.bias", y, .{ .owned = false });
    try lmap.put("layer_b.foo.baz.weights", x, .{ .owned = false });
    try lmap.put("layer_b.foo.baz.bias", y, .{ .owned = false });

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

    var counter: struct {
        total_params: usize = 0,
        total_tensors: u32 = 0,
        largest_tensor: usize = 0,
        largest_tensor_key: []const u8 = "",
        pub fn call(self: *@This(), key: []const u8, t: anytype) void {
            const param_count = t.get_size();
            self.total_tensors += 1;
            self.total_params += param_count;

            if (self.largest_tensor < param_count) {
                self.largest_tensor = param_count;
                self.largest_tensor_key = key;
            }
        }
    } = .{};

    lmap.for_all(&counter);
    std.debug.print(
        \\Parameter Statistics:
        \\  Total tensors: {d}
        \\  Largest tensor: {d} at '{s}'
        \\
    , .{
        counter.total_tensors,
        counter.largest_tensor,
        counter.largest_tensor_key,
    });

    lmap.print_tree();
}
