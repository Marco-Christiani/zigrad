const std = @import("std");
// const scalar = @import("scalar/grad.zig");
// const scalarnn = @import("scalar/nn.zig");
// const layer = @import("tensor/layer.zig");
// const mnist = @import("tensor/mnist.zig");
pub const zg = @import("zigrad");

pub fn main() !void {
    std.debug.print("{any}\n", .{zg.settings});
    // try runScalar();
    // try mnist.main();
}
//
// fn runScalar() !void {
//     // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     // const allocator = gpa.allocator();
//     // try scalarnn.trainLayer(&std.heap.page_allocator);
//     // try scalarnn.trainLayer(&std.heap.c_allocator);
//     _ = try scalar.linearModel(&std.heap.page_allocator, epoch_callback);
// }
//
// fn epoch_callback(value: *const scalar.Value, epoch_i: usize) anyerror!void {
//     const allocator = std.heap.page_allocator;
//     std.fs.cwd().makeDir("outputs-2") catch |err| switch (err) {
//         error.PathAlreadyExists => std.debug.print("output/ already exists\n", .{}),
//         else => |e| return e,
//     };
//     const filename = try std.fmt.allocPrint(allocator, "outputs-2/graph_epoch_{}.json", .{epoch_i});
//     defer allocator.free(filename);
//
//     const file = try std.fs.cwd().createFile(filename, .{});
//     defer file.close();
//     const graphJson = try scalar.serializeValueToJson(allocator, value);
//     try std.json.stringify(graphJson, .{}, file.writer());
//
//     const d2filename = try std.fmt.allocPrint(allocator, "outputs-2/graph_epoch_{}.d2", .{epoch_i});
//     const d2file = try std.fs.cwd().createFile(d2filename, .{});
//     defer d2file.close();
//     value.renderD2(d2file.writer());
// }
