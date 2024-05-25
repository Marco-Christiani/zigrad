const std = @import("std");
const tensor = @import("tensor.zig");
const NDTensor = tensor.NDTensor;
const Graph = tensor.Graph;
const Add = tensor.Add;
// const zigrad_settings = .{ .gradEnabled = true };

pub fn main() !void {
    std.debug.print("This is a debug message.", .{});

    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);
    const stdout = bw.writer();

    try stdout.print("Running...\n", .{});
    const shape = &[_]u32{3};
    const Tensor = NDTensor(f32, shape);
    const t1 = comptime Tensor.init([_]f32{ 1, 2, 3 });
    const t2 = comptime Tensor.init([_]f32{ 1, 1, 1 });
    const t3 = comptime Tensor.init([_]f32{ 0, 0, 0 });

    const GraphT = Graph(Tensor, 1);
    const addOp = comptime Add(f32, shape);
    const op1 = comptime addOp.init(&t1, &t2);
    var output = op1.output;
    const g = comptime &GraphT{
        .operations = .{op1},
    };
    g.run();
    t3.print();
    output.?.print();

    try stdout.print("Done.\n", .{});

    try bw.flush(); // don't forget to flush!
}

// const zigrad = @import("grad.zig");
// const nn = @import("nn.zig");

// pub fn main() !void {
//     std.debug.print("This is a debug message.", .{});

//     const stdout_file = std.io.getStdOut().writer();
//     var bw = std.io.bufferedWriter(stdout_file);
//     const stdout = bw.writer();

//     try stdout.print("Running...\n", .{});
//     // _ = try zigrad.linearModel(&std.heap.page_allocator, epoch_callback);
//     // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     // const allocator = gpa.allocator();
//     // try nn.trainLayer(&std.heap.page_allocator);
//     try nn.trainLayer(&std.heap.c_allocator);
//     try stdout.print("Done.\n", .{});

//     try bw.flush(); // don't forget to flush!
// }

// fn epoch_callback(value: *const zigrad.Value, epoch_i: usize) anyerror!void {
//     var allocator = std.heap.page_allocator;
//     const graphJson = try zigrad.serializeValueToJson(allocator, value);
//     const filename = try std.fmt.allocPrint(allocator, "outputs/graph_epoch_{}.json", .{epoch_i});
//     defer allocator.free(filename);
//     std.fs.cwd().makeDir("outputs") catch |err| switch (err) {
//         error.PathAlreadyExists => std.debug.print("output/ already exists\n", .{}),
//         else => |e| return e,
//     };

//     const file = try std.fs.cwd().createFile(filename, .{});
//     defer file.close();
//     const fileWriter = file.writer();
//     try std.json.stringify(graphJson, .{}, fileWriter);
// }

// test "simple test" {
//     var list = std.ArrayList(i32).init(std.testing.allocator);
//     defer list.deinit(); // try commenting this out and see if zig detects the memory leak!
//     try list.append(42);
//     try std.testing.expectEqual(@as(i32, 42), list.pop());
// }
