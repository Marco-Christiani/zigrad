//! Emit the example computation as serialized PR.

const std = @import("std");
const zg = @import("zigrad");
const model = @import("model.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = try init.minimal.args.toSlice(allocator);
    defer allocator.free(args);
    if (args.len != 2) return error.ExpectedOutputPath;

    var traced = try zg.trace(
        model.forward,
        allocator,
        model.input_spec,
        .{},
    );
    defer traced.deinit();

    var file = try std.Io.Dir.cwd().createFile(init.io, args[1], .{ .truncate = true });
    defer file.close(init.io);
    var buffer: [8192]u8 = undefined;
    var writer = file.writer(init.io, &buffer);
    try zg.pr.serialize.emit(&traced.program, &writer.interface);
    try writer.interface.flush();
}
