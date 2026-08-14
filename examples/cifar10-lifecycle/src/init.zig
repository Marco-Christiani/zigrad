//! Write the deterministic initial parameter checkpoint.

const std = @import("std");
const zg = @import("zigrad");
const model = @import("model.zig");
const parameters = @import("parameters.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = try init.minimal.args.toSlice(allocator);
    defer allocator.free(args);
    if (args.len != 2) return error.ExpectedOutputPath;

    var params = try parameters.initialize(allocator);
    defer parameters.deinit(&params);
    const checkpoint = try zg.to_safetensors(model.Params, params, allocator);
    defer allocator.free(checkpoint);
    var output = try std.Io.Dir.cwd().createFile(init.io, args[1], .{ .truncate = true });
    defer output.close(init.io);
    try output.writeStreamingAll(init.io, checkpoint);
}
