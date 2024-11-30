const std = @import("std");
pub const zg = @import("zigrad");
const mnist = zg.mnist;
const layer = zg.layer;
pub const std_options = zg.std_options;

pub fn main() !void {
    std.debug.print("{any}\n", .{zg.settings});
    try mnist.main();
}
