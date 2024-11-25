pub fn main() !void {
    // try @import("train.zig").trainDQN();
    try @import("train_tb.zig").trainDQN();
}
