pub fn main() !void {
    // try @import("headless.zig").main();
    try @import("dqn_train.zig").trainDQN();
}
