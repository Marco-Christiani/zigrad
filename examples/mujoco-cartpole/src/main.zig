pub fn main() !void {
    // try @import("dqn_train.zig").trainDQN();
    // try @import("render.zig").main();
    // try @import("render_buffer.zig").main();
    try @import("cartpole.zig").main();
}
