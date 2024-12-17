const std = @import("std");
const zg = @import("zigrad");
const CartPole = @import("headless.zig").CartPoleMj;
const DQNAgent = @import("dqn_agent.zig").DQNAgent;
const tb = @import("tensorboard");
const T = f32;

pub fn trainDQN() !void {
    // Use ArenaAllocator for bulk allocations
    var arena = std.heap.ArenaAllocator.init(std.heap.raw_c_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Separate pool for intermediate tensors
    var im_pool = std.heap.ArenaAllocator.init(std.heap.raw_c_allocator);
    defer im_pool.deinit();
    const im_alloc = im_pool.allocator();

    // Initialize environment and logger
    var env = try CartPole.init();
    defer env.deinit();
    var tb_logger = try tb.TensorBoardLogger.init("/tmp/dqn_logs", allocator);
    defer tb_logger.deinit();

    const max_steps = 200;
    const tau = 0.005;

    // Configure optimizer with clipping
    var optimizer = zg.optim.Adam(T).init(allocator, 1e-4, 0.9, 0.999, 1e-8);
    optimizer.grad_clip_enabled = true;

    var agent = try DQNAgent(T, 10_000).init(allocator, .{
        .input_size = 4,
        .hidden_size = 128,
        .output_size = 2,
        .gamma = 0.99,
        .eps_start = 0.9,
        .eps_end = 0.05,
        .eps_decay = 1000,
        .optimizer = optimizer.optimizer(),
    });
    defer agent.deinit();
    agent.target_net.eval();

    const num_episodes = 5_000;
    var total_rewards = try allocator.alloc(T, num_episodes);
    defer allocator.free(total_rewards);

    var total_steps: usize = 0;
    for (0..num_episodes) |episode| {
        var state: [4]T = env.reset();
        // std.debug.print("state: {d}\n", .{state});
        var total_reward: T = 0;
        var action_sum: T = 0;
        var loss_sum: T = 0;
        var loss_count: T = 0;
        var steps: T = 0;

        // Training loop
        while (true) {
            const action = try agent.selectAction(state, total_steps, im_alloc);
            action_sum += @as(T, @floatFromInt(action));
            const step_result = env.step(action);

            agent.storeTransition(.{
                .state = state,
                .action = action,
                .next_state = step_result.state,
                .reward = step_result.reward,
                .done = @as(T, @floatFromInt(step_result.done)),
            });

            steps += 1;
            total_reward += step_result.reward;
            state = step_result.state;

            if (total_steps > 128) {
                agent.policy_net.train();
                const loss = try agent.train(im_alloc, tb_logger);
                loss_sum += loss;
                loss_count += 1;
                try agent.updateTargetNetwork(tau);
                agent.policy_net.eval();

                // Log training metrics to tensorboard
                try tb_logger.addScalar("training/loss", loss, @intCast(total_steps));
                try tb_logger.addScalar("training/epsilon", agent.eps, @intCast(total_steps));
            }

            total_steps += 1;
            _ = im_pool.reset(.retain_capacity);

            if (step_result.done > 0 or steps >= max_steps) break;
        }

        const avg_action: T = action_sum / steps;
        total_rewards[episode] = total_reward;

        // Log episode metrics
        if (loss_count > 0) {
            const avg_loss = loss_sum / loss_count;
            try tb_logger.addScalar("episode/reward", total_reward, @intCast(episode));
            try tb_logger.addScalar("episode/avg_loss", avg_loss, @intCast(episode));
            try tb_logger.addScalar("episode/avg_action", avg_action, @intCast(episode));
        }

        // Track moving average and check for early stopping
        if (episode >= 100) {
            const running_avg = blk: {
                var sum: T = 0;
                for (total_rewards[episode - 100 .. episode]) |r| {
                    sum += r;
                }
                break :blk sum / 100;
            };

            try tb_logger.addScalar("episode/running_avg", running_avg, @intCast(episode));

            if (running_avg >= 195) {
                std.debug.print("Solved in {d} episodes\n", .{episode + 1});
                break;
            }
        }
    }
}
