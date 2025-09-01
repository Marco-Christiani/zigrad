///! NOTE: The underlying device abstractions have changed and this example has not yet been migrated
/// please disregard code surround allocators in this example until its migrated, everything else is fine.
///
/// NOTE: the replay buffer needs thought to optimize for device agnosticism + performance. I could write
/// it here, but this might hint at primitives we should provide the user.
const std = @import("std");
const zg = @import("zigrad");
const CartPole = @import("CartPole.zig");
const DQNAgent = @import("dqn_agent.zig").DQNAgent;
const tb = @import("tensorboard");
const T = f32;

pub fn trainDQN() !void {
    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();
    const device = cpu.reference();

    // Zigrad has a global graph that can be overriden for user-provided graphs.
    zg.global_graph_init(std.heap.smp_allocator, .{
        .eager_teardown = true,
    });
    defer zg.global_graph_deinit();

    // TODO: device side replay buffer, rm this allocator
    var im_pool = std.heap.ArenaAllocator.init(std.heap.raw_c_allocator);
    defer im_pool.deinit();
    const im_alloc = im_pool.allocator();

    // For tracking stats
    var arena = std.heap.ArenaAllocator.init(std.heap.raw_c_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Initialize environment and logger (old)
    const s = std.crypto.random.int(usize);
    var env = CartPole.init(s);
    var tb_logger = try tb.TensorBoardLogger.init("/tmp/", allocator);
    defer tb_logger.deinit();

    const max_steps = 200;
    const tau = 0.005;

    // Configure optimizer with clipping
    var optimizer = zg.optim.Adam.init(allocator, .{
        .lr = 1e-4,
        .beta1 = 0.9,
        .beta2 = 0.999,
        .epsilon = 1e-8,
        .grad_clip_enabled = true,
    });

    var agent = try DQNAgent(T, 10_000, 3).init(device, .{
        .input_size = 4,
        .hidden_size = 128,
        .output_size = 2,
        .gamma = 0.99,
        .eps_start = 0.9,
        .eps_end = 0.05,
        .eps_decay = 1000,
    });
    defer agent.deinit();
    try agent.policy_net.attach_optimizer(optimizer.optimizer()); // register policy net params for training

    const num_episodes = 10_000;
    var total_rewards = try allocator.alloc(T, num_episodes);
    defer allocator.free(total_rewards);

    var total_steps: usize = 0;
    for (0..num_episodes) |episode| {
        var state: [4]T = env.reset();
        var total_reward: T = 0;
        var action_sum: T = 0;
        var loss_sum: T = 0;
        var loss_count: T = 0;
        var steps: T = 0;

        // Training loop
        while (true) {
            const action = try agent.select_action(state, total_steps, device);
            action_sum += @as(T, @floatFromInt(action));
            const step_result = env.step(action);

            agent.store_transition(.{
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
                agent.policy_net.set_requires_grad(true);
                const loss = try agent.train(im_alloc, tb_logger, device);
                try optimizer.optimizer().step();
                loss_sum += loss;
                loss_count += 1;
                try agent.update_target_network(tau);
                agent.policy_net.set_requires_grad(false);

                // Log training metrics to tensorboard
                try tb_logger.addScalar("training/loss", loss, @intCast(total_steps));
                try tb_logger.addScalar("training/epsilon", agent.eps, @intCast(total_steps));
            }

            total_steps += 1;
            _ = im_pool.reset(.retain_capacity); // (old)

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
