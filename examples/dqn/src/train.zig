const std = @import("std");
const zg = @import("zigrad");
const CartPole = @import("CartPole.zig");
const DQNAgent = @import("agent.zig").DQNAgent;
const T = f32;

fn FileLogger(comptime fmt: []const u8) type {
    return struct {
        const Self = @This();
        file: std.fs.File,

        fn init(filepath: []const u8, header: []const u8) !Self {
            const self = Self{
                .file = try std.fs.createFileAbsolute(filepath, .{ .read = true, .truncate = true }),
            };
            errdefer self.deinit();
            try self.file.writeAll(header);
            return self;
        }

        fn deinit(self: Self) void {
            self.file.close();
        }

        fn log(self: Self, args: anytype) !void {
            errdefer self.deinit();
            try self.file.writer().print(fmt, args);
        }
    };
}

pub fn trainDQN() !void {
    for (0..99) |_| {
        const s = std.crypto.random.int(usize);
        CartPole.runDemoSim(s);
    }
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var im_pool = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer im_pool.deinit();
    const im_alloc = im_pool.allocator();
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // defer _ = gpa.deinit();
    // const allocator = gpa.allocator();
    // const seed = zg.settings.seed;
    const s = std.crypto.random.int(usize);
    var env = CartPole.init(s);
    const max_steps = 200;
    const tau = 0.005;
    // const tau = 0.5;
    var optimizer = zg.optim.Adam(T).init(allocator, 1e-4, 0.9, 0.999, 1e-8);
    // var optimizer = zg.optim.SGD(T){ .lr = 0.001 };
    optimizer.grad_clip_enabled = true;
    // optimizer.grad_clip_opts = .{
    //     // .Norm = .{ .grad_clip_max_norm = 100 },
    //     .Value = .{ .vmin = 0, .vmax = 100 },
    // };
    var agent = try DQNAgent(T, 10_000).init(allocator, .{
        .input_size = 4,
        .hidden_size = 128,
        .output_size = 2,
        .gamma = 0.99,
        .eps_start = 0.9,
        .eps_end = 0.05,
        .eps_decay = 1000,
        // .optimizer = (zg.optim.SGD(T){ .lr = 0.001 }).optimizer(),
        .optimizer = optimizer.optimizer(),
    });
    defer agent.deinit();
    agent.target_net.eval();
    // agent.policy_net.eval();

    const num_episodes = 10_000;
    // const target_update_frequency = 1;
    // const n_updates = 1;
    var total_rewards = try allocator.alloc(T, num_episodes);
    defer allocator.free(total_rewards);

    const dqnlog = try FileLogger("{d},{d:.2},{d:.2},{d:.2}\n").init("/tmp/dqnlog.csv", "episode,total_reward,loss,avg_action\n");
    defer dqnlog.deinit();
    var total_steps: usize = 0;
    for (0..num_episodes) |episode| {
        var state: [4]T = env.reset();
        var total_reward: T = 0;
        var action_sum: T = 0;
        var loss_sum: T = 0;
        var loss_count: T = 0;
        var steps: T = 0;
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

            // if (agent.replay_buffer.isFull()) {
            if (total_steps > 128) {
                agent.policy_net.train(); // shouldnt matter
                loss_sum += try agent.train(im_alloc);
                loss_count += 1;
                try agent.updateTargetNetwork(tau);
                agent.policy_net.eval(); // shouldnt matter
            }
            total_steps += 1;
            _ = im_pool.reset(.retain_capacity);
            // if (total_steps % 10_000 == 0) {
            //     std.debug.print("\n\nSaving model...\n", .{});
            //     try zg.saveModelToFile(T, agent.policy_net.model, "policy_net.json", allocator);
            // }
            if (step_result.done > 0 or steps >= max_steps) break;
        }
        const avg_action: T = action_sum / steps;
        total_rewards[episode] = total_reward;

        std.debug.print("({d:<8}) Episode {d}: Total Reward = {d:.2} Loss Sum: {d:.2} Loss/upd: {d:.2}\n", .{ total_steps, episode + 1, total_reward, loss_sum, loss_sum / loss_count });
        // try dqnlog.log(.{ episode + 1, total_reward, @reduce(.Add, @as(@Vector(n_updates, T), loss_sum)) / n_updates, avg_action });
        if (loss_count > 0) try dqnlog.log(.{ episode + 1, total_reward, loss_sum / loss_count, avg_action });

        if (episode >= 1000) {
            const last_1000_avg = blk: {
                var sum: T = 0;
                for (total_rewards[episode - 999 .. episode + 1]) |r| {
                    sum += r;
                }
                break :blk sum / 1000;
            };
            if (last_1000_avg >= 195) {
                std.debug.print("Solved in {d} episodes\n", .{episode + 1});
                break;
            }
            std.debug.print("last_1000_avg: {d} ", .{last_1000_avg});
        }
    }
}
