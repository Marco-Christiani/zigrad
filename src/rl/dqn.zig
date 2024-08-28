const std = @import("std");
const zg = @import("zigrad");
const ReplayBuffer = @import("replay_buffer.zig").ReplayBuffer;
const NDTensor = zg.NDTensor;
const Model = zg.Model;
const LinearLayer = zg.layer.LinearLayer;
const ReLULayer = zg.layer.ReLULayer;
const Trainer = zg.Trainer;
const CartPole = @import("CartPole.zig");
const T = zg.settings.precision;

const SpaceType = enum { cont, discrete };
const ContSpace = struct {
    low: []const T,
    high: []const T,
    shape: []const u32,
};

const DiscreteSpace = struct {
    len: u32,
};

const Space = union(SpaceType) {
    cont: ContSpace,
    discrete: DiscreteSpace,
};

const CartpoleObsSpace = Space{
    .cont = .{
        .low = &.{ -4.8, -1e32 - 1, -0.42, -1e32 - 1 },
        .high = &.{ 4.8, 1e32 - 1, 0.42, 1e32 - 1 },
        .shape = &.{4},
    },
};

const CartpoleActionSpace = Space{
    .discrete = .{ .len = 2 },
};

const Transition = struct {
    state: [4]T,
    action: u32,
    reward: T,
    next_state: [4]T,
    done: bool,
};

const DQN = struct {
    const Self = @This();

    tau: T,
    gamma: T,
    batch_size: u32,
    action_space: Space,
    obs_space: Space,
    q_network: Model(T),
    target_network: Model(T),
    trainer: Trainer(T, .ce),
    replay_buffer: ReplayBuffer(Transition, 1000),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, action_space: Space, obs_space: Space) !*Self {
        std.debug.print("Lets make a DQN", .{});
        const self = try allocator.create(Self);
        self.* = .{
            .tau = 1e-3,
            .gamma = 0.99,
            .batch_size = 64,
            .action_space = action_space,
            .obs_space = obs_space,
            .q_network = try createQNetwork(allocator, obs_space, action_space),
            .target_network = try createQNetwork(allocator, obs_space, action_space),
            .trainer = Trainer(T, .ce).init(self.q_network, 0.001, .{}),
            .replay_buffer = ReplayBuffer(Transition, 1000).init(allocator, zg.random),
            .allocator = allocator,
        };
        try self.updateTargetNetwork();
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.q_network.deinit();
        self.target_network.deinit();
        self.trainer.deinit();
        // self.replay_buffer.deinit();
        self.allocator.destroy(self);
    }

    fn createQNetwork(allocator: std.mem.Allocator, obs_space: Space, action_space: Space) !Model(T) {
        std.debug.print("DQN.createNet\n", .{});
        var model = try Model(T).init(allocator);
        const input_size = obs_space.cont.shape[0];
        const output_size = action_space.discrete.len;

        try model.addLayer((try LinearLayer(T).init(allocator, input_size, 64)).asLayer());
        try model.addLayer((try ReLULayer(T).init(allocator)).asLayer());
        try model.addLayer((try LinearLayer(T).init(allocator, 64, 64)).asLayer());
        try model.addLayer((try ReLULayer(T).init(allocator)).asLayer());
        try model.addLayer((try LinearLayer(T).init(allocator, 64, output_size)).asLayer());

        return model;
    }

    pub fn updateTargetNetwork(self: *Self) !void {
        std.debug.print("DQN.updateTargetNetwork\n", .{});
        const q_params = self.q_network.getParameters();
        const target_params = self.target_network.getParameters();

        for (q_params, target_params) |q_param, target_param| {
            for (q_param.data.data, target_param.data.data) |*q_value, *target_value| {
                target_value.* = (1 - self.tau) * target_value.* + self.tau * q_value.*;
            }
        }
    }

    pub fn trainStep(self: *Self) !*NDTensor(T) {
        var batch = try self.replay_buffer.sample(self.batch_size);
        defer batch.deinit(self.allocator);

        var states_data = try self.allocator.alloc(T, self.batch_size * 4);
        var next_states_data = try self.allocator.alloc(T, self.batch_size * 4);
        var actions_data = try self.allocator.alloc(T, self.batch_size);
        var rewards_data = try self.allocator.alloc(T, self.batch_size);
        var dones_data = try self.allocator.alloc(T, self.batch_size);
        defer self.allocator.free(states_data);
        defer self.allocator.free(next_states_data);
        defer self.allocator.free(actions_data);
        defer self.allocator.free(rewards_data);
        defer self.allocator.free(dones_data);

        for (batch.items(.state), 0..) |state, i| {
            @memcpy(states_data[i * 4 .. (i + 1) * 4], &state);
        }
        for (batch.items(.next_state), 0..) |next_state, i| {
            @memcpy(next_states_data[i * 4 .. (i + 1) * 4], &next_state);
        }
        for (batch.items(.action), 0..) |action, i| {
            actions_data[i] = @floatFromInt(action);
        }
        for (batch.items(.reward), 0..) |reward, i| {
            rewards_data[i] = reward;
        }
        for (batch.items(.done), 0..) |done, i| {
            dones_data[i] = if (done) 1 else 0;
        }

        const states = try NDTensor(T).init(states_data, &[_]usize{ self.batch_size, 4 }, true, self.allocator);
        const actions = try NDTensor(T).init(actions_data, &[_]usize{self.batch_size}, true, self.allocator);
        const rewards = try NDTensor(T).init(rewards_data, &[_]usize{self.batch_size}, true, self.allocator);
        const next_states = try NDTensor(T).init(next_states_data, &[_]usize{ self.batch_size, 4 }, true, self.allocator);
        const dones = try NDTensor(T).init(dones_data, &[_]usize{self.batch_size}, true, self.allocator);
        defer states.deinit();
        defer actions.deinit();
        defer rewards.deinit();
        defer next_states.deinit();
        defer dones.deinit();

        const q_values = try self.q_network.forward(states, self.allocator);
        const next_q_values = try self.target_network.forward(next_states, self.allocator);

        const q_values_actions = try NDTensor(T).init(try self.allocator.alloc(T, self.batch_size), &[_]usize{self.batch_size}, true, self.allocator);
        defer q_values_actions.deinit();
        for (0..self.batch_size) |i| {
            const action_idx: usize = @intFromFloat(actions.data.data[i]);
            q_values_actions.data.data[i] = q_values.data.data[i * self.action_space.discrete.len + action_idx];
        }

        const max_next_q_values = try NDTensor(T).init(try self.allocator.alloc(T, self.batch_size), &[_]usize{self.batch_size}, true, self.allocator);
        defer max_next_q_values.deinit();
        for (0..self.batch_size) |i| {
            var max_q = next_q_values.data.data[i * self.action_space.discrete.len];
            for (1..self.action_space.discrete.len) |j| {
                max_q = @max(max_q, next_q_values.data.data[i * self.action_space.discrete.len + j]);
            }
            max_next_q_values.data.data[i] = max_q;
        }

        const target_q_values = try rewards.add(
            try max_next_q_values.mul(try NDTensor(T).init(&[_]T{self.gamma}, null, true, self.allocator), self.allocator),
            self.allocator,
        );
        std.debug.print("q_values_actions: {d} target_q_values.shape: {d}\n", .{
            q_values_actions.data.shape.shape,
            target_q_values.data.shape.shape,
        });
        const loss = try self.trainer.trainStep(q_values_actions, target_q_values, self.allocator, self.allocator);

        try self.updateTargetNetwork();

        return loss;
    }

    pub fn act(self: *Self, state: *NDTensor(T)) !u32 {
        const q_values = try self.q_network.forward(state, self.allocator);
        var max_q_value: T = q_values.data.data[0];
        var best_action: u32 = 0;

        for (1..self.action_space.discrete.len) |i| {
            if (q_values.data.data[i] > max_q_value) {
                max_q_value = q_values.data.data[i];
                best_action = @intCast(i);
            }
        }

        return best_action;
    }
};
fn resetEnvironment(allocator: std.mem.Allocator, env: *CartPole) !*NDTensor(T) {
    const state = env.reset();
    return try NDTensor(T).init(&state, &[_]usize{4}, true, allocator);
}

fn takeStep(allocator: std.mem.Allocator, env: *CartPole, action: u32) !struct { next_state: *NDTensor(T), reward: T, done: bool } {
    const result = env.step(action);
    const next_state = try NDTensor(T).init(&result.state, &[_]usize{4}, true, allocator);
    return .{ .next_state = next_state, .reward = result.reward, .done = result.done };
}

pub fn train() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    std.debug.print("DQN1\n", .{});
    var env = CartPole.init(42); // Use a fixed seed for reproducibility
    std.debug.print("DQN2\n", .{});
    var dqn = try DQN.init(allocator, CartpoleActionSpace, CartpoleObsSpace);
    defer dqn.deinit();
    std.debug.print("DQN3\n", .{});

    // Training loop
    const num_episodes = 1000;
    const max_steps = 500;

    for (0..num_episodes) |episode| {
        std.debug.print("{d}\n", .{episode});
        var state = try resetEnvironment(allocator, &env);
        // defer state.deinit();
        var total_reward: T = 0;

        for (0..max_steps) |step| {
            const action = try dqn.act(state);
            const step_result = try takeStep(allocator, &env, action);
            // defer step_result.next_state.deinit();
            total_reward += step_result.reward;

            dqn.replay_buffer.add(.{
                .state = state.data.data[0..4].*,
                .action = action,
                .reward = step_result.reward,
                .next_state = step_result.next_state.data.data[0..4].*,
                .done = step_result.done,
            });

            if (dqn.replay_buffer.size >= dqn.batch_size) {
                const loss = try dqn.trainStep();
                std.debug.print("Episode: {}, Step: {}, Loss: {d:.4}\n", .{ episode, step, loss.data.data });
            }

            state = step_result.next_state;

            if (step_result.done) break;
        }

        std.debug.print("Episode: {}, Total Reward: {d:.2}\n", .{ episode, total_reward });
    }
}

pub fn main() !void {
    try @import("dqn_train.zig").trainDQN();
}
