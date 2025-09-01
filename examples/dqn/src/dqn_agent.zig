///! NOTE: In the incoming device api it is illegal to directly mutate memory in the manner
/// shown here. There will be small changes to use methods instead of direct access.
const std = @import("std");
const tb = @import("tensorboard");
const zg = @import("zigrad");
const NDTensor = zg.NDTensor;
const ReplayBuffer = @import("replay_buffer.zig").ReplayBuffer;
const DQNModel = @import("dqn_model.zig").DQNModel;
const GraphManager = zg.GraphManager;
const DeviceReference = zg.DeviceReference;

pub fn DQNAgent(
    /// Type
    comptime T: type,
    buffer_capacity: usize,
    /// Number of affine layers
    comptime depth: usize,
) type {
    return struct {
        const Self = @This();

        policy_net: DQNModel(T, depth),
        target_net: DQNModel(T, depth),
        replay_buffer: ReplayBuffer(Transition, buffer_capacity),
        eps_start: T,
        eps_end: T,
        eps_decay: T,
        eps: T,
        gamma: T,
        steps_done: usize,

        const Transition = struct {
            state: [4]T,
            action: usize,
            next_state: [4]T,
            reward: T,
            done: T, // TODO: should be bool but pending masking support, cant be u1 without casting support
        };

        pub const DqnConfig = struct {
            input_size: usize,
            hidden_size: usize,
            output_size: usize,
            gamma: T,
            eps_start: T,
            eps_end: T,
            eps_decay: T,
        };

        pub fn init(device: DeviceReference, config: DqnConfig) !Self {
            var policy_net = try DQNModel(T, depth).init(device, config.input_size, config.hidden_size, config.output_size);
            errdefer policy_net.deinit();
            // initialize target network weights to match policy network
            var target_net = try policy_net.clone();
            errdefer target_net.deinit();

            // Not training the target net
            target_net.set_requires_grad(false);

            return Self{
                .policy_net = policy_net,
                .target_net = target_net,
                .replay_buffer = ReplayBuffer(Transition, buffer_capacity).init(),
                .eps = config.eps_start,
                .eps_start = config.eps_start,
                .eps_end = config.eps_end,
                .eps_decay = config.eps_decay,
                .gamma = config.gamma,
                .steps_done = 0,
            };
        }

        pub fn deinit(self: *Self) void {
            self.policy_net.deinit();
            self.target_net.deinit();
            self.* = undefined;
        }

        pub fn select_action(self: *Self, state: [4]T, step: usize, device: DeviceReference) !u32 {
            self.eps = self.eps_end + (self.eps_start - self.eps_end) * @exp(-1.0 * @as(T, @floatFromInt(step)) / self.eps_decay);

            if (std.crypto.random.float(T) <= self.eps) {
                return std.crypto.random.intRangeAtMost(u32, 0, 1);
            }

            const state_tensor = try NDTensor(T).from_slice(device, &state, &[_]usize{ 1, 4 }, .{});
            defer state_tensor.deinit();

            const og_grad_enabled = zg.runtime.grad_enabled;
            zg.runtime.grad_enabled = false;
            defer zg.runtime.grad_enabled = og_grad_enabled;

            const q_values = try self.policy_net.forward(state_tensor);
            defer q_values.deinit();

            std.debug.assert(q_values.data.shape.realdims() == 1);
            std.debug.assert(q_values.get_shape().len == 2);

            return if (q_values.get(0) > q_values.get(1)) 0 else 1;
        }

        pub fn store_transition(self: *Self, transition: Transition) void {
            self.replay_buffer.add(transition);
        }

        pub fn update_target_network(self: *Self, tau: T) !void {
            try self.target_net.soft_update_from(&self.policy_net, tau);
        }

        pub fn train(
            self: *Self,
            allocator: std.mem.Allocator,
            tb_logger: tb.TensorBoardLogger,
            device: DeviceReference,
        ) !T {
            const bs = 128;
            var batch = try self.replay_buffer.sample2(bs, allocator);
            defer batch.deinit(allocator);

            // Setup input tensors
            var states_flat = try allocator.alloc(T, bs * 4);
            defer allocator.free(states_flat);
            var next_states_flat = try allocator.alloc(T, bs * 4);
            defer allocator.free(next_states_flat);

            for (batch.items(.state), 0..) |state, i| {
                @memcpy(states_flat[i * 4 .. (i + 1) * 4], &state);
            }
            for (batch.items(.next_state), 0..) |next_state, i| {
                @memcpy(next_states_flat[i * 4 .. (i + 1) * 4], &next_state);
            }

            const states = try NDTensor(T).from_slice(device, states_flat, &[_]usize{ bs, 4 }, .{ .requires_grad = true });
            const actions = try NDTensor(usize).from_slice(device, batch.items(.action), &[_]usize{ bs, 1 }, .{});
            const next_states = try NDTensor(T).from_slice(device, next_states_flat, &[_]usize{ bs, 4 }, .{});
            const rewards = try NDTensor(T).from_slice(device, batch.items(.reward), &[_]usize{bs}, .{});
            const dones = try NDTensor(T).from_slice(device, batch.items(.done), &[_]usize{bs}, .{});

            defer {
                states.deinit();
                actions.deinit();
                next_states.deinit();
                rewards.deinit();
                dones.deinit();
            }

            // compute all target values with gradients disabled
            zg.runtime.grad_enabled = false;
            const all_next_q_values = try self.target_net.forward(next_states);
            defer all_next_q_values.deinit();
            all_next_q_values.set_label("all_next_q_values");

            // clip Q-values
            const q_max: T = 100.0;
            const q_min: T = -100.0;
            all_next_q_values._clamp(q_min, q_max);

            const max_next_q_values = try all_next_q_values.max_along(.{ .dim = 1, .keep_dims = false });
            defer max_next_q_values.deinit();

            // compute targets
            const gamma_tensor = try NDTensor(T).from_slice(device, &[_]T{self.gamma}, &[_]usize{1}, .{});
            defer gamma_tensor.deinit();
            const discounted_max_next_q_values = try max_next_q_values.mul(gamma_tensor);
            defer discounted_max_next_q_values.deinit();

            const ones = try NDTensor(T).from_slice(device, &[_]T{1}, &[_]usize{1}, .{});
            defer ones.deinit();
            const dones_complement = try ones.sub(dones);
            defer dones_complement.deinit();

            const discounted_max_next_q_values_masked = try discounted_max_next_q_values.mul(dones_complement);
            defer discounted_max_next_q_values_masked.deinit();

            const targets = try rewards.add(discounted_max_next_q_values_masked);
            defer targets.deinit();
            targets.set_label("targets");

            // clip targets
            targets._clamp(q_min, q_max);

            // enable gradients for policy forward pass and loss
            zg.runtime.grad_enabled = true;
            const all_q_values = try self.policy_net.forward(states);
            defer all_q_values.deinit();
            all_q_values.set_label("all_q_values");

            // clip predicted Q-values
            all_q_values._clamp(q_min, q_max);

            const q_values = try all_q_values.gather(actions.data, 1);
            defer q_values.deinit();
            q_values.set_label("q_values");

            // compute loss
            const loss = try zg.loss.smooth_l1_loss(T, q_values, targets, 1.0);
            defer loss.deinit();

            // log metrics
            // TODO: update for device compat but make it opt-in
            _ = tb_logger;
            // try tb_logger.addHistogram("training/q_values", q_values.data.data, @intCast(self.steps_done));
            // try tb_logger.addHistogram("training/target_values", targets.data.data, @intCast(self.steps_done));
            // try tb_logger.addScalar("training/loss", loss.get(&.{0}), @intCast(self.steps_done));
            // try tb_logger.addScalar("training/epsilon", self.eps, @intCast(self.steps_done));

            // backward pass and optimization
            self.policy_net.zero_grad();
            try loss.backward();

            self.steps_done += 1;
            zg.runtime.grad_enabled = false;
            return loss.get(0); // TODO: item
        }
    };
}
