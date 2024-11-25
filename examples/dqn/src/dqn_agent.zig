const std = @import("std");
const zg = @import("zigrad");
const NDTensor = zg.NDTensor;
const ReplayBuffer = @import("replay_buffer.zig").ReplayBuffer;
const DQNModel = @import("dqn_model.zig").DQNModel;
const GraphManager = zg.GraphManager;

pub fn DQNAgent(comptime T: type, buffer_capacity: usize) type {
    return struct {
        const Self = @This();
        var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);

        policy_net: *DQNModel(T),
        target_net: *DQNModel(T),
        replay_buffer: ReplayBuffer(Transition, buffer_capacity),
        eps_start: T,
        eps_end: T,
        eps_decay: T,
        eps: T,

        gamma: T,
        optimizer: zg.optim.Optimizer(T),
        graph_manager: GraphManager(NDTensor(T)),

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
            optimizer: zg.optim.Optimizer(T),
        };

        pub fn init(allocator: std.mem.Allocator, config: DqnConfig) !*Self {
            const self = try allocator.create(Self);

            const policy_net = try DQNModel(T).init(allocator, config.input_size, config.hidden_size, config.output_size);
            const target_net = try DQNModel(T).init(allocator, config.input_size, config.hidden_size, config.output_size);
            const policy_params = policy_net.getParameters();
            const target_params = target_net.getParameters();
            defer policy_net.allocator.free(policy_params);
            defer target_net.allocator.free(target_params);
            for (policy_params, target_params) |policy_param, target_param| {
                @memcpy(target_param.data.data, policy_param.data.data);
            }
            target_net.eval();

            self.* = .{
                .policy_net = policy_net,
                .target_net = target_net,
                .replay_buffer = ReplayBuffer(Transition, buffer_capacity).init(allocator),
                .eps = config.eps_start,
                .eps_start = config.eps_start,
                .eps_end = config.eps_end,
                .eps_decay = config.eps_decay,
                .gamma = config.gamma,
                .optimizer = config.optimizer,
                .graph_manager = GraphManager(NDTensor(T)).init(allocator, .{}),
            };
            return self;
        }

        pub fn deinit(self: *Self) void {
            self.policy_net.deinit();
            self.target_net.deinit();
            self.graph_manager.deinit();
            // FIXME: self or clarify lifetime in docs
        }

        pub fn selectAction(self: *Self, state: [4]T, step: usize, allocator: std.mem.Allocator) !u32 {
            self.eps = self.eps_end + (self.eps_start - self.eps_end) * @exp(-1.0 * @as(T, @floatFromInt(step)) / self.eps_decay);
            if (std.crypto.random.float(T) <= self.eps) {
                return std.crypto.random.intRangeAtMost(u32, 0, 1);
            }

            const state_tensor = try NDTensor(T).init(&state, &[_]usize{ 1, 4 }, false, arena.allocator());
            defer state_tensor.deinit();

            zg.rt_grad_enabled = false;
            const q_values = try self.policy_net.forward(state_tensor, allocator);
            defer q_values.deinit();

            // -------------------------------------------------------------------------------------------------
            // const stacked = try allocator.alloc(T, 8);
            // defer allocator.free(stacked);
            // for (0..stacked.len) |i| stacked[i] = state[i % 4];
            // const state_tensor2 = try NDTensor(T).init(stacked, &[_]usize{ 2, 4 }, false, arena.allocator());
            // defer state_tensor2.deinit();
            //
            // const q_values2 = try self.policy_net.forward(state_tensor2, allocator);
            // defer q_values2.deinit();
            // return if (q_values2.data.data[0] > q_values2.data.data[1]) 0 else 1;
            // -------------------------------------------------------------------------------------------------

            std.debug.assert(try q_values.data.shape.realdims() == 1);
            std.debug.assert(q_values.data.shape.len() == 2);
            return if (q_values.data.data[0] > q_values.data.data[1]) 0 else 1;
        }

        pub fn storeTransition(self: *Self, transition: Transition) void {
            self.replay_buffer.add(transition);
        }

        pub fn updateTargetNetwork(self: *Self, tau: T) !void {
            const policy_params = self.policy_net.params;
            const target_params = self.target_net.params;

            // soft update
            for (policy_params, target_params) |policy_param, target_param| {
                std.debug.assert(policy_param.data.shape.eq(target_param.data.shape.*, .{ .strict = true }));
                for (policy_param.data.data, target_param.data.data) |policy_value, *target_value| {
                    target_value.* = (1 - tau) * target_value.* + tau * policy_value;
                }
            }
        }

        pub fn train(self: *Self, allocator: std.mem.Allocator) !T {
            const bs = 128;
            var batch = try self.replay_buffer.sample2(bs);
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

            const states = try NDTensor(T).init(states_flat, &[_]usize{ bs, 4 }, true, allocator);
            const actions = try NDTensor(usize).init(batch.items(.action), &[_]usize{ bs, 1 }, false, allocator);
            const next_states = try NDTensor(T).init(next_states_flat, &[_]usize{ bs, 4 }, false, allocator);
            const rewards = try NDTensor(T).init(batch.items(.reward), &[_]usize{bs}, false, allocator);
            const dones = try NDTensor(T).init(batch.items(.done), &[_]usize{bs}, false, allocator);

            defer {
                states.deinit();
                actions.deinit();
                next_states.deinit();
                rewards.deinit();
                dones.deinit();
            }

            // compute all target values with gradients disabled
            zg.rt_grad_enabled = false;
            const all_next_q_values = try self.target_net.forward(next_states, allocator);
            defer all_next_q_values.deinit();
            _ = all_next_q_values.setLabel("all_next_q_values");

            // clip Q-values
            const q_max: T = 100.0;
            const q_min: T = -100.0;
            for (all_next_q_values.data.data) |*q| q.* = std.math.clamp(q.*, q_min, q_max);

            const max_next_q_values = try all_next_q_values.maxOverDim(allocator, .{ .dim = 1, .keep_dims = false });
            defer max_next_q_values.deinit();

            const gamma_tensor = try NDTensor(T).init(&[_]T{self.gamma}, &[_]usize{1}, false, allocator);
            defer gamma_tensor.deinit();
            const discounted_max_next_q_values = try max_next_q_values.mul(gamma_tensor, allocator);
            defer discounted_max_next_q_values.deinit();

            const ones = try NDTensor(T).init(&[_]T{1}, &[_]usize{1}, false, allocator);
            defer ones.deinit();
            const dones_complement = try ones.sub(dones, allocator);
            defer dones_complement.deinit();

            const discounted_max_next_q_values_masked = try discounted_max_next_q_values.mul(dones_complement, allocator);
            defer discounted_max_next_q_values_masked.deinit();

            const targets = try rewards.add(discounted_max_next_q_values_masked, allocator);
            defer targets.deinit();
            _ = targets.setLabel("targets");

            // clip Q-targets
            for (targets.data.data) |*t| t.* = std.math.clamp(t.*, q_min, q_max);

            // enable gradients for policy network forward pass and loss computation
            zg.rt_grad_enabled = true;
            const all_q_values = try self.policy_net.forward(states, allocator);
            defer all_q_values.deinit();
            _ = all_q_values.setLabel("all_q_values");

            // clip predicted Q-values
            for (all_q_values.data.data) |*q| q.* = std.math.clamp(q.*, q_min, q_max);

            const q_values = try all_q_values.gather(allocator, .{ .indices = actions, .dim = 1 });
            defer q_values.deinit();
            _ = q_values.setLabel("q_values");

            const loss = try zg.loss.smooth_l1_loss(T, q_values, targets, 1.0, allocator);
            defer loss.deinit();

            self.policy_net.zeroGrad();
            loss.grad.?.fill(1.0);

            try self.graph_manager.backward(loss, allocator);
            try self.optimizer.step(self.policy_net.params);

            zg.rt_grad_enabled = false;
            return loss.get(&.{0});
        }
    };
}
