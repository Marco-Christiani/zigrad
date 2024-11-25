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
            // if (self.random.float(T) < epsilon) {
            if (std.crypto.random.float(T) <= self.eps) {
                return std.crypto.random.intRangeAtMost(u32, 0, 1);
                // return self.random.intRangeAtMost(u32, 0, 1);
            }

            const state_tensor = try NDTensor(T).init(&state, &[_]usize{ 1, 4 }, false, arena.allocator());
            defer state_tensor.deinit();

            zg.rt_grad_enabled = false;
            const q_values = try self.policy_net.forward(state_tensor, allocator);
            defer q_values.deinit();
            // q_values.setLabel("q_values").print();

            // -------------------------------------------------------------------------------------------------
            const stacked = try allocator.alloc(T, 8);
            defer allocator.free(stacked);
            for (0..stacked.len) |i| stacked[i] = state[i % 4];
            const state_tensor2 = try NDTensor(T).init(stacked, &[_]usize{ 2, 4 }, false, arena.allocator());
            defer state_tensor2.deinit();

            const q_values2 = try self.policy_net.forward(state_tensor2, allocator);
            defer q_values2.deinit();
            // q_values2.setLabel("q_values2").print();
            return if (q_values2.data.data[0] > q_values2.data.data[1]) 0 else 1;
            // -------------------------------------------------------------------------------------------------

            // std.debug.assert(try q_values.data.shape.realdims() == 1);
            // std.debug.assert(q_values.data.shape.len() == 2);
            // return if (q_values.data.data[0] > q_values.data.data[1]) 0 else 1;
        }

        pub fn storeTransition(self: *Self, transition: Transition) void {
            self.replay_buffer.add(transition);
        }

        pub fn updateTargetNetwork(self: *Self, tau: T) !void {
            const debug = false;
            if (debug) std.debug.print("{?s}\n", .{"-" ** 80});
            const policy_params = self.policy_net.params;
            const target_params = self.target_net.params;

            // soft update
            for (policy_params, target_params, 0..) |policy_param, target_param, i| {
                std.debug.assert(policy_param.data.shape.eq(target_param.data.shape.*, .{ .strict = true }));
                for (policy_param.data.data, target_param.data.data) |policy_value, *target_value| {
                    target_value.* = (1 - tau) * target_value.* + tau * policy_value;
                }
                if (debug) std.debug.print("\t(post-target-update) [param {d} {?s}] l2 diff {d:.3}\n", .{ i, policy_param.label, policy_param.data.l2_norm() - target_param.data.l2_norm() });
            }
        }

        pub fn train(self: *Self, allocator: std.mem.Allocator) !T {
            const debug = false;
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

            if (debug) {
                states.setLabel("states").print();
                actions.setLabel("actions").print();
                rewards.setLabel("rewards").print();
                dones.setLabel("dones").print();
            }

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

            if (debug) targets.print();

            // enable gradients for policy network forward pass and loss computation
            zg.rt_grad_enabled = true;
            const all_q_values = try self.policy_net.forward(states, allocator);
            defer all_q_values.deinit();
            _ = all_q_values.setLabel("all_q_values");

            // clip predicted Q-values
            for (all_q_values.data.data) |*q| q.* = std.math.clamp(q.*, q_min, q_max);

            if (debug) all_q_values.print();

            const q_values = try all_q_values.gather(allocator, .{ .indices = actions, .dim = 1 });
            defer q_values.deinit();
            _ = q_values.setLabel("q_values");

            if (debug) q_values.print();

            const loss = try zg.loss.smooth_l1_loss(T, q_values, targets, 1.0, allocator);
            defer loss.deinit();

            self.policy_net.zeroGrad();
            loss.grad.?.fill(1.0);

            try self.graph_manager.backward(loss, allocator);
            try self.optimizer.step(self.policy_net.params);

            zg.rt_grad_enabled = false;
            return loss.get(&.{0});
        }

        pub fn train0(self: *Self, allocator: std.mem.Allocator) !T {
            const debug = false;
            if (debug) std.debug.print("\n", .{});
            const bs = 128;
            var batch = try self.replay_buffer.sample2(bs);
            defer batch.deinit(allocator);

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

            if (debug) states.setLabel("states").print();
            if (debug) actions.setLabel("actions").print();
            if (debug) rewards.setLabel("rewards").print();
            if (debug) dones.setLabel("dones").print();

            defer {
                states.deinit();
                actions.deinit();
                next_states.deinit();
                rewards.deinit();
                dones.deinit();
            }

            // enable grad for policy forward pass
            zg.rt_grad_enabled = true;
            const all_q_values = try self.policy_net.forward(states.setLabel("states"), allocator);
            defer all_q_values.deinit();
            _ = all_q_values.setLabel("all_q_values");

            // Clip predicted Q-values
            const q_max: T = 100.0;
            const q_min: T = -100.0;
            for (all_q_values.data.data) |*q| {
                q.* = std.math.clamp(q.*, q_min, q_max);
            }

            if (debug) all_q_values.print();

            // gather q for taken actions
            const q_values = try all_q_values.gather(allocator, .{ .indices = actions, .dim = 1 });
            defer q_values.deinit();
            _ = q_values.setLabel("q_values");

            if (debug) q_values.print();

            // disable grad for target net forward pass
            zg.rt_grad_enabled = false;
            // next q w tgt net
            const all_next_q_values = try self.target_net.forward(next_states, allocator);
            defer all_next_q_values.deinit();

            // Clip target network Q-values
            for (all_next_q_values.data.data) |*q| {
                q.* = std.math.clamp(q.*, q_min, q_max);
            }

            const max_next_q_values = try all_next_q_values.maxOverDim(allocator, .{ .dim = 1, .keep_dims = false });
            defer max_next_q_values.deinit();

            // calc tgts
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

            // Clip final target values
            for (targets.data.data) |*t| {
                t.* = std.math.clamp(t.*, q_min, q_max);
            }

            if (debug) targets.print();

            // enable grad for loss calc
            zg.rt_grad_enabled = true;
            const loss = try zg.loss.smooth_l1_loss(T, q_values, targets, 1.0, allocator);
            defer loss.deinit();
            self.policy_net.zeroGrad();
            loss.grad.?.fill(1.0);

            const params = self.policy_net.params;
            for (params, 0..) |p, i| {
                if (debug) std.debug.print("\t(pre-update) [param {d} {?s}] l2 {d:.3} gradl2 {d:.3}\n", .{ i, p.label, p.data.l2_norm(), p.grad.?.l2_norm() });
            }

            try self.graph_manager.backward(loss, allocator);
            try self.optimizer.step(params);
            for (params, 0..) |p, i| {
                if (debug) std.debug.print("\t(post-update) [param {d} {?s}] l2 {d:.3} gradl2 {d:.3}\n", .{ i, p.label, p.data.l2_norm(), p.grad.?.l2_norm() });
            }
            zg.rt_grad_enabled = false;
            return loss.get(&.{0});
        }
        // Direct straightforward dqn train step
        pub fn train1(self: *Self, allocator: std.mem.Allocator) !T {
            const debug = false;
            const bs = 128;
            var bbatch = try self.replay_buffer.sample(bs);
            defer bbatch.deinit(allocator);
            var batch = bbatch.slice();

            // Flatten the batch states and next_states
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
            _ = states.setLabel("states");
            _ = actions.setLabel("actions");
            _ = next_states.setLabel("next_states");
            _ = rewards.setLabel("rewards");
            _ = dones.setLabel("dones");

            defer {
                states.deinit();
                actions.deinit();
                next_states.deinit();
                rewards.deinit();
                dones.deinit();
            }
            zg.rt_grad_enabled = false;
            const all_next_q_values = try self.target_net.forward(next_states, allocator);
            _ = all_next_q_values.setLabel("all_next_q_values");
            zg.rt_grad_enabled = true;
            const all_q_values = try self.policy_net.forward(states, allocator);
            _ = all_q_values.setLabel("all_q_values");

            defer {
                all_q_values.deinit();
                all_next_q_values.deinit();
            }
            const q_values = try all_q_values.gather(allocator, .{ .indices = actions, .dim = 1 });
            defer q_values.deinit();
            _ = q_values.setLabel("q_values");
            zg.rt_grad_enabled = false;

            var target_q_values = try NDTensor(T).empty(&[_]usize{bs}, false, allocator);
            defer target_q_values.deinit();
            _ = target_q_values.setLabel("target_q_values");

            for (0..bs) |i| {
                const action = actions.data.data[i];
                const reward = rewards.data.data[i];
                const done = dones.data.data[i];
                const next_q_value_0 = all_next_q_values.get(&.{ i, 0 });
                const next_q_value_1 = all_next_q_values.get(&.{ i, 1 });
                const max_next_q = @max(next_q_value_0, next_q_value_1);
                // max_next_q = @max(@min(max_next_q, 100.0), -100.0);

                const target_q: T = if (done == 0) reward + self.gamma * max_next_q else reward;

                // update
                std.debug.assert(action == 0 or action == 1);

                try target_q_values.set(&.{i}, target_q);
                // try q_values.set(&.{i}, all_q_values.get(&.{ i, action }));
                if (debug) std.debug.print("-- Sample {d}: action={d}, reward={d:.4}, done={d}, next_q_values=({d:.4}, {d:.4}), target_q={d:.4}\n", .{
                    i,
                    action,
                    reward,
                    done,
                    next_q_value_0,
                    next_q_value_1,
                    target_q,
                });
            }

            if (debug) std.debug.print("q_values.shape: {d} target_q_values.shape: {d}\n", .{ q_values.data.shape.shape, target_q_values.data.shape.shape });
            // loss
            zg.rt_grad_enabled = true;
            defer zg.rt_grad_enabled = false;
            // const loss = try zg.loss.mse_loss(T, q_values, target_q_values, allocator);
            // const loss = try zg.loss.softmax_cross_entropy_loss(T, q_values, target_q_values, allocator);
            const loss = try zg.loss.smooth_l1_loss(T, q_values, target_q_values, 1.0, allocator);
            defer loss.deinit();
            std.debug.assert(loss.data.size() == 1);
            if (debug) std.debug.print("Loss: {d:.4} ", .{loss.data.data[0]});

            // Debug output
            const should_print = false;
            if (should_print) {
                const s1 = try q_values.data.sum(allocator);
                defer s1.deinit(allocator);
                const s2 = try target_q_values.data.sum(allocator);
                defer s2.deinit(allocator);
                std.debug.print("Q-values: min={d:.4} max={d:.4} mean={d:.4}\n", .{
                    std.mem.min(T, q_values.data.data),
                    std.mem.max(T, q_values.data.data),
                    s1.get(&.{0}) / @as(T, @floatFromInt(q_values.data.data.len)),
                });
                std.debug.print("Target Q-values: min={d:.4} max={d:.4} mean={d:.4}\n", .{
                    std.mem.min(T, target_q_values.data.data),
                    std.mem.max(T, target_q_values.data.data),
                    s2.get(&.{0}) / @as(T, @floatFromInt(target_q_values.data.data.len)),
                });
                std.debug.print("Loss: {d:.4}\n", .{loss.get(&.{0})});
            }
            self.policy_net.zeroGrad();
            loss.grad.?.fill(1.0);

            // back
            try self.graph_manager.backward(loss, allocator);
            const params = self.policy_net.params;
            try self.optimizer.step(params);
            const loss_value = loss.get(&.{0});

            // try zg.utils.renderD2(loss, zg.utils.PrintOptions.plain, allocator, "/tmp/dqngraph.svg");
            // try zg.utils.sesame("/tmp/dqngraph.svg", allocator);

            return loss_value;
        }
    };
}
