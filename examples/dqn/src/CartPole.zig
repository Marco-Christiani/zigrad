const std = @import("std");
const math = std.math;
// const zg = @import("zigrad");
const T = f32;
const Self = @This();
const KinematicsIntegrator = enum { Euler, SI };

gravity: T = 9.8,
masscart: T = 1.0,
masspole: T = 0.1,
total_mass: T,
length: T = 0.5,
polemass_length: T,
force_mag: T = 10.0,
tau: T = 0.02,
kinematics_integrator: KinematicsIntegrator = .SI,
theta_threshold_radians: T = 12.0 * 2.0 * math.pi / 360.0,
x_threshold: T = 2.4,

x: T = 0,
x_dot: T = 0,
theta: T = 0,
theta_dot: T = 0,
steps_beyond_done: ?i32 = null,
rng: std.rand.DefaultPrng,

pub fn init(seed: u64) Self {
    var self = Self{
        .total_mass = undefined,
        .polemass_length = undefined,
        .rng = std.Random.DefaultPrng.init(seed),
    };
    self.total_mass = self.masscart + self.masspole;
    self.polemass_length = self.masspole * self.length;
    return self;
}

pub fn reset(self: *Self) [4]T {
    const low = -0.05;
    const high = 0.05;
    self.x = self.rng.random().float(T) * (high - low) + low;
    self.x_dot = self.rng.random().float(T) * (high - low) + low;
    self.theta = self.rng.random().float(T) * (high - low) + low;
    self.theta_dot = self.rng.random().float(T) * (high - low) + low;
    self.steps_beyond_done = null;
    return self.getState();
}

pub const StepResult = struct { state: [4]T, reward: T, done: u1 };

pub fn step(self: *Self, action: u32) StepResult {
    const force = if (action == 1) self.force_mag else -self.force_mag;
    const costheta = @cos(self.theta);
    const sintheta = @sin(self.theta);

    const temp = (force + self.polemass_length * self.theta_dot * self.theta_dot * sintheta) / self.total_mass;
    const thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass));
    const xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass;

    switch (self.kinematics_integrator) {
        .Euler => {
            self.x += self.tau * self.x_dot;
            self.x_dot += self.tau * xacc;
            self.theta += self.tau * self.theta_dot;
            self.theta_dot += self.tau * thetaacc;
        },
        .SI => { // semi-implicit euler
            self.x_dot += self.tau * xacc;
            self.x += self.tau * self.x_dot;
            self.theta_dot += self.tau * thetaacc;
            self.theta += self.tau * self.theta_dot;
        },
    }

    const state = self.getState();
    const terminated: u1 = @intFromBool(self.x < -self.x_threshold or
        self.x > self.x_threshold or
        self.theta < -self.theta_threshold_radians or
        self.theta > self.theta_threshold_radians);

    var reward: T = undefined;
    if (terminated == 0) {
        reward = 1.0;
    } else if (self.steps_beyond_done == null) {
        self.steps_beyond_done = 0;
        reward = 1.0;
    } else {
        if (self.steps_beyond_done.? == 0) {
            std.debug.print("Sim is over but you called step. Further steps are undefined behavior.\n", .{});
        }
        self.steps_beyond_done.? += 1;
        reward = 0.0;
    }

    return .{ .state = state, .reward = reward, .done = terminated };
}

fn getState(self: Self) [4]T {
    return .{ self.x, self.x_dot, self.theta, self.theta_dot };
}

fn pdController(state: [4]T) u32 {
    const theta = state[2]; // pole angle
    const theta_dot = state[3]; // pole vel.

    const Kp = 1.0;
    const Kd = 1.0;

    const control = Kp * theta + Kd * theta_dot;

    return @intFromBool(control >= 0);
}

pub fn runDemoSim(seed: usize) void {
    var env = Self.init(seed);
    var state = env.reset();
    var totalReward: T = 0.0;
    for (200) |_| {
        const action = pdController(state);
        const result = env.step(action);
        state = result.state;
        totalReward += result.reward;
        if (result.done > 0) {
            break;
        }
    }
    std.debug.print("Total reward: {d}\n", .{totalReward});
}
