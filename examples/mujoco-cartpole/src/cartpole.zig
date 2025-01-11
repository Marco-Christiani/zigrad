const std = @import("std");
const c = @cImport({
    @cInclude("mujoco/mujoco.h");
});

const rendering = @import("render_buffer.zig");
const RenderOptions = rendering.RenderOptions;
const RecordingOptions = rendering.RecordingOptions;
const MjRenderer = rendering.MjRenderer;

pub fn main() !void {
    var cartpole = try CartPoleMj.init();
    defer cartpole.deinit();

    const handleInput = struct {
        pub const GLFW_KEY_RIGHT = @as(c_int, 262);
        pub const GLFW_KEY_LEFT = @as(c_int, 263);

        fn callback(renderer: MjRenderer, state: [4]CartPoleMj.T) u32 {
            _ = state;
            if (renderer.getKey(GLFW_KEY_LEFT)) {
                return 0;
            } else if (renderer.getKey(GLFW_KEY_RIGHT)) {
                return 1;
            }
            return 0; // default action
        }
    }.callback;

    const render_opts = RenderOptions{
        .width = 800,
        .height = 600,
        .title = "CartPole Sim",
        .cam_distance = 3.0,
        .cam_azimuth = 90.0,
        .cam_elevation = -20.0,
        .offscreen = false,
    };

    try cartpole.render(
        std.heap.page_allocator,
        handleInput,
        100,
        render_opts,
        .{ .filepath = "cartpole.rgb" },
    );
}

pub const CartPoleMj = struct {
    const Self = @This();
    const T = f32;

    force_mag: f32 = 0.05,
    curr_step: usize = 0,
    model: *c.mjModel,
    data: *c.mjData,
    actuator: usize,

    pub fn init() !Self {
        var err: [1000]u8 = undefined;
        const model = c.mj_loadXML("cartpole.xml", null, &err, @intCast(err.len));
        if (model == null) {
            std.debug.print("Could not load model: {s}\n", .{err});
            return error.LoadModelFailed;
        }

        const data = c.mj_makeData(model);
        if (data == null) {
            std.debug.print("Could not allocate mjData\n", .{});
            return error.AllocateDataFailed;
        }
        return Self{
            .model = model,
            .data = data,
            .actuator = @as(usize, @intCast(c.mj_name2id(model, c.mjOBJ_ACTUATOR, "slide"))),
        };
    }

    pub fn deinit(self: Self) void {
        c.mj_deleteModel(self.model);
        c.mj_deleteData(self.data);
    }

    pub fn reset(self: *Self) [4]T {
        c.mj_resetData(self.model, self.data);
        if (std.crypto.random.boolean()) {
            self.data.*.ctrl[self.actuator] = -self.force_mag;
        } else {
            self.data.*.ctrl[self.actuator] = self.force_mag;
        }
        self.curr_step = 0;
        c.mj_step(self.model, self.data);
        return self.getState();
    }

    pub const StepResult = struct { state: [4]T, reward: T, done: u1 };

    pub fn step(self: *Self, action: u32) StepResult {
        if (action > 0) {
            self.data.*.ctrl[self.actuator] = self.force_mag;
        } else if (action <= 0) {
            self.data.*.ctrl[self.actuator] = -self.force_mag;
        } else {
            // reset force application to 0
            self.data.*.ctrl[self.actuator] = 0;
        }

        c.mj_step(self.model, self.data);
        const pole_angle = self.data.*.qpos[1];
        const fell = pole_angle < -0.2 or pole_angle > 0.2;
        // const fell = pole_angle < -0.5 or pole_angle > 0.5;
        const reward: T = if (fell) 0.0 else 1.0;
        self.curr_step += 1;
        const done = (self.curr_step >= 200) or fell;
        return .{ .state = self.getState(), .reward = reward, .done = @intFromBool(done) };
    }

    fn getState(self: Self) [4]T {
        // HACK: this might now be right, just for testing
        return .{
            @floatCast(self.data.qpos[0]),
            @floatCast(self.data.qpos[1]),
            @floatCast(self.data.qvel[0]),
            @floatCast(self.data.qvel[1]),
        };
    }

    /// Renders a simulation using GLFW and the given action callback.
    /// Optionally on screen and optionally records to disk.
    pub fn render(
        self: *Self,
        allocator: std.mem.Allocator,
        /// Callback function that takes the current state and returns an action.
        action_callback: fn (MjRenderer, [4]T) u32,
        max_steps: usize,
        render_opts: RenderOptions,
        recording_opts: ?RecordingOptions,
    ) !void {
        var renderer = try MjRenderer.init(self.model, render_opts, allocator);
        defer renderer.deinit();

        if (recording_opts) |opts| try renderer.startRecording(opts);
        var i: usize = 0;
        while (!renderer.shouldClose() and i < max_steps) : (i += 1) {
            const current_state = self.getState();
            const action = action_callback(renderer, current_state);
            _ = self.step(action);
            try renderer.render(self.model, self.data);
        }
    }
};
