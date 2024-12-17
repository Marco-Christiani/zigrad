const std = @import("std");
const c = @cImport({
    @cInclude("mujoco/mujoco.h");
});

pub fn main() !void {
    // var cartpole = try CartPoleMj.init();
    // _ = cartpole.reset();
    // for (0..50) |_| {
    //     const out = cartpole.step(0);
    //     std.debug.print("{d}\n", .{out.state});
    // }

    const glfw = @cImport({
        @cInclude("GLFW/glfw3.h");
    });

    var cartpole = try CartPoleMj.init();
    defer cartpole.deinit();

    if (glfw.glfwInit() == 0) {
        std.debug.print("Could not initialize GLFW\n", .{});
        return error.GlfwInitFailed;
    }
    defer glfw.glfwTerminate();

    const window = glfw.glfwCreateWindow(800, 600, "CartPole Sim", null, null);
    if (window == null) {
        std.debug.print("Could not create GLFW window\n", .{});
        return error.WindowCreationFailed;
    }
    glfw.glfwMakeContextCurrent(window);

    // Initialize visualization
    var cam: c.mjvCamera = undefined;
    var opt: c.mjvOption = undefined;
    var scn: c.mjvScene = undefined;
    var con: c.mjrContext = undefined;

    c.mjv_defaultCamera(&cam);
    c.mjv_defaultOption(&opt);
    c.mjv_defaultScene(&scn);
    c.mjr_defaultContext(&con);

    c.mjv_makeScene(cartpole.model, &scn, 2000);
    c.mjr_makeContext(cartpole.model, &con, c.mjFONTSCALE_150);
    defer {
        c.mjv_freeScene(&scn);
        c.mjr_freeContext(&con);
    }

    // Set camera position
    cam.distance = 3;
    cam.azimuth = 90;
    cam.elevation = -20;

    const control_freq = 1; // every N steps
    var steps_since_input: u32 = 0;
    var state: ?CartPoleMj.StepResult = null;
    while (glfw.glfwWindowShouldClose(window) == 0) {
        // Handle input at control frequency
        if (steps_since_input >= control_freq) {
            var action: u32 = 0;
            if (glfw.glfwGetKey(window, glfw.GLFW_KEY_LEFT) == glfw.GLFW_PRESS) {
                action = 0;
            } else if (glfw.glfwGetKey(window, glfw.GLFW_KEY_RIGHT) == glfw.GLFW_PRESS) {
                action = 1;
            }

            // Take step and print state
            state = cartpole.step(action);
            std.debug.print("\rState: pos={d:.3}, angle={d:.3}, vel={d:.3}, ang_vel={d:.3} | R: {d:.1}, Done: {}", .{
                state.?.state[0],
                state.?.state[1],
                state.?.state[2],
                state.?.state[3],
                state.?.reward,
                state.?.done,
            });

            steps_since_input = 0;
        } else {
            steps_since_input += 1;
        }

        // Render
        var viewport_width: c_int = undefined;
        var viewport_height: c_int = undefined;
        glfw.glfwGetFramebufferSize(window, &viewport_width, &viewport_height);

        c.mjv_updateScene(cartpole.model, cartpole.data, &opt, null, &cam, c.mjCAT_ALL, &scn);
        c.mjr_render(.{ .left = 0, .bottom = 0, .width = viewport_width, .height = viewport_height }, &scn, &con);

        glfw.glfwSwapBuffers(window);
        glfw.glfwPollEvents();

        // Handle reset
        if (glfw.glfwGetKey(window, glfw.GLFW_KEY_F1) == glfw.GLFW_PRESS or state.?.done > 0) {
            _ = cartpole.reset();
            std.debug.print("\nEnvironment reset\n", .{});
        }

        // Handle quit
        if (glfw.glfwGetKey(window, glfw.GLFW_KEY_Q) == glfw.GLFW_PRESS) {
            break;
        }
    }
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
};
