const std = @import("std");
const c = @cImport({
    @cInclude("mujoco/mujoco.h");
    @cInclude("GLFW/glfw3.h");
});

pub fn main() !void {
    var err: [1000]u8 = undefined;
    const model = c.mj_loadXML("cartpole.xml", null, &err, @intCast(err.len));
    if (model == null) {
        std.debug.print("Could not load model: {s}\n", .{err});
        return error.LoadModelFailed;
    }
    defer c.mj_deleteModel(model);

    const data = c.mj_makeData(model);
    if (data == null) {
        std.debug.print("Could not allocate mjData\n", .{});
        return error.AllocateDataFailed;
    }
    defer c.mj_deleteData(data);

    if (c.glfwInit() == 0) {
        std.debug.print("Could not initialize GLFW\n", .{});
        return error.GlfwInitFailed;
    }
    defer c.glfwTerminate();

    const window = c.glfwCreateWindow(800, 600, "Sim", null, null);
    if (window == null) {
        std.debug.print("Could not create GLFW window\n", .{});
        return error.WindowCreationFailed;
    }
    c.glfwMakeContextCurrent(window);

    var cam: c.mjvCamera = undefined;
    var opt: c.mjvOption = undefined;
    var scn: c.mjvScene = undefined;
    var con: c.mjrContext = undefined;

    c.mjv_defaultCamera(&cam);
    c.mjv_defaultOption(&opt);
    c.mjv_defaultScene(&scn);
    c.mjr_defaultContext(&con);

    c.mjv_makeScene(model, &scn, 2000);
    c.mjr_makeContext(model, &con, c.mjFONTSCALE_150);

    cam.distance = 3;
    cam.azimuth = 90;
    cam.elevation = -20;

    const control_freq = 1; // every N steps
    var paused = false;
    var steps_since_input: u32 = 0;
    while (c.glfwWindowShouldClose(window) == 0) {
        if (steps_since_input >= control_freq) {
            handleInput(window, model, data);
            std.debug.print("{d}\n", .{[_]f64{
                data.*.qpos[0],
                data.*.qpos[1],
                data.*.qvel[0],
                data.*.qvel[1],
            }});
            steps_since_input = 0;
        } else {
            steps_since_input += 1;
        }
        c.mj_step(model, data);

        // Render
        var viewport_width: c_int = undefined;
        var viewport_height: c_int = undefined;
        c.glfwGetFramebufferSize(window, &viewport_width, &viewport_height);

        c.mjv_updateScene(model, data, &opt, null, &cam, c.mjCAT_ALL, &scn);
        c.mjr_render(.{ .left = 0, .bottom = 0, .width = viewport_width, .height = viewport_height }, &scn, &con);

        c.glfwSwapBuffers(window);
        c.glfwPollEvents();

        // Handle pause/reset
        if (c.glfwGetKey(window, c.GLFW_KEY_SPACE) == c.GLFW_PRESS) {
            paused = !paused;
            c.glfwWaitEvents();
        }

        if (c.glfwGetKey(window, c.GLFW_KEY_R) == c.GLFW_PRESS) {
            c.mj_resetData(model, data);
        }
    }

    c.mjv_freeScene(&scn);
    c.mjr_freeContext(&con);
}
fn handleInput(window: ?*c.GLFWwindow, model: *c.mjModel, data: *c.mjData) void {
    const actuator = @as(usize, @intCast(c.mj_name2id(model, c.mjOBJ_ACTUATOR, "slide")));

    const speed = 0.05;

    if (c.glfwGetKey(window, c.GLFW_KEY_LEFT) == c.GLFW_PRESS) {
        data.*.ctrl[actuator] = -speed;
    } else if (c.glfwGetKey(window, c.GLFW_KEY_RIGHT) == c.GLFW_PRESS) {
        data.*.ctrl[actuator] = speed;
    } else {
        data.*.ctrl[actuator] = 0;
    }
}
