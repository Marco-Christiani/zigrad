// render with an offscreen buffer for recording
const std = @import("std");

pub const RenderOptions = struct {
    width: u32 = 800,
    height: u32 = 600,
    title: [:0]const u8 = "MuJoCo Simulation",
    cam_distance: f32 = 3.0,
    cam_azimuth: f32 = 90.0,
    cam_elevation: f32 = -20.0,
    offscreen: bool = false,
};

pub const RecordingOptions = struct {
    /// Path to save the recording (e.g., "recording.rgb").
    filepath: []const u8,
    fps: f64 = 30.0,
    add_depth: bool = false,
};

pub const MjRenderer = struct {
    // nested so the user can opt-in to requiring a glfw dependency
    pub const c = @cImport({
        @cInclude("mujoco/mujoco.h");
        @cInclude("GLFW/glfw3.h");
    });
    const Self = @This();

    window: ?*c.GLFWwindow,
    cam: c.mjvCamera,
    opt: c.mjvOption,
    scn: c.mjvScene,
    con: c.mjrContext,
    viewport: c.mjrRect,
    rgb_buffer: ?[]u8,
    depth_buffer: ?[]f32,
    file: ?std.fs.File,
    last_frame_time: f64,
    record_opts: ?RecordingOptions,
    allocator: std.mem.Allocator,

    pub fn init(model_: *anyopaque, options: RenderOptions, allocator: std.mem.Allocator) !Self {
        // HACK: this casting is a nasty workaround for some zig issues I assume with be fixed.
        const model: *c.mjModel = @ptrCast(@alignCast(model_));
        if (c.glfwInit() == 0) {
            return error.GlfwInitFailed;
        }
        errdefer c.glfwTerminate();

        // Setup window and context
        var window: ?*c.GLFWwindow = null;
        if (!options.offscreen) {
            window = c.glfwCreateWindow(
                @intCast(options.width),
                @intCast(options.height),
                options.title.ptr,
                null,
                null,
            ) orelse return error.WindowCreationFailed;
            c.glfwMakeContextCurrent(window);
        } else {
            // Create invisible window for offscreen rendering
            c.glfwWindowHint(c.GLFW_VISIBLE, 0);
            c.glfwWindowHint(c.GLFW_DOUBLEBUFFER, c.GLFW_FALSE);
            window = c.glfwCreateWindow(
                @intCast(options.width),
                @intCast(options.height),
                "Offscreen",
                null,
                null,
            ) orelse return error.WindowCreationFailed;
            c.glfwMakeContextCurrent(window);
        }

        var cam: c.mjvCamera = undefined;
        var opt: c.mjvOption = undefined;
        var scn: c.mjvScene = undefined;
        var con: c.mjrContext = undefined;

        c.mjv_defaultCamera(&cam);
        c.mjv_defaultOption(&opt);
        c.mjv_defaultScene(&scn);
        c.mjr_defaultContext(&con);

        c.mjv_makeScene(model, &scn, 2000);
        c.mjr_makeContext(model, &con, 200);

        cam.distance = options.cam_distance;
        cam.azimuth = options.cam_azimuth;
        cam.elevation = options.cam_elevation;

        // If offscreen, set buffer
        if (options.offscreen) {
            c.mjr_setBuffer(c.mjFB_OFFSCREEN, &con);
            if (con.currentBuffer != c.mjFB_OFFSCREEN) {
                return error.OffscreenNotSupported;
            }
        }

        return Self{
            .window = window,
            .cam = cam,
            .opt = opt,
            .scn = scn,
            .con = con,
            .viewport = c.mjr_maxViewport(&con),
            .rgb_buffer = null,
            .depth_buffer = null,
            .file = null,
            .last_frame_time = 0,
            .record_opts = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.rgb_buffer) |buf| self.allocator.free(buf);
        if (self.depth_buffer) |buf| self.allocator.free(buf);
        if (self.file) |f| f.close();

        c.mjv_freeScene(&self.scn);
        c.mjr_freeContext(&self.con);
        if (self.window) |w| {
            c.glfwDestroyWindow(w);
        }
        c.glfwTerminate();
    }

    pub fn startRecording(self: *Self, opts: RecordingOptions) !void {
        const total_pixels = @as(usize, @intCast(self.viewport.width * self.viewport.height));

        // rendering buffers
        self.rgb_buffer = try self.allocator.alloc(u8, 3 * total_pixels);
        self.depth_buffer = try self.allocator.alloc(f32, total_pixels);

        // output file
        self.file = try std.fs.cwd().createFile(opts.filepath, .{});
        self.record_opts = opts;
        self.last_frame_time = 0;

        // ffmpeg command for later conversion
        std.debug.print("\nTo convert the recording to MP4, use this command:\n", .{});
        std.debug.print("ffmpeg -f rawvideo -pixel_format rgb24 -video_size {}x{} -framerate {} -i {s} -vf \"vflip,format=yuv420p\" output.mp4\n\n", .{
            self.viewport.width,
            self.viewport.height,
            opts.fps,
            opts.filepath,
        });
    }

    pub fn render(self: *Self, model_: *anyopaque, data_: *anyopaque) !void {
        // HACK: this casting is a nasty workaround for some zig issues I assume with be fixed.
        const model: *c.mjModel = @ptrCast(@alignCast(model_));
        const data: *c.mjData = @ptrCast(@alignCast(data_));

        // TODO: buffer clearing could be better? not sure, theres an artifact. This is a linker error.
        // c.glClearColor(0, 0, 0, 1);
        // c.glClear(c.GL_COLOR_BUFFER_BIT | c.GL_DEPTH_BUFFER_BIT);

        c.mjv_updateScene(model, data, &self.opt, null, &self.cam, c.mjCAT_ALL, &self.scn);

        c.mjr_render(self.viewport, &self.scn, &self.con);

        if (self.record_opts) |opts| {
            // check if it's time for a new frame
            if ((data.time - self.last_frame_time) > 1.0 / opts.fps or self.last_frame_time == 0) {
                // add timestamp text
                var stamp_buf: [50]u8 = undefined;
                const stamp = try std.fmt.bufPrintZ(&stamp_buf, "Time = {d:.3}", .{data.time});
                c.mjr_overlay(c.mjFONT_NORMAL, c.mjGRID_TOPLEFT, self.viewport, stamp.ptr, null, &self.con);

                // read pixels
                if (self.rgb_buffer) |rgb| {
                    if (self.depth_buffer) |depth| {
                        c.mjr_readPixels(@ptrCast(rgb.ptr), @ptrCast(depth.ptr), self.viewport, &self.con);

                        // add depth visualization
                        if (opts.add_depth) {
                            const NS = 3; // subsample factor
                            const W = @as(usize, @intCast(self.viewport.width));
                            const H = @as(usize, @intCast(self.viewport.height));
                            var r: usize = 0;
                            while (r < H) : (r += NS) {
                                var cc: usize = 0;
                                while (cc < W) : (cc += NS) {
                                    const adr = (r / NS) * W + cc / NS;
                                    const d = @as(u8, @intFromFloat((1.0 - depth[r * W + cc]) * 255.0));
                                    rgb[3 * adr] = d;
                                    rgb[3 * adr + 1] = d;
                                    rgb[3 * adr + 2] = d;
                                }
                            }
                        }

                        // write frame
                        try self.file.?.writeAll(rgb);

                        // write exact frame size
                        // const frame_size = 3 * @as(usize, @intCast(self.viewport.width * self.viewport.height));
                        // if (rgb.len != frame_size) {
                        //     return error.InvalidFrameSize;
                        // }
                        // const bytes_written = try self.file.?.write(rgb[0..frame_size]);
                        // if (bytes_written != frame_size) {
                        //     return error.IncompleteWrite;
                        // }
                        self.last_frame_time = data.time;
                    }
                }
            }
        }

        // swap buffers if not offscreen
        if (self.window) |w| {
            c.glfwSwapBuffers(w);
            c.glfwPollEvents();
        }
    }

    pub fn shouldClose(self: Self) bool {
        return if (self.window) |w| c.glfwWindowShouldClose(w) != 0 else false;
    }

    pub fn getKey(self: Self, key: c_int) bool {
        return if (self.window) |w| c.glfwGetKey(w, key) == c.GLFW_PRESS else false;
    }
};
