const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zigrad_dep = b.dependency("zigrad", .{
        .target = target,
        .optimize = optimize,
        .tracy_enable = false,
    });
    const tensorboard_dep = b.dependency("tensorboard", .{
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "main",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.linkLibC();
    exe.root_module.addImport("zigrad", zigrad_dep.module("zigrad"));
    exe.root_module.addImport("tensorboard", tensorboard_dep.module("tensorboard"));

    exe.addIncludePath(.{ .cwd_relative = "./mujoco-3.2.6/include/" });
    exe.addLibraryPath(.{ .cwd_relative = "./mujoco-3.2.6/lib/" });
    exe.linkSystemLibrary("mujoco");
    exe.linkSystemLibrary("glfw");
    b.installArtifact(exe);
}
