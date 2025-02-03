const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zigrad_dep = b.dependency("zigrad", .{
        .target = target,
        .optimize = optimize,
        .tracy_enable = false,
    });
    const exe = b.addExecutable(.{
        .name = "cuda_tests",
        .root_source_file = b.path("src/cuda_tests.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    exe.root_module.addImport("zigrad", zigrad_dep.module("zigrad"));
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    // Arg passthru (`zig build run -- arg1 arg2 etc`)
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("cuda_tests", "Compare cuda to host.");
    run_step.dependOn(&run_cmd.step);
}
