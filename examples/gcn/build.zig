const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const cfg = .{ "src/main.zig", "main" };

    const zigrad_dep = b.dependency("zigrad", .{
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = cfg[1],
        .root_source_file = b.path(cfg[0]),
        .target = target,
        .optimize = optimize,
    });
    exe.linkLibC();
    exe.root_module.addImport("zigrad", zigrad_dep.module("zigrad"));
    b.installArtifact(exe);

    const run_step = b.step("run", "run the main");
    const run_exe = b.addRunArtifact(exe);
    run_exe.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_exe.addArgs(args);
    }
    run_step.dependOn(&run_exe.step);
}
