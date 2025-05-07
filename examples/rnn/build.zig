const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    inline for (&.{ "main", "gru" }) |file| {
        const zigrad_dep = b.dependency("zigrad", .{
            .target = target,
            .optimize = optimize,
        });

        const exe = b.addExecutable(.{
            .name = file,
            .root_source_file = b.path("src/" ++ file ++ ".zig"),
            .target = target,
            .optimize = optimize,
        });
        exe.linkLibC();
        exe.root_module.addImport("zigrad", zigrad_dep.module("zigrad"));
        b.installArtifact(exe);
        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| {
            run_cmd.addArgs(args);
        }
        const run_step = b.step(file, "Run the app");
        run_step.dependOn(&run_cmd.step);
    }
}
