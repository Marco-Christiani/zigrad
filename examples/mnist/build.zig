const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const configs = .{
        .{ "src/main-simple.zig", "simple", "Run simple mnist." },
        .{ "src/main-timing.zig", "timing", "Run mnist with timing." },
    };

    const zigrad_dep = b.dependency("zigrad", .{
        .target = target,
        .optimize = optimize,
        .tracy_enable = false,
    });

    inline for (configs) |cfg| {
        const exe = b.addExecutable(.{
            .name = cfg[1],
            .root_source_file = b.path(cfg[0]),
            .target = target,
            .optimize = optimize,
        });
        exe.root_module.addImport("zigrad", zigrad_dep.module("zigrad"));
        b.installArtifact(exe);

        const cmd = b.addRunArtifact(exe);
        cmd.step.dependOn(b.getInstallStep());
        b.step(cfg[1], cfg[2]).dependOn(&cmd.step);
    }
}
