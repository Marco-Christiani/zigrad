const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const configs = .{
        .{ "src/main-simple.zig", "main" },
    };

    const zigrad_dep = b.dependency("zigrad", .{
        .target = target,
        .optimize = optimize,
    });

    inline for (configs) |cfg| {
        const exe = b.addExecutable(.{
            .name = cfg[1],
            .root_source_file = b.path(cfg[0]),
            .target = target,
            .optimize = optimize,
        });
        exe.linkLibC();
        exe.root_module.addImport("zigrad", zigrad_dep.module("zigrad"));
        b.installArtifact(exe);
    }
}
