const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const sdk_root = b.option([]const u8, "sdk", "Path to Zigrad external SDK root") orelse "../../result";
    const sdk_abs = resolve_absolute(b, sdk_root);

    const zigrad_dep = b.dependency("zigrad", .{
        .target = target,
        .optimize = optimize,
        .sdk = sdk_abs,
    });
    const zigrad_mod = zigrad_dep.module("zigrad");

    const exe = b.addExecutable(.{
        .name = "mnist",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zigrad", .module = zigrad_mod },
            },
        }),
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    b.step("run", "Run MNIST training").dependOn(&run_cmd.step);
}

fn resolve_absolute(b: *std.Build, path: []const u8) []const u8 {
    if (std.fs.path.isAbsolute(path)) return path;
    const cwd_abs = std.fs.cwd().realpathAlloc(b.allocator, ".") catch @panic("realpathAlloc failed");
    return std.fs.path.join(b.allocator, &.{ cwd_abs, path }) catch @panic("path join failed");
}
