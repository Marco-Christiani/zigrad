const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const build_options = b.addOptions();
    build_options.step.name = "Zigrad build options";
    const build_options_module = build_options.createModule();
    build_options.addOption(
        std.log.Level,
        "log_level",
        b.option(std.log.Level, "log_level", "The Log Level to be used.") orelse .info,
    );

    const zigrad = b.addModule("zigrad", .{
        .root_source_file = b.path("src/zigrad.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "build_options", .module = build_options_module },
        },
    });
    switch (target.result.os.tag) {
        .linux => {
            zigrad.linkSystemLibrary("blas", .{});
        },
        .macos => zigrad.linkFramework("Accelerate", .{}),
        else => @panic("Os not supported."),
    }

    const tracy_enable = b.option(bool, "tracy_enable", "Enable profiling") orelse false;
    const tracy = b.lazyDependency("tracy", .{
        .target = target,
        .optimize = optimize,
        .tracy_enable = tracy_enable,
    });
    if (tracy_enable) zigrad.addImport("tracy", tracy.?.module("tracy"));

    const lib = b.addStaticLibrary(.{
        .name = "zigrad",
        .root_source_file = zigrad.root_source_file.?,
        .target = target,
        .optimize = optimize,
    });
    link(target, lib);
    if (tracy_enable) add_tracy(lib, tracy.?);
    b.installArtifact(lib);

    const exe = b.addExecutable(.{
        .name = "zigrad",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe.root_module.addImport("zigrad", zigrad);
    link(target, exe);
    if (tracy_enable) add_tracy(exe, tracy.?);
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    // Arg passthru (`zig build run -- arg1 arg2 etc`)
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const unit_tests = b.addTest(.{
        .root_source_file = b.path("src/zigrad.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    link(target, unit_tests);
    if (tracy_enable) add_tracy(unit_tests, tracy.?);
    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_unit_tests.step);

    // doc gen
    const docs_step = b.addInstallDirectory(.{
        .source_dir = lib.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    docs_step.step.dependOn(&exe.step);

    const docs = b.step("docs", "Generate documentation");
    docs.dependOn(&docs_step.step);
}

fn link(target: std.Build.ResolvedTarget, exe: *std.Build.Step.Compile) void {
    switch (target.result.os.tag) {
        .linux => {
            exe.linkSystemLibrary("blas");
            exe.linkLibC();
        },
        .macos => exe.linkFramework("Accelerate"),
        else => @panic("Os not supported."),
    }
}

fn add_tracy(exe: *std.Build.Step.Compile, tracy: *std.Build.Dependency) void {
    exe.root_module.addImport("tracy", tracy.module("tracy"));
    exe.linkLibrary(tracy.artifact("tracy"));
    exe.linkLibCpp();
}
