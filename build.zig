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

    const zigrad_module = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .imports = &.{
            .{ .name = "build_options", .module = build_options_module },
        },
    });

    const tracy_enable = b.option(bool, "tracy_enable", "Enable profiling") orelse false;
    const tracy = b.dependency("tracy", .{
        .target = target,
        .optimize = optimize,
        .tracy_enable = tracy_enable,
    });

    zigrad_module.addImport("tracy", tracy.module("tracy"));

    // const lib = b.addStaticLibrary(.{
    //     .name = "zigrad",
    //     .root_source_file = zigrad_module.root_source_file.?,
    //     .target = target,
    //     .optimize = optimize,
    // });
    // lib.root_module.addImport("zigrad", zigrad_module);
    // b.installArtifact(lib);

    const exe = b.addExecutable(.{
        .name = "zigrad",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe.root_module.addImport("zigrad", zigrad_module);
    std.debug.print("target.result.os.tag={}\n", .{target.result.os.tag});
    // TODO: cblas
    link(target, exe);
    add_tracy(exe, tracy);
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // MNIST -------------------------------------------------------------------
    // const mnist_exe = b.addExecutable(.{
    //     .name = "zigrad",
    //     .root_source_file = b.path("src/nn/mnist.zig"),
    //     .target = target,
    //     .optimize = optimize,
    // });
    // mnist_exe.root_module.addImport("zigrad", zigrad_module);
    //
    // link(target, mnist_exe);
    // b.installArtifact(mnist_exe);
    // const run_mnist_cmd = b.addRunArtifact(mnist_exe);
    // run_mnist_cmd.step.dependOn(b.getInstallStep());
    // b.step("run-mnist", "Run mnist example").dependOn(&run_mnist_cmd.step);
    // -------------------------------------------------------------------------

    const unit_tests = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    link(target, unit_tests);
    // unit_tests.root_module.addImport("tracy", tracy.module("tracy"));
    add_tracy(unit_tests, tracy);
    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_unit_tests.step);
}

fn link(target: std.Build.ResolvedTarget, exe: *std.Build.Step.Compile) void {
    switch (target.result.os.tag) {
        .linux => {
            // exe.addSystemFrameworkPath
            // exe.linkFramework("cblas");
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
