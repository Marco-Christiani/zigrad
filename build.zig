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

    // TODO: cblas
    // switch (target.query.os_tag) {
    //     .linux => {},
    //     .macos => {},
    // }
    // const zarray_module = b.createModule(.{ .root_source_file = b.path("src/tensor/zarray.zig") });
    // const backend_module = b.createModule(.{ .root_source_file = b.path("src/backend/blas.zig") });
    // zarray_module.addImport("blas", backend_module);

    const zigrad_module = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .imports = &.{
            .{ .name = "build_options", .module = build_options_module },
        },
    });
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
    // exe.root_module.addImport("build_options", build_options_module);

    exe.linkFramework("Accelerate");
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

    const unit_tests = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    unit_tests.linkFramework("Accelerate");
    const run_unit_tests = b.addRunArtifact(unit_tests);
    // unit_tests.root_module.addImport("zigrad", zigrad_module);

    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_unit_tests.step);

    // inline for ([_]struct {
    //     name: []const u8,
    //     src: []const u8,
    // }{
    //     // .{ .name = "test-zarray", .src = "src/tensor/zarray.zig" },
    //     // .{ .name = "test-tensor", .src = "src/tensor/tensor.zig" },
    //     // .{ .name = "test-mnist", .src = "src/tensor/mnist.zig" },
    //     // .{ .name = "test-ops", .src = "src/tensor/ops.zig" },
    //     // .{ .name = "test-layer", .src = "src/tensor/layer.zig" },
    //     // .{ .name = "test-conv-utils", .src = "src/tensor/conv_utils.zig" },
    //     .{ .name = "test-conv", .src = "src/tensor/conv_test.zig" },
    // }) |test_cfg| {
    //     const test_name = test_cfg.name;
    //     const test_src = test_cfg.src;
    //     const libtest = b.addTest(.{
    //         .name = test_name,
    //         .root_source_file = b.path(test_src),
    //         .optimize = optimize,
    //     });
    //     libtest.linkFramework("Accelerate");
    //     // const curr_test = b.step(test_name, test_name);
    //     const run_lib_unit_tests = b.addRunArtifact(libtest);
    //     libtest.root_module.addImport("zigrad", zigrad_module);
    //     // hahaha. done the hacky way for zls.
    //     libtest.root_module.addImport("blas", b.addModule("blas", .{ .root_source_file = b.path("src/backend/blas.zig") }));
    //     test_step.dependOn(&run_lib_unit_tests.step);
    // }
}
