const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const build_options = b.addOptions();
    build_options.step.name = "Zigrad build options";
    const build_options_module = build_options.createModule();
    build_options.addOption(std.log.Level, "log_level", b.option(std.log.Level, "log_level", "The Log Level to be used.") orelse .info);

    // TODO: cblas
    // switch (target.query.os_tag) {
    //     .linux => {},
    //     .macos => {},
    // }

    const zigrad_module = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
    });
    const lib = b.addStaticLibrary(.{
        .name = "zigrad",
        .root_source_file = zigrad_module.root_source_file.?,
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(lib);
    lib.root_module.addImport("build_options", build_options_module);

    const exe = b.addExecutable(.{
        .name = "zigrad",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("zigrad", zigrad_module);
    exe.root_module.addImport("build_options", build_options_module);

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

    // inline for ([_]struct {
    //     name: []const u8,
    //     src: []const u8,
    // }{
    //     .{ .name = "test", .src = "src/zarray.zig" },
    // }) |test| {
    // // const lib_unit_tests = b.addTest(.{
    // //     .root_source_file = test.name,
    // //     .target = target,
    // //     .optimize = optimize,
    // // });
    // // lib_unit_tests.linkFramework("Accelerate");
    // // const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);
    // // lib_unit_tests.root_module.addImport("zigrad", zigrad_module);
    // // test_step.dependOn(&run_lib_unit_tests.step);
    // }

    // const lib_unit_tests = b.addTest(.{
    //     .root_source_file = zigrad_module.root_source_file.?,
    //     .target = target,
    //     .optimize = optimize,
    // });
    // lib_unit_tests.linkFramework("Accelerate");

    const exe_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe_unit_tests.linkFramework("Accelerate");
    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);
    exe_unit_tests.root_module.addImport("zigrad", zigrad_module);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);

    inline for ([_]struct {
        name: []const u8,
        src: []const u8,
    }{
        .{ .name = "test", .src = "src/tensor/zarray.zig" },
        .{ .name = "test-tensor", .src = "src/tensor/tensor.zig" },
        .{ .name = "test-mnist", .src = "src/tensor/mnist.zig" },
    }) |excfg| {
        const ex_name = excfg.name;
        const ex_src = excfg.src;
        const libtest = b.addTest(.{
            .name = ex_name,
            .root_source_file = b.path(ex_src),
            .optimize = optimize,
        });
        libtest.linkFramework("Accelerate");
        const run_lib_unit_tests = b.addRunArtifact(libtest);
        libtest.root_module.addImport("zigrad", zigrad_module);
        // hahaha
        libtest.root_module.addImport("../backend/blas.zig", b.addModule("blas", .{ .root_source_file = b.path("src/backend/blas.zig") }));
        test_step.dependOn(&run_lib_unit_tests.step);
    }
}
