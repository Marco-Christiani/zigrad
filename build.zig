const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const tgt_exe_file = b.option([]const u8, "file", "target file") orelse "src/main.zig";
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
        // .imports = &.{ // not working or working??
        //     .{ .name = "build_options", .module = build_options_module },
        // },
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
        .root_source_file = b.path(tgt_exe_file),
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

    const lib_unit_tests = b.addTest(.{
        .root_source_file = zigrad_module.root_source_file.?,
        .target = target,
        .optimize = optimize,
    });
    lib_unit_tests.linkFramework("Accelerate");
    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);
    lib_unit_tests.root_module.addImport("zigrad", zigrad_module);

    const exe_unit_tests = b.addTest(.{
        .root_source_file = b.path(tgt_exe_file),
        .target = target,
        .optimize = optimize,
    });
    exe_unit_tests.linkFramework("Accelerate");
    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);
    exe_unit_tests.root_module.addImport("zigrad", zigrad_module);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);
}
