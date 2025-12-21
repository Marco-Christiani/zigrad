const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Library module
    const zigrad_mod = b.addModule("zigrad", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Link libc for dlopen
    zigrad_mod.link_libc = true;

    // Add include path for PJRT C headers
    zigrad_mod.addIncludePath(b.path("src"));

    // M1 test executable
    const exe = b.addExecutable(.{
        .name = "zigrad-pjrt-m1",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zigrad", .module = zigrad_mod },
            },
        }),
    });

    exe.linkLibC();
    exe.root_module.addIncludePath(b.path("src"));

    b.installArtifact(exe);

    // Run step for M1
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the M1 test executable");
    run_step.dependOn(&run_cmd.step);

    // M2 test executable (matmul)
    const exe_m2 = b.addExecutable(.{
        .name = "zigrad-pjrt-m2",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/m2_matmul.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zigrad", .module = zigrad_mod },
            },
        }),
    });

    exe_m2.linkLibC();
    exe_m2.root_module.addIncludePath(b.path("src"));

    b.installArtifact(exe_m2);

    // Run step for M2
    const run_m2_cmd = b.addRunArtifact(exe_m2);
    run_m2_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_m2_cmd.addArgs(args);
    }

    const run_m2_step = b.step("run-m2", "Run the M2 test executable");
    run_m2_step.dependOn(&run_m2_cmd.step);

    // Unit tests
    const lib_tests = b.addTest(.{
        .root_module = zigrad_mod,
    });

    const run_lib_tests = b.addRunArtifact(lib_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_tests.step);
}
