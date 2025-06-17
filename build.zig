const std = @import("std");
const Build = std.Build;
const Module = Build.Module;
const OptimizeMode = std.builtin.OptimizeMode;

// this is a comment

pub fn build(b: *Build) !void {
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

    const enable_cuda = b.option(bool, "enable_cuda", "Enable CUDA backend.") orelse false;
    build_options.addOption(bool, "enable_cuda", enable_cuda);

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

    if (enable_cuda)
        add_cuda_module(b, zigrad, target);

    const lib = b.addStaticLibrary(.{
        .name = "zigrad",
        .root_module = zigrad,
    });
    lib.root_module.addImport("build_options", build_options_module);
    link(target, lib);
    b.installArtifact(lib);

    const exe = b.addExecutable(.{
        .name = "main",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zigrad", .module = zigrad },
            },
        }),
    });

    link(target, exe);
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

    unit_tests.root_module.addImport("build_options", build_options_module);
    const run_unit_tests = b.addRunArtifact(unit_tests);
    link(target, unit_tests);
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

    // Tracy -------------------------------------------------------------------
    const tracy_enable = b.option(bool, "tracy_enable", "Enable profiling") orelse false;
    if (tracy_enable) {
        const tracy = build_tracy(b, target).?;
        inline for (.{ zigrad, exe.root_module, unit_tests.root_module }) |e| e.addImport("tracy", tracy);
    }
}

fn link(target: Build.ResolvedTarget, exe: *Build.Step.Compile) void {
    switch (target.result.os.tag) {
        .linux => {
            exe.linkSystemLibrary("blas");
            exe.linkLibC();
        },
        .macos => exe.linkFramework("Accelerate"),
        else => @panic("Os not supported."),
    }
}

pub fn build_tracy(b: *Build, target: Build.ResolvedTarget) ?*Module {
    const optimize: OptimizeMode = b.option(OptimizeMode, "tracy_optimize_mode", "Defaults to ReleaseFast") orelse .ReleaseFast;
    const options = b.addOptions();
    const enable = true; // HACK: lazy
    options.addOption(bool, "tracy_enable", enable);
    options.addOption(bool, "tracy_allocation_enable", false);
    options.addOption(bool, "tracy_callstack_enable", true);
    options.addOption(usize, "tracy_callstack_depth", 10);

    const tracy = b.addModule("tracy", .{
        .root_source_file = b.path("src/tracy.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "tracy_build_options", .module = options.createModule() },
        },
    });

    const tracy_src = b.lazyDependency("tracy", .{}) orelse return null;
    const tracy_c_flags = &.{ "-DTRACY_ENABLE=1", "-fno-sanitize=undefined" };
    tracy.addCSourceFile(.{ .file = tracy_src.path("public/TracyClient.cpp"), .flags = tracy_c_flags });

    const unit_tests = b.addTest(.{
        .name = "tracy_test",
        .root_module = tracy,
    });
    const test_step = b.step("tracy_test", "Run unit tests");
    const run_unit_tests = b.addRunArtifact(unit_tests);
    test_step.dependOn(&run_unit_tests.step);

    const exe = b.addExecutable(.{
        .name = "tracy_demo",
        .root_module = tracy,
    });
    const run_step = b.step("tracy_demo", "");
    const run_demo = b.addRunArtifact(exe);
    run_step.dependOn(&run_demo.step);
    return tracy;
}

pub fn add_cuda_module(
    b: *Build,
    zigrad: *Module,
    target: Build.ResolvedTarget,
) void {
    const rebuild_cuda: bool = b.option(bool, "rebuild_cuda", "force CUDA backend to recompile") orelse false;

    const here = b.path(".").getPath(b);

    const cuda = b.createModule(.{
        .root_source_file = b.path("src/cuda/root.zig"),
        .target = target,
        .link_libc = true,
    });

    const exists = amalgamate_exists(b);

    if (rebuild_cuda or !exists) {
        std.log.info("COMPILING CUDA BACKEND", .{});
        run_command(b, &.{
            "python3",
            b.pathJoin(&.{ here, "scripts", "cuda_setup.py" }),
            if (rebuild_cuda) "y" else "n",
        });
    }

    cuda.addIncludePath(b.path("src/cuda/"));
    cuda.addLibraryPath(b.path("src/cuda/build"));
    cuda.linkSystemLibrary("amalgamate", .{});
    zigrad.addImport("cuda", cuda);
}

fn amalgamate_exists(b: *Build) bool {
    const here = b.path(".").getPath(b);
    const path = b.pathJoin(&.{ here, "src", "cuda", "build", "libamalgamate.so" });
    var file = std.fs.openFileAbsolute(path, .{});
    if (file) |*_file| {
        _file.close();
        return true;
    } else |_| {
        return false;
    }
}

pub fn run_command(b: *Build, args: []const []const u8) void {
    const output = b.run(args);

    if (output.len > 0)
        std.debug.print("{s}", .{output});
}
