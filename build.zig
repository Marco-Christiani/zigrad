const std = @import("std");
const Backend = @import("src/device/root.zig").Backend;
const backend = @import("src/device/root.zig").backend;
const builtin = @import("builtin");

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

    const rebuild = b.option(bool, "rebuild", "force backend to recompile") orelse false;
    const device_module = build_device_module(b, target, rebuild);

    const zigrad = b.addModule("zigrad", .{
        .root_source_file = b.path("src/zigrad.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "build_options", .module = build_options_module },
            .{ .name = "device", .module = device_module },
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
    lib.root_module.addImport("build_options", build_options_module);
    lib.root_module.addImport("device", device_module);
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
    unit_tests.root_module.addImport("build_options", build_options_module);
    unit_tests.root_module.addImport("device", device_module);
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

pub fn build_device_module(b: *std.Build, target: std.Build.ResolvedTarget, rebuild: bool) *std.Build.Module {
    //const new_backend = get_backend(b);
    const new_backend: Backend = .HOST;

    const here = b.path(".").getPath(b);

    if (backend != new_backend) {
        run_command(b, &.{ "python3", b.pathJoin(&.{ here, "scripts", "backend.py" }), @tagName(new_backend) });
    }

    const device = b.createModule(.{
        .root_source_file = b.path("src/device/root.zig"),
        .link_libc = true,
        .target = target,
    });

    switch (target.result.os.tag) {
        .linux => {
            device.linkSystemLibrary("blas", .{});
        },
        .macos => device.linkFramework("Accelerate", .{}),
        else => @panic("Os not supported."),
    }

    if (new_backend == .CUDA) {
        const cuda = b.createModule(.{
            .root_source_file = b.path("src/cuda/root.zig"),
            .target = target,
            .link_libc = true,
        });

        const exists = amalgamate_exists(b);

        if (rebuild or !exists) {
            run_command(b, &.{
                "python3",
                b.pathJoin(&.{ here, "scripts", "cuda_setup.py" }),
                if (rebuild or !exists) "y" else "n",
            });
        }

        cuda.addIncludePath(b.path("src/cuda/"));
        cuda.addLibraryPath(b.path("src/cuda/"));
        cuda.linkSystemLibrary("amalgamate", .{});
        device.addImport("cuda", cuda);
    }

    return device;
}

fn amalgamate_exists(b: *std.Build) bool {
    const here = b.path(".").getPath(b);
    const path = b.pathJoin(&.{ here, "src", "cuda", "libamalgamate.so" });
    var file = std.fs.openFileAbsolute(path, .{});
    if (file) |*_file| {
        _file.close();
        return true;
    } else |_| {
        return false;
    }
}

fn get_backend(b: *std.Build) Backend {
    const env_backend = std.process.getEnvVarOwned(b.allocator, "ZIGRAD_BACKEND") catch {
        @panic("Environment variable 'ZIGRAD_BACKEND' not found.");
    };
    return std.meta.stringToEnum(Backend, env_backend) orelse {
        @panic("Invalid value for 'ZIGRAD_BACKEND' environment variable.");
    };
}

pub fn run_command(b: *std.Build, args: []const []const u8) void {
    const output = b.run(args);

    if (output.len > 0)
        std.debug.print("{s}", .{output});
}
