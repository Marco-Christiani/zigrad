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

    const zigrad = b.addModule("zigrad", .{
        .root_source_file = b.path("src/zigrad.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "build_options", .module = build_options_module },
            .{ .name = "device", .module = buildDeviceModule(b, target) },
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

pub fn buildDeviceModule(b: *std.Build, target: std.Build.ResolvedTarget) *std.Build.Module {
    const rebuild = b.option(bool, "rebuild", "force backend to recompile") orelse false;

    // const new_backend = getBackend(b);
    const new_backend: Backend = .HOST;

    const here = b.path(".").getPath(b);

    if (backend != new_backend) {
        runCommand(b, &.{ "python3", b.pathJoin(&.{ here, "scripts", "backend.py" }), @tagName(new_backend) });
    }

    const device = b.addModule("device", .{ 
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

        const cuda = b.addModule("cuda", .{
            .root_source_file = b.path("src/cuda/root.zig"),
            .target = target,
            .link_libc = true,
        });
        
        const info = getCudaInfo(b);

        cuda.addIncludePath(b.path("src/cuda/"));
        cuda.addLibraryPath(b.path("src/cuda/"));
        cuda.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ info.cuda_path, "lib64" }) });

        switch (builtin.os.tag) {
            .linux => cuda.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ info.cuda_path, "targets/x86_64-linux/lib/stubs" }) }),            
            else => @panic("Windows OS is currently unsupported by Zigrad."),
        }

        cuda.linkSystemLibrary("cuda", .{});
        cuda.linkSystemLibrary("cudart", .{});
        cuda.linkSystemLibrary("nvrtc", .{});
        cuda.linkSystemLibrary("cublas", .{});
        cuda.linkSystemLibrary("cudnn", .{});

        if (rebuild or !amalgomateExists(b)) {
            runCommand(b, &.{ "python3", b.pathJoin(&.{ here, "scripts", "cuda_includes.py" }), info.cuda_path });
            compileCuda(b, info.cuda_path, info.gpu_arch);
        }

        // this is created by the compile cuda step
        cuda.linkSystemLibrary("amalgomate", .{});

        device.addImport("cuda", cuda);
    }

    return device;
}

pub fn compileCuda(
    b: *std.Build,
    cuda_path: []const u8,
    gpu_arch: []const u8,
) void {
    std.log.info("Compiling libamalgomate.so...\n", .{});

    const cuda_nvcc_path = b.pathJoin(&.{ cuda_path, "bin", "nvcc" });
    const cuda_include_path = b.pathJoin(&.{ "-I", cuda_path, "include" });
    const cuda_library_path = b.pathJoin(&.{ "-L", cuda_path, "lib64" });
    const gpu_architecture = std.mem.join(b.allocator, "", &.{ "--gpu-architecture=", gpu_arch }) catch unreachable;

    const here = b.path(".").getPath(b);
    const source_path = b.pathJoin(&.{ here, "src", "cuda", "amalgomate.cu" });
    const target_path = b.pathJoin(&.{ here, "src", "cuda", "libamalgomate.so" });

    const libgen_utils_argv: []const []const u8 = &.{
        cuda_nvcc_path,
        "--shared",
        "-o",
        target_path,
        source_path,
        "-O3",
        gpu_architecture,
        "--expt-extended-lambda",
        "--compiler-options",
        "-fPIC",
        cuda_include_path,
        cuda_library_path,
        "-lcudart",
        "-lcuda",
        "-lcublas",
        "-lcudnn",
    };

    const result = std.process.Child.run(.{ 
        .allocator = b.allocator, 
        .argv = libgen_utils_argv,
    }) catch |e| {
        std.log.err("Error: {}", .{e});
        @panic("Failed to create amalgomate.so");
    };

    if (result.stderr.len != 0) {
        std.log.err("Error: {s}", .{result.stderr});
        @panic("Failed to create amalgomate.so");
    }
}

fn amalgomateExists(b: *std.Build) bool {
    const here = b.path(".").getPath(b);
    const path = b.pathJoin(&.{ here, "src", "cuda", "libamalgomate.so" });
    var file = std.fs.openFileAbsolute(path, .{});
    if (file) |*_file| {
        _file.close();
        return true;
    } else |_| {
        return false;
    }
}

fn getBackend(b: *std.Build) Backend {
    const env_backend = std.process.getEnvVarOwned(b.allocator, "ZIGRAD_BACKEND") catch {
        @panic("Environment variable 'ZIGRAD_BACKEND' not found.");  
    };
    return std.meta.stringToEnum(Backend, env_backend) orelse {
        @panic("Invalid value for 'ZIGRAD_BACKEND' environment variable.");
    };
}

fn getCudaInfo(b: *std.Build) struct {
    cuda_path: []const u8,
    gpu_arch: []const u8,
} {
    const env_cuda_path = std.process.getEnvVarOwned(b.allocator, "ZIGRAD_CUDA_PATH") catch {
        @panic("Environment variable 'ZIGRAD_CUDA_PATH' not found.");  
    };
    const env_gpu_arch = std.process.getEnvVarOwned(b.allocator, "ZIGRAD_GPU_ARCH") catch {
        @panic("Environment variable 'ZIGRAD_GPU_ARCH' not found.");  
    };
    return .{
        .cuda_path = env_cuda_path,
        .gpu_arch = env_gpu_arch,  
    };
}

pub fn runCommand(b: *std.Build, args: []const []const u8) void {
    const output = b.run(args);

    if (output.len > 0)
        std.debug.print("{s}", .{ output });
}
