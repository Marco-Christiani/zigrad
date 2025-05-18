const std = @import("std");
const OptimizeMode = std.builtin.OptimizeMode;
const Build = std.Build;
const Compile = std.Build.Step.Compile;

// pub fn build(b: *std.Build) void {
//     const target = b.standardTargetOptions(.{});
//     const optimize = b.standardOptimizeOption(.{});
//
//     const zigrad_dep = b.dependency("zigrad", .{
//         .target = target,
//         .optimize = optimize,
//         .tracy_enable = false,
//     });
//     const exe = b.addExecutable(.{
//         .name = "cuda_tests",
//         .root_source_file = b.path("src/cuda_tests.zig"),
//         .target = target,
//         .optimize = optimize,
//         .link_libc = true,
//     });
//     exe.root_module.addImport("zigrad", zigrad_dep.module("zigrad"));
//     b.installArtifact(exe);
//
//     const run_cmd = b.addRunArtifact(exe);
//     run_cmd.step.dependOn(b.getInstallStep());
//
//     // Arg passthru (`zig build run -- arg1 arg2 etc`)
//     if (b.args) |args| {
//         run_cmd.addArgs(args);
//     }
//     const run_step = b.step("cuda_tests", "Compare cuda to host.");
//     run_step.dependOn(&run_cmd.step);
// }

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zigrad_dep = b.dependency("zigrad", .{
        .target = target,
        .optimize = optimize,
        .tracy_enable = false,
    });

    const exe = build_python(b, target, optimize) catch @panic("Build failed");
    exe.root_module.addImport("zigrad", zigrad_dep.module("zigrad"));
    b.installArtifact(exe);
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run tests");
    run_step.dependOn(&run_cmd.step);

    const unit_tests = b.addTest(.{
        .root_module = exe.root_module,
        .target = target,
        .optimize = optimize,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_unit_tests.step);
}

fn build_python(b: *Build, target: Build.ResolvedTarget, optimize: OptimizeMode) !*Compile {
    const exe = b.addExecutable(.{
        .name = "main",
        .root_source_file = b.path("embed_python.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const alloc = b.allocator;

    const base_root = try getPythonInfo(alloc,
        \\import sys, pathlib; print(pathlib.Path(sys.base_prefix).resolve())
    );
    const pyver = try getPythonInfo(alloc,
        \\import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")
    );
    // const libname = pyver["python".len..];

    const incl = try std.fs.path.join(alloc, &.{ base_root, "include", pyver });
    const libdir = try std.fs.path.join(alloc, &.{ base_root, "lib" });

    // std.debug.print("INCL: {s}\nLIBDIR: {s}\nLIBNAME: python{s}\n", .{ incl, libdir, libname });
    exe.addIncludePath(.{ .cwd_relative = incl });
    exe.linkSystemLibrary(pyver);
    exe.addLibraryPath(.{ .cwd_relative = libdir });

    return exe;
}

fn getPythonInfo(allocator: std.mem.Allocator, script: []const u8) ![]const u8 {
    const child = try std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "python3", "-c", script },
        // .= .Pipe,
    });

    // defer allocator.free(child.stdout);
    defer allocator.free(child.stderr);
    return std.mem.trimRight(u8, child.stdout, "\n");
}
