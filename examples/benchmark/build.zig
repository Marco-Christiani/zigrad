const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const sdk_root = b.option([]const u8, "sdk", "Path to zigrad external SDK root (include/, lib/, runtime/)") orelse "../../result";

    // Resolve to absolute so the path works from both the example and zigrad contexts.
    const sdk_abs = resolve_absolute(b, sdk_root);
    const sdk_lib = b.fmt("{s}/lib", .{sdk_abs});

    const zigrad_dep = b.dependency("zigrad", .{
        .target = target,
        .optimize = optimize,
        .sdk = @as([]const u8, sdk_abs),
    });
    const zigrad_mod = zigrad_dep.module("zigrad");

    const exe = b.addExecutable(.{
        .name = "benchmark",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "zigrad", .module = zigrad_mod },
            },
        }),
    });

    // SDK library linking. Taken from zigrad.
    exe.root_module.addLibraryPath(.{ .cwd_relative = sdk_lib });
    if (sdk_has(b, sdk_abs, &.{ "include", "mlir-c", "IR.h" })) {
        exe.root_module.linkSystemLibrary("stdc++", .{});
        exe.root_module.linkSystemLibrary("MLIR-C", .{});
        exe.root_module.linkSystemLibrary("StablehloCAPI", .{});
    }
    if (sdk_has(b, sdk_abs, &.{ "include", "mkl_cblas.h" })) {
        exe.root_module.addSystemIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{sdk_abs}) });
        exe.root_module.linkSystemLibrary("mkl_rt", .{});
    }

    exe.root_module.linkSystemLibrary("dl", .{});

    // Runtime rpaths (match parent project layout).
    const rpaths = [_][]const u8{
        "$ORIGIN",
        "$ORIGIN/../lib",
        "$ORIGIN/../runtime",
        "$ORIGIN/../runtime/xla/pjrt/c",
        "$ORIGIN/../runtime/nvidia/cudnn/lib",
        "$ORIGIN/../runtime/nvidia/cublas/lib",
        "$ORIGIN/../runtime/nvidia/cudart/lib",
        "$ORIGIN/../runtime/nvidia/cufft/lib",
        "$ORIGIN/../runtime/nvidia/cupti/lib",
        "$ORIGIN/../runtime/nvidia/cusparse/lib",
        "$ORIGIN/../runtime/nvidia/nvjitlink/lib",
        "$ORIGIN/../runtime/nvidia/nvrtc/lib",
        "$ORIGIN/../runtime/nvidia/nccl/lib",
        "$ORIGIN/../runtime/nvidia/nvshmem/lib",
        "$ORIGIN/../runtime/sys/lib",
        "/run/opengl-driver/lib",
    };
    inline for (rpaths) |p| exe.root_module.addRPathSpecial(p);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    b.step("run", "Run the benchmark").dependOn(&run_cmd.step);

    // Tests
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "zigrad", .module = zigrad_mod },
        },
    });
    const tests = b.addTest(.{ .root_module = test_mod });
    tests.root_module.addLibraryPath(.{ .cwd_relative = sdk_lib });
    const run_tests = b.addRunArtifact(tests);
    b.step("test", "Run benchmark tests").dependOn(&run_tests.step);
}

fn resolve_absolute(b: *std.Build, path: []const u8) []const u8 {
    if (std.fs.path.isAbsolute(path)) return path;
    const cwd_abs = std.fs.cwd().realpathAlloc(b.allocator, ".") catch @panic("realpathAlloc failed");
    return std.fs.path.join(b.allocator, &.{ cwd_abs, path }) catch @panic("path join failed");
}

fn sdk_has(b: *std.Build, sdk_root_abs: []const u8, sub_path: []const []const u8) bool {
    var parts: [8][]const u8 = undefined;
    parts[0] = sdk_root_abs;
    for (sub_path, 0..) |p, i| parts[i + 1] = p;
    const header_path = std.fs.path.join(b.allocator, parts[0 .. sub_path.len + 1]) catch return false;
    std.fs.accessAbsolute(header_path, .{}) catch return false;
    return true;
}
