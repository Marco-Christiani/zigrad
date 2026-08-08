const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const sdk_root = b.option([]const u8, "sdk", "Path to zigrad external SDK root (include/, lib/, runtime/)") orelse
        b.graph.environ_map.get("ZG_EXTERNAL_SDK_ROOT") orelse
        std.debug.panic("the benchmark requires -Dsdk or ZG_EXTERNAL_SDK_ROOT", .{});

    // Resolve to absolute so the path works from both the example and zigrad contexts.
    const sdk_abs = resolve_absolute(b, sdk_root);
    const sdk_lib = b.fmt("{s}/lib", .{sdk_abs});

    const zigrad_dep = b.dependency("zigrad", .{
        .target = target,
        .optimize = optimize,
        .sdk = @as([]const u8, sdk_abs),
        .pjrt = true,
        .mlir = true,
        .iree = true,
        .tvm = true,
        .nvrtc = true,
        .@"cuda-runtime" = true,
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

    const has_mkl = sdk_has(b, sdk_abs, &.{ "include", "mkl_cblas.h" });
    const build_options = b.addOptions();
    build_options.addOption(bool, "has_mkl", has_mkl);
    exe.root_module.addOptions("build_options", build_options);

    // SDK library linking. Taken from zigrad.
    exe.root_module.addLibraryPath(.{ .cwd_relative = sdk_lib });
    if (sdk_has(b, sdk_abs, &.{ "include", "mlir-c", "IR.h" })) {
        exe.root_module.linkSystemLibrary("stdc++", .{});
        exe.root_module.linkSystemLibrary("MLIR-C", .{});
        exe.root_module.linkSystemLibrary("StablehloCAPI", .{});
    }
    if (has_mkl) {
        exe.root_module.addSystemIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{sdk_abs}) });
        exe.root_module.linkSystemLibrary("mkl_rt", .{});
    }

    exe.root_module.linkSystemLibrary("dl", .{});

    // Runtime rpaths (match parent project layout).
    for (zigrad_mod.rpaths.items) |p| exe.root_module.addRPathSpecial(p.special);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    b.step("run", "Run the benchmark").dependOn(&run_cmd.step);

    // Tests.
    const tests = b.addTest(.{ .root_module = exe.root_module });
    tests.root_module.addLibraryPath(.{ .cwd_relative = sdk_lib });
    const run_tests = b.addRunArtifact(tests);
    b.step("test", "Run benchmark tests").dependOn(&run_tests.step);
}

fn resolve_absolute(b: *std.Build, path: []const u8) []const u8 {
    if (std.fs.path.isAbsolute(path)) return path;
    const build_root = b.build_root.path orelse ".";
    return std.fs.path.join(b.allocator, &.{ build_root, path }) catch @panic("path join failed");
}

fn sdk_has(b: *std.Build, sdk_root_abs: []const u8, sub_path: []const []const u8) bool {
    var parts: [8][]const u8 = undefined;
    parts[0] = sdk_root_abs;
    for (sub_path, 0..) |p, i| parts[i + 1] = p;
    const header_path = std.fs.path.join(b.allocator, parts[0 .. sub_path.len + 1]) catch return false;
    std.Io.Dir.cwd().access(b.graph.io, header_path, .{}) catch return false;
    return true;
}
