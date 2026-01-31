const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Canonical external SDK root containing:
    //   - include/  (mlir-c, stablehlo, xla/pjrt/c)
    //   - lib/      (libMLIR-C.so, libStablehloCAPI.so, LLVM/MLIR deps)
    //   - runtime/  (xla/pjrt/c plugins + bundled CUDA user-space libs)
    const sdk_root = b.option([]const u8, "sdk", "Path to zigrad external SDK root (include/, lib/, runtime/)") orelse "./result";

    // Dev convenience: override runtime bundle root directory.
    // If set, we symlink `zig-out/runtime` to this path.
    const runtime_root_opt = b.option([]const u8, "runtime", "Override runtime bundle root (dev convenience)");

    const sdk_include = b.fmt("{s}/include", .{sdk_root});
    const sdk_lib = b.fmt("{s}/lib", .{sdk_root});
    const sdk_runtime = b.fmt("{s}/runtime", .{sdk_root});

    const safetensors_zg_dep = b.dependency("safetensors_zg", .{});
    const zigrad_mod = b.addModule("zigrad", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    zigrad_mod.addIncludePath(b.path("src"));
    zigrad_mod.addIncludePath(.{ .cwd_relative = sdk_include });

    const exe = b.addExecutable(.{
        .name = "zigrad",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "zigrad", .module = zigrad_mod },
                .{ .name = "safetensors_zg", .module = safetensors_zg_dep.module("safetensors_zg") },
            },
        }),
    });

    exe.root_module.addIncludePath(b.path("src"));
    exe.root_module.addIncludePath(.{ .cwd_relative = sdk_include });
    link_mlir_stablehlo_capi(exe, sdk_lib);
    add_runtime_bundle(b, exe, runtime_root_opt orelse sdk_runtime);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    b.step("run", "Run the v0 demo executable").dependOn(&run_cmd.step);

    const lib_tests = b.addTest(.{ .root_module = zigrad_mod });
    link_mlir_stablehlo_capi(lib_tests, sdk_lib);
    add_runtime_bundle(b, lib_tests, runtime_root_opt orelse sdk_runtime);

    const run_lib_tests = b.addRunArtifact(lib_tests);
    b.step("test", "Run unit tests").dependOn(&run_lib_tests.step);
}

fn link_mlir_stablehlo_capi(exe: *std.Build.Step.Compile, sdk_lib: []const u8) void {
    exe.root_module.addLibraryPath(.{ .cwd_relative = sdk_lib });

    // MLIR + StableHLO were built with libstdc++ ABI.
    exe.root_module.linkSystemLibrary("stdc++", .{});

    // Shared MLIR C boundary (the SDK provides libMLIR-C.so).
    exe.root_module.linkSystemLibrary("MLIR-C", .{});

    // StableHLO C API boundary (libStablehloCAPI.so provides stablehlo dialect handle symbols).
    exe.root_module.linkSystemLibrary("StablehloCAPI", .{});
}

fn add_runtime_bundle(b: *std.Build, exe: *std.Build.Step.Compile, runtime_root: []const u8) void {
    exe.root_module.linkSystemLibrary("dl", .{});

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
        // NixOS host driver injection
        "/run/opengl-driver/lib",
    };
    inline for (rpaths) |p| exe.root_module.addRPathSpecial(p);

    // Dev convenience: symlink `zig-out/runtime` -> runtime_root
    const runtime_root_abs = if (std.fs.path.isAbsolute(runtime_root))
        runtime_root
    else
        std.fs.cwd().realpathAlloc(b.allocator, runtime_root) catch @panic("realpathAlloc failed");
    const link_step = b.addSystemCommand(&[_][]const u8{
        "bash",
        "-lc",
        std.fmt.allocPrint(b.allocator,
            \\set -euo pipefail
            \\prefix="{s}"
            \\mkdir -p "$prefix"
            \\ln -sfn "{s}" "$prefix/runtime"
        , .{ b.install_prefix, runtime_root_abs }) catch @panic("OOM"),
    });
    exe.step.dependOn(&link_step.step);
}
