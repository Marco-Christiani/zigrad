const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Optional path to an already-built runtime bundle root directory.
    // E.g., zig build -Druntime=/nix/store/...-pjrt-cuda-bundle/runtime
    const runtime_root_opt = b.option([]const u8, "runtime", "Path to PJRT runtime bundle root");

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

    // MLIR C API support (for M4+)
    // Using MLIR static libraries from ZML's bazel build (all 66 libraries)
    const zml_mlir_path = "reference/zml/bazel-bin/external/+llvm+llvm-project/mlir";
    const zml_stablehlo_path = "reference/zml/bazel-bin/external/+xla+stablehlo";

    // Link ALL MLIR static libraries (sledgehammer approach - links all 66 .a files)
    const mlir_libs_dir = std.fs.cwd().openDir(zml_mlir_path, .{ .iterate = true }) catch @panic("Failed to open MLIR libs dir");
    var mlir_libs_iter = mlir_libs_dir.iterate();
    while (mlir_libs_iter.next() catch null) |entry| {
        if (std.mem.endsWith(u8, entry.name, ".a")) {
            const full_path = b.fmt("{s}/{s}", .{ zml_mlir_path, entry.name });
            zigrad_mod.addObjectFile(.{ .cwd_relative = full_path });
        }
    }

    // Link StableHLO C API
    zigrad_mod.addObjectFile(b.path(zml_stablehlo_path ++ "/libstablehlo_dialect_capi.a"));

    // Link LLVM libraries from ZML's bazel build (needed for MLIR dependencies)
    const zml_llvm_path = "reference/zml/bazel-bin/external/+llvm+llvm-project/llvm";
    zigrad_mod.addObjectFile(b.path(zml_llvm_path ++ "/libSupport.a"));
    zigrad_mod.addObjectFile(b.path(zml_llvm_path ++ "/libDemangle.a"));

    // Add vendored MLIR and StableHLO C headers (extracted from ZML's bazel build)
    zigrad_mod.addIncludePath(b.path("vendor"));

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

    // M2 fusion proof executable
    const exe_m2_fusion = b.addExecutable(.{
        .name = "zigrad-pjrt-m2-fusion",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/m2_fusion.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .link_libcpp = true,
            .imports = &.{
                .{ .name = "zigrad", .module = zigrad_mod },
            },
        }),
    });

    exe_m2_fusion.root_module.addIncludePath(b.path("src"));
    addRuntimeBundle(b, exe_m2_fusion, runtime_root_opt);
    b.installArtifact(exe_m2_fusion);

    const run_m2_fusion_cmd = b.addRunArtifact(exe_m2_fusion);
    run_m2_fusion_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_m2_fusion_cmd.addArgs(args);
    }

    const run_m2_fusion_step = b.step("run-m2-fusion", "Run the M2 fusion executable");
    run_m2_fusion_step.dependOn(&run_m2_fusion_cmd.step);

    // M4 test executable (in-memory MLIR construction)
    const exe_m4 = b.addExecutable(.{
        .name = "zigrad-pjrt-m4",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/m4_matmul.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .link_libcpp = true,
            .imports = &.{
                .{ .name = "zigrad", .module = zigrad_mod },
            },
        }),
    });

    exe_m4.root_module.addIncludePath(b.path("src"));
    exe_m4.root_module.addIncludePath(b.path("vendor"));  // MLIR/StableHLO headers
    addRuntimeBundle(b, exe_m4, runtime_root_opt);
    b.installArtifact(exe_m4);

    const run_m4_cmd = b.addRunArtifact(exe_m4);
    run_m4_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_m4_cmd.addArgs(args);
    }

    const run_m4_step = b.step("run-m4", "Run the M4 test executable");
    run_m4_step.dependOn(&run_m4_cmd.step);

    // Unit tests
    const lib_tests = b.addTest(.{
        .root_module = zigrad_mod,
    });

    const run_lib_tests = b.addRunArtifact(lib_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_tests.step);
}

fn addRuntimeBundle(b: *std.Build, exe: *std.Build.Step.Compile, runtime_root_opt: ?[]const u8) void {
    const rpaths = [_][]const u8{
        // assume layout is $prefix/bin/zigrad and $prefix/runtime/...
        "$ORIGIN",
        "$ORIGIN/../runtime",
        "$ORIGIN/../runtime/jax_plugins/xla_cuda13",
        "$ORIGIN/../runtime/nvidia/cudnn/lib",
        "$ORIGIN/../runtime/nvidia/cu13/lib",
        "$ORIGIN/../runtime/nvidia/cublas/lib",
        "$ORIGIN/../runtime/nvidia/nccl/lib",
        "$ORIGIN/../runtime/nvidia/nvshmem/lib",
        "$ORIGIN/../runtime/nvidia/cuda_nvrtc/lib",
        "$ORIGIN/../runtime/sys/lib",
        // need to be careful about doing this due to re-entry and glibc conflicts this can create (same is true for LD_PRELOAD)
        // "/lib/x86_64-linux-gnu",
        // "/usr/lib/x86_64-linux-gnu",
        // // if we get here, I didnt plan for that, sorry
        // "/lib64",
        // "/usr/lib64",
        // "/lib",
        // "/usr/lib",
    };
    inline for (rpaths) |p| exe.root_module.addRPathSpecial(p);
    // Dev convenience (to avoid copying gb every build): symlink zig-out/runtime -> runtime_root (if provided)
    if (runtime_root_opt) |runtime_root| {
        const link_step = b.addSystemCommand(&[_][]const u8{
            "bash",
            "-lc",
            // make sure prefix exists then update symlink
            // ee create sibling runtime at that same install prefix as everything else (zig-out/ by default)
            std.fmt.allocPrint(b.allocator,
                \\set -euo pipefail
                \\prefix="{s}"
                \\mkdir -p "$prefix"
                \\ln -sfn "{s}" "$prefix/runtime"
            , .{ b.install_prefix, runtime_root }) catch @panic("OOM"),
        });

        // run after install so prefix exists
        // const install_step = b.getInstallStep();

        // link_step.step.dependOn(install_step);
        exe.step.dependOn(&link_step.step);
    }
}
