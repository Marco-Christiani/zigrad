const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Canonical SDK root containing:
    //   - include/  (mlir-c, stablehlo, xla/pjrt/c)
    //   - lib/      (libMLIRCAPI*.a, libStablehloCAPI.a)
    //   - runtime/  (jax_plugins/, nvidia/, sys/lib/, PROVENANCE.json)
    //
    // E.g.,
    //   nix build .#zigrad-pjrt-sdk --print-out-paths
    //   zig build -Dsdk=/nix/store/...-zigrad-pjrt-sdk
    const sdk_root = b.option([]const u8, "sdk", "Path to zigrad PJRT SDK root (include/, lib/, runtime/)") orelse "./result";

    // Optional override for an already-built runtime bundle root directory.
    // If set, we symlink zig-out/runtime -> this path (dev convenience).
    const runtime_root_opt = b.option([]const u8, "runtime", "Override runtime bundle root (dev convenience)");

    const sdk_include = b.fmt("{s}/include", .{sdk_root});
    const sdk_lib = b.fmt("{s}/lib", .{sdk_root});
    const sdk_runtime = b.fmt("{s}/runtime", .{sdk_root});

    // Library module
    const zigrad_mod = b.addModule("zigrad", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        // dlopen + friends
        .link_libc = true,
    });

    // Project-local headers (if any)
    zigrad_mod.addIncludePath(b.path("src"));

    // SDK headers:
    //   - mlir-c/...
    //   - stablehlo/integrations/c/...
    //   - xla/pjrt/c/...
    zigrad_mod.addIncludePath(.{ .cwd_relative = sdk_include });

    // ---------------------------------------------------------------------------------------------
    // M1
    const exe = b.addExecutable(.{
        .name = "zigrad-pjrt-m1",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "zigrad", .module = zigrad_mod }},
        }),
    });
    exe.linkLibC();
    exe.root_module.addIncludePath(b.path("src"));
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    b.step("run", "Run the M1 test executable").dependOn(&run_cmd.step);

    // ---------------------------------------------------------------------------------------------
    // M2 (matmul)
    const exe_m2 = b.addExecutable(.{
        .name = "zigrad-pjrt-m2",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/m2_matmul.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "zigrad", .module = zigrad_mod }},
        }),
    });
    exe_m2.root_module.addIncludePath(b.path("src"));

    // shouldnt need these two lines
    exe_m2.root_module.addIncludePath(.{ .cwd_relative = sdk_include });
    linkMlirStablehloCapi(b, exe_m2, sdk_lib);

    addRuntimeBundle(b, exe_m2, runtime_root_opt orelse sdk_runtime);
    b.installArtifact(exe_m2);

    const run_m2_cmd = b.addRunArtifact(exe_m2);
    run_m2_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_m2_cmd.addArgs(args);
    b.step("run-m2", "Run the M2 test executable").dependOn(&run_m2_cmd.step);

    // ---------------------------------------------------------------------------------------------
    // M2 fusion proof
    const exe_m2_fusion = b.addExecutable(.{
        .name = "zigrad-pjrt-m2-fusion",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/m2_fusion.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            // NOTE: SDK artifacts were built against libstdc++ ABI, we will NOT link cpp with this flag. See linkMlirStablehloCapi().
            // .link_libcpp = true,
            .imports = &.{.{ .name = "zigrad", .module = zigrad_mod }},
        }),
    });
    exe_m2_fusion.root_module.addIncludePath(b.path("src"));
    addRuntimeBundle(b, exe_m2_fusion, runtime_root_opt orelse sdk_runtime);
    b.installArtifact(exe_m2_fusion);

    const run_m2_fusion_cmd = b.addRunArtifact(exe_m2_fusion);
    run_m2_fusion_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_m2_fusion_cmd.addArgs(args);
    b.step("run-m2-fusion", "Run the M2 fusion executable").dependOn(&run_m2_fusion_cmd.step);

    // ---------------------------------------------------------------------------------------------
    // M4 (in-memory MLIR construction) - needs MLIR CAPI + StableHLO CAPI libs
    const exe_m4 = b.addExecutable(.{
        .name = "zigrad-pjrt-m4",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/m4_matmul.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            // NOTE: SDK artifacts were built against libstdc++ ABI, we will NOT link cpp with this flag. See linkMlirStablehloCapi().
            // .link_libcpp = true,
            .imports = &.{.{ .name = "zigrad", .module = zigrad_mod }},
        }),
    });

    exe_m4.root_module.addIncludePath(b.path("src"));
    exe_m4.root_module.addIncludePath(.{ .cwd_relative = sdk_include });

    linkMlirStablehloCapi(b, exe_m4, sdk_lib);

    addRuntimeBundle(b, exe_m4, runtime_root_opt orelse sdk_runtime);
    b.installArtifact(exe_m4);

    const run_m4_cmd = b.addRunArtifact(exe_m4);
    run_m4_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_m4_cmd.addArgs(args);
    b.step("run-m4", "Run the M4 test executable").dependOn(&run_m4_cmd.step);

    // ---------------------------------------------------------------------------------------------
    // M4.2 (custom call boundaries)
    const exe_m4_custom = b.addExecutable(.{
        .name = "zigrad-pjrt-m4-custom",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/m4_custom_call.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            // NOTE: SDK artifacts were built against libstdc++ ABI, we will NOT link cpp with this flag. See linkMlirStablehloCapi().
            // .link_libcpp = true,
            .imports = &.{.{ .name = "zigrad", .module = zigrad_mod }},
        }),
    });

    exe_m4_custom.root_module.addIncludePath(b.path("src"));
    exe_m4_custom.root_module.addIncludePath(.{ .cwd_relative = sdk_include });

    linkMlirStablehloCapi(b, exe_m4_custom, sdk_lib);

    addRuntimeBundle(b, exe_m4_custom, runtime_root_opt orelse sdk_runtime);
    b.installArtifact(exe_m4_custom);

    const run_m4_custom_cmd = b.addRunArtifact(exe_m4_custom);
    run_m4_custom_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_m4_custom_cmd.addArgs(args);
    b.step("run-m4-custom", "Run the M4.2 custom call test").dependOn(&run_m4_custom_cmd.step);

    // ---------------------------------------------------------------------------------------------
    // Unit tests
    const lib_tests = b.addTest(.{ .root_module = zigrad_mod });
    const run_lib_tests = b.addRunArtifact(lib_tests);
    b.step("test", "Run unit tests").dependOn(&run_lib_tests.step);
}

fn linkMlirStablehloCapi(b: *std.Build, exe: *std.Build.Step.Compile, sdk_lib: []const u8) void {
    _ = b;
    exe.root_module.addLibraryPath(.{ .cwd_relative = sdk_lib });

    // Shared MLIR C boundary (we ensured libMLIR-C.so symlink exists in the SDK)
    exe.root_module.linkSystemLibrary("MLIR-C", .{});

    // StableHLO C API boundary (shared library). This provides the StableHLO dialect handle symbol:
    //   mlirGetDialectHandle__stablehlo__()
    // and depends on the MLIR/LLVM DSOs shipped in the SDK lib/.
    exe.root_module.linkSystemLibrary("StablehloCAPI", .{});

    // C++ runtime: MLIR/StableHLO were built with libstdc++ ABI.
    exe.root_module.linkSystemLibrary("stdc++", .{});
}

// OLD -- Kept for reference as we are still determining how to fix the build so dialect registration works
// fn linkMlirStablehloCapi(b: *std.Build, exe: *std.Build.Step.Compile, sdk_lib: []const u8) void {
//     // const libs = [_][]const u8{
//     //     "libMLIRCAPIIR.a",
//     //     "libMLIRCAPIArith.a",
//     //     "libMLIRCAPIMath.a",
//     //     "libMLIRCAPISCF.a",
//     //     "libMLIRCAPITransforms.a",
//     //     "libMLIRCAPIFunc.a",
//     //     "libMLIRCAPITensor.a",
//     //     "libStablehloCAPI.a",
//     // };
//     //
//     // inline for (libs) |name| {
//     //     const full = b.fmt("{s}/{s}", .{ sdk_lib, name });
//     //     exe.addObjectFile(.{ .cwd_relative = full });
//     // }
//
//     // Prefer shared libs. This avoids manually enumerating the huge static dependency closure
//     // behind the MLIR C API wrappers.
//     exe.addLibraryPath(.{ .cwd_relative = sdk_lib });
//
//     // libMLIR-C.so
//     exe.linkSystemLibrary("MLIR-C");
//
//     // StableHLO C API may be static or shared depending on build; handle both.
//     // If the derivation produces libStablehloCAPI.so, this works.
//     // If it produces only libStablehloCAPI.a, we add it as an object file.
//     exe.linkSystemLibrary("StablehloCAPI");
//
//     _ = b;
// }

fn addRuntimeBundle(b: *std.Build, exe: *std.Build.Step.Compile, runtime_root_opt: ?[]const u8) void {
    // dlopen is used by PJRT loader code.
    exe.root_module.linkSystemLibrary("dl", .{});

    const rpaths = [_][]const u8{
        // assume layout is $prefix/bin/zigrad and $prefix/runtime/...
        "$ORIGIN",
        // MLIR/StableHLO DSOs
        "$ORIGIN/../lib",
        // PJRT plugin + CUDA bundle
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
        // nixos
        "/run/opengl-driver/lib",
    };
    inline for (rpaths) |p| exe.root_module.addRPathSpecial(p);
    // Dev convenience (to avoid copying gb every build): symlink zig-out/runtime -> runtime_root (if provided)
    if (runtime_root_opt) |runtime_root| {
        const runtime_root_abs = if (std.fs.path.isAbsolute(runtime_root))
            runtime_root
        else
            std.fs.cwd().realpathAlloc(b.allocator, runtime_root) catch @panic("realpathAlloc failed");
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
            , .{ b.install_prefix, runtime_root_abs }) catch @panic("OOM"),
        });

        // run after install so prefix exists
        // const install_step = b.getInstallStep();

        // link_step.step.dependOn(install_step);
        exe.step.dependOn(&link_step.step);
    }
}
