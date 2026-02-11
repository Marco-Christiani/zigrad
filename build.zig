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

    const tvm_enable_opt = b.option(bool, "tvm", "Enable TVM runtime if present in SDK");
    const tvm_available = sdk_has_tvm(b, sdk_root);
    const tvm_enabled = if (tvm_enable_opt) |v| v else tvm_available;
    if (tvm_enabled and !tvm_available) {
        @panic("-Dtvm requested but TVM not found under SDK (need include/tvm/ffi/c_api.h or include/tvm/runtime/c_runtime_api.h, plus lib/libtvm_runtime.so)");
    }

    const build_options = b.addOptions();
    build_options.addOption(bool, "enable_tvm", tvm_enabled);

    const safetensors_zg_dep = b.dependency("safetensors_zg", .{});
    const zigrad_mod = b.addModule("zigrad", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    zigrad_mod.addOptions("build_options", build_options);
    zigrad_mod.addIncludePath(b.path("src"));
    zigrad_mod.addIncludePath(.{ .cwd_relative = sdk_include });
    zigrad_mod.linkSystemLibrary("mkl_rt", .{});

    // Add CUDA include path if available (needed for nvrtc.h in tvm builds)
    if (tvm_enabled) {
        if (std.posix.getenv("CUDA_HOME")) |cuda_home| {
            const cuda_include = b.fmt("{s}/include", .{cuda_home});
            zigrad_mod.addIncludePath(.{ .cwd_relative = cuda_include });
        }
    }

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
    exe.root_module.addOptions("build_options", build_options);

    exe.root_module.addIncludePath(b.path("src"));
    exe.root_module.addIncludePath(.{ .cwd_relative = sdk_include });
    link_mlir_stablehlo_capi(exe, sdk_lib);
    add_runtime_bundle(b, exe, runtime_root_opt orelse sdk_runtime);
    if (tvm_enabled) {
        link_tvm_runtime(exe, sdk_lib);
    }

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    b.step("run", "Run the v0 demo executable").dependOn(&run_cmd.step);

    const lib_tests = b.addTest(.{ .root_module = zigrad_mod });
    link_mlir_stablehlo_capi(lib_tests, sdk_lib);
    add_runtime_bundle(b, lib_tests, runtime_root_opt orelse sdk_runtime);
    if (tvm_enabled) {
        link_tvm_runtime(lib_tests, sdk_lib);
    }

    const run_lib_tests = b.addRunArtifact(lib_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_tests.step);
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

fn link_tvm_runtime(exe: *std.Build.Step.Compile, tvm_lib: []const u8) void {
    exe.root_module.addLibraryPath(.{ .cwd_relative = tvm_lib });
    exe.root_module.addRPathSpecial(tvm_lib);
    exe.root_module.linkSystemLibrary("tvm_ffi", .{});
    exe.root_module.linkSystemLibrary("tvm_runtime", .{});
    // Full compiler library — registers TE, TIR, codegen packed functions.
    exe.root_module.linkSystemLibrary("tvm", .{});

    // NVRTC for runtime CUDA compilation callback
    if (exe.rootModuleTarget().os.tag == .linux) {
        exe.root_module.linkSystemLibrary("nvrtc", .{});

        // Add CUDA include path for nvrtc.h if CUDA_HOME is set
        if (std.posix.getenv("CUDA_HOME")) |cuda_home| {
            const cuda_include = std.fmt.allocPrint(exe.step.owner.allocator, "{s}/include", .{cuda_home}) catch @panic("OOM");
            exe.root_module.addIncludePath(.{ .cwd_relative = cuda_include });
        }
    }
}

fn sdk_has_tvm(b: *std.Build, sdk_root: []const u8) bool {
    const sdk_root_abs = if (std.fs.path.isAbsolute(sdk_root)) blk: {
        break :blk sdk_root;
    } else blk: {
        const cwd_abs = std.fs.cwd().realpathAlloc(b.allocator, ".") catch return false;
        break :blk std.fs.path.join(b.allocator, &.{ cwd_abs, sdk_root }) catch return false;
    };

    const header_runtime_path = b.pathJoin(&.{ sdk_root_abs, "include", "tvm", "runtime", "c_runtime_api.h" });
    const header_ffi_path = b.pathJoin(&.{ sdk_root_abs, "include", "tvm", "ffi", "c_api.h" });
    const has_header = blk: {
        if (std.fs.accessAbsolute(header_runtime_path, .{})) |_| break :blk true else |_| {}
        if (std.fs.accessAbsolute(header_ffi_path, .{})) |_| break :blk true else |_| {}
        break :blk false;
    };
    if (!has_header) return false;

    const lib_runtime_path = b.pathJoin(&.{ sdk_root_abs, "lib", "libtvm_runtime.so" });
    if (std.fs.accessAbsolute(lib_runtime_path, .{})) |_| {} else |_| return false;

    // v0.22+: runtime depends on libtvm_ffi.so.
    const lib_ffi_path = b.pathJoin(&.{ sdk_root_abs, "lib", "libtvm_ffi.so" });
    if (std.fs.accessAbsolute(lib_ffi_path, .{})) |_| {} else |_| return false;

    return true;
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
    const runtime_root_abs = if (std.fs.path.isAbsolute(runtime_root)) blk: {
        break :blk runtime_root;
    } else blk: {
        const cwd_abs = std.fs.cwd().realpathAlloc(b.allocator, ".") catch @panic("realpathAlloc failed");
        break :blk std.fs.path.join(b.allocator, &.{ cwd_abs, runtime_root }) catch @panic("path join failed");
    };
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
