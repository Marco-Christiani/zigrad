const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const mode = b.option(Mode, "mode", "Select initialization, training, PR emission, or inference") orelse .init;
    const strip = b.option(bool, "strip", "Omit debug information from the executable");
    const with_pjrt = b.option(bool, "pjrt", "Enable PJRT support") orelse (mode == .train);
    const with_mlir = b.option(bool, "mlir", "Enable MLIR support") orelse (mode == .train);
    const with_iree = b.option(bool, "iree", "Enable IREE support") orelse (mode == .infer);
    const with_tvm = b.option(bool, "tvm", "Enable TVM support") orelse false;
    const with_mirage = b.option(bool, "mirage", "Enable Mirage support") orelse false;
    const with_nvrtc = b.option(bool, "nvrtc", "Enable NVRTC support") orelse false;
    const with_cuda_runtime = b.option(bool, "cuda-runtime", "Enable CUDA runtime support") orelse false;
    const runtime = b.option(Runtime, "runtime", "Select the IREE device construction path") orelse .embedded_elf_sync;
    const driver = b.option([]const u8, "driver", "Select the registered IREE HAL driver") orelse "local-sync";

    const sdk_option = b.option([]const u8, "sdk", "Path to Zigrad external inputs");
    const has_external_integration = with_pjrt or with_mlir or with_iree or with_tvm or with_mirage or with_nvrtc or with_cuda_runtime;
    const sdk = if (has_external_integration)
        sdk_option orelse b.graph.environ_map.get("ZG_EXTERNAL_SDK_ROOT") orelse
            std.debug.panic("this mode requires -Dsdk or ZG_EXTERNAL_SDK_ROOT", .{})
    else
        sdk_option;

    const zigrad = b.dependency("zigrad", .{
        .target = target,
        .optimize = optimize,
        .sdk = sdk,
        .pjrt = with_pjrt,
        .mlir = with_mlir,
        .iree = with_iree,
        .@"iree-embedded-elf" = with_iree and runtime == .embedded_elf_sync,
        .tvm = with_tvm,
        .mirage = with_mirage,
        .nvrtc = with_nvrtc,
        .@"cuda-runtime" = with_cuda_runtime,
    });

    const executable = b.addExecutable(.{
        .name = switch (mode) {
            .init => "cifar10-init",
            .train => "cifar10-train",
            .emit_pr => "cifar10-emit-pr",
            .infer => "cifar10-infer",
        },
        .root_module = b.createModule(.{
            .root_source_file = b.path(switch (mode) {
                .init => "src/init.zig",
                .train => "src/train.zig",
                .emit_pr => "src/emit_pr.zig",
                .infer => "src/infer.zig",
            }),
            .target = target,
            .optimize = optimize,
            .strip = strip,
            .link_libc = mode == .infer,
            .imports = &.{.{ .name = "zigrad", .module = zigrad.module("zigrad") }},
        }),
    });
    const options = b.addOptions();
    options.addOption(Runtime, "runtime", runtime);
    options.addOption([]const u8, "driver", driver);
    executable.root_module.addOptions("build_options", options);
    b.installArtifact(executable);

    const core_zigrad = b.dependency("zigrad", .{
        .target = target,
        .optimize = optimize,
    });
    const model_tests = b.addTest(.{
        .name = "cifar10-model-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/model.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "zigrad", .module = core_zigrad.module("zigrad") }},
        }),
    });
    const run_model_tests = b.addRunArtifact(model_tests);
    b.step("test", "Trace inference and training programs").dependOn(&run_model_tests.step);
    const install_model_tests = b.addInstallArtifact(model_tests, .{});
    b.step("test-compile", "Build model tests without running").dependOn(&install_model_tests.step);
}

const Mode = enum { init, train, emit_pr, infer };
const Runtime = enum { embedded_elf_sync, registered };
