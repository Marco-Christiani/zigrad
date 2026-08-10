const std = @import("std");

const Mode = enum {
    emit_pr,
    run,
};

const Runtime = enum {
    embedded_elf_sync,
    registered,
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const strip = b.option(bool, "strip", "Omit debug information from the executable");
    const mode = b.option(Mode, "mode", "Select PR emission or embedded execution") orelse .run;
    const runtime = b.option(Runtime, "runtime", "Select the IREE device construction path") orelse .embedded_elf_sync;
    const driver = b.option([]const u8, "driver", "Select the registered IREE HAL driver") orelse "local-sync";
    const with_pjrt = b.option(bool, "pjrt", "Enable PJRT support") orelse false;
    const with_mlir = b.option(bool, "mlir", "Enable MLIR support") orelse false;
    const with_tvm = b.option(bool, "tvm", "Enable TVM support") orelse false;
    const with_mirage = b.option(bool, "mirage", "Enable Mirage support") orelse false;
    const with_iree = b.option(bool, "iree", "Enable IREE support") orelse (mode == .run);
    const with_nvrtc = b.option(bool, "nvrtc", "Enable NVRTC support") orelse false;
    const with_cuda_runtime = b.option(bool, "cuda-runtime", "Enable CUDA runtime support") orelse false;

    const sdk_option = b.option([]const u8, "sdk", "Path to the target IREE runtime inputs");
    const sdk = if (mode == .run)
        sdk_option orelse
            b.graph.environ_map.get("ZG_EXTERNAL_SDK_ROOT") orelse
            std.debug.panic("embedded execution requires -Dsdk or ZG_EXTERNAL_SDK_ROOT", .{})
    else
        sdk_option;

    const zigrad = b.dependency("zigrad", .{
        .target = target,
        .optimize = optimize,
        .sdk = sdk,
        .pjrt = with_pjrt,
        .mlir = with_mlir,
        .tvm = with_tvm,
        .mirage = with_mirage,
        .iree = with_iree,
        .@"iree-embedded-elf" = runtime == .embedded_elf_sync,
        .nvrtc = with_nvrtc,
        .@"cuda-runtime" = with_cuda_runtime,
    });

    const executable = b.addExecutable(.{
        .name = switch (mode) {
            .emit_pr => "emit-pr",
            .run => "basic-deployment",
        },
        .root_module = b.createModule(.{
            .root_source_file = b.path(switch (mode) {
                .emit_pr => "src/emit_pr.zig",
                .run => "src/main.zig",
            }),
            .target = target,
            .optimize = optimize,
            .strip = strip,
            .link_libc = mode == .run,
            .imports = &.{.{ .name = "zigrad", .module = zigrad.module("zigrad") }},
        }),
    });
    const build_options = b.addOptions();
    build_options.addOption(Runtime, "runtime", runtime);
    build_options.addOption([]const u8, "driver", driver);
    executable.root_module.addOptions("build_options", build_options);

    b.installArtifact(executable);
}
