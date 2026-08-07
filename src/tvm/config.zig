//! Pure Zig configuration for the optional TVM integration.

const std = @import("std");
const cuda_nvrtc = @import("../cuda/nvrtc.zig");
const device = @import("../device.zig");
const runtime = @import("../runtime.zig");
const linker = @import("../toolchain/linker.zig");

pub const ffi_path_env = "ZG_TVM_FFI_PATH";
pub const runtime_path_env = "ZG_TVM_RUNTIME_PATH";
pub const compiler_path_env = "ZG_TVM_COMPILER_PATH";
pub const default_ffi_path = "libtvm_ffi.so";
pub const default_runtime_path = "libtvm_runtime.so";
pub const default_compiler_path = "libtvm.so";
pub const CompileConfigError = cuda_nvrtc.ConfigError || linker.ConfigError;

/// Current TVM compilation target.
pub const TargetKind = enum {
    cpu,
    cuda,

    /// Return whether this compiler target can consume the selected device.
    pub fn accepts(self: TargetKind, selected: device.Device) bool {
        return switch (self) {
            .cpu => selected.platform.eql(.cpu),
            .cuda => selected.platform.eql(.cuda),
        };
    }
};

/// Maximum TVM capability available to one process.
pub const RuntimeSurface = enum {
    ffi,
    runtime,
    compiler,

    /// Return whether this surface satisfies an operation's requirement.
    pub fn satisfies(self: RuntimeSurface, requirement: RuntimeSurface) bool {
        return @intFromEnum(self) >= @intFromEnum(requirement);
    }
};

/// Runtime-loaded TVM libraries.
pub const RuntimeConfig = struct {
    /// Maximum TVM capability selected before the first library load.
    surface: RuntimeSurface,

    /// TVM FFI library that owns the packed-call ABI.
    ffi: runtime.RuntimeLibrary = .{ .path = default_ffi_path },

    /// TVM deployment library that executes generated modules.
    runtime: runtime.RuntimeLibrary = .{ .path = default_runtime_path },

    /// TVM compiler library that registers compilation functions.
    compiler: runtime.RuntimeLibrary = .{ .path = default_compiler_path },

    /// Resolve optional library overrides from an explicit environment map.
    pub fn from_environ(
        environ: *const std.process.Environ.Map,
        surface: RuntimeSurface,
    ) RuntimeConfig {
        return .{
            .surface = surface,
            .ffi = .from_environ(environ, ffi_path_env, default_ffi_path),
            .runtime = .from_environ(
                environ,
                runtime_path_env,
                default_runtime_path,
            ),
            .compiler = .from_environ(
                environ,
                compiler_path_env,
                default_compiler_path,
            ),
        };
    }
};

/// Target-specific inputs used while TVM compiles candidates.
pub const CompileConfig = struct {
    /// Target selected for this compilation policy.
    target: TargetKind,

    /// Linker used to produce candidate shared libraries.
    linker: linker.Linker,

    /// NVRTC inputs required only by the CUDA target.
    nvrtc: ?cuda_nvrtc.Config = null,

    /// Resolve target-specific inputs from an explicit environment map.
    pub fn from_environ(
        environ: *const std.process.Environ.Map,
        target: TargetKind,
    ) CompileConfigError!CompileConfig {
        return .{
            .target = target,
            .linker = try .from_environ(environ),
            .nvrtc = switch (target) {
                .cpu => null,
                .cuda => try .from_environ(environ),
            },
        };
    }
};

test "RuntimeConfig resolves TVM library paths" {
    var environ: std.process.Environ.Map = .init(std.testing.allocator);
    defer environ.deinit();

    const defaults = RuntimeConfig.from_environ(&environ, .runtime);
    try std.testing.expectEqual(RuntimeSurface.runtime, defaults.surface);
    try std.testing.expectEqualStrings(default_ffi_path, defaults.ffi.path);
    try std.testing.expectEqualStrings(default_runtime_path, defaults.runtime.path);
    try std.testing.expectEqualStrings(default_compiler_path, defaults.compiler.path);

    try environ.put(ffi_path_env, "/runtime/libtvm_ffi.so");
    try environ.put(runtime_path_env, "/runtime/libtvm_runtime.so");
    try environ.put(compiler_path_env, "/runtime/libtvm.so");
    const configured = RuntimeConfig.from_environ(&environ, .compiler);
    try std.testing.expectEqual(RuntimeSurface.compiler, configured.surface);
    try std.testing.expectEqualStrings("/runtime/libtvm_ffi.so", configured.ffi.path);
    try std.testing.expectEqualStrings("/runtime/libtvm_runtime.so", configured.runtime.path);
    try std.testing.expectEqualStrings("/runtime/libtvm.so", configured.compiler.path);
}

test "RuntimeSurface satisfies weaker operation requirements" {
    try std.testing.expect(RuntimeSurface.compiler.satisfies(.runtime));
    try std.testing.expect(RuntimeSurface.runtime.satisfies(.ffi));
    try std.testing.expect(!RuntimeSurface.runtime.satisfies(.compiler));
    try std.testing.expect(!RuntimeSurface.ffi.satisfies(.runtime));
}

test "CompileConfig resolves NVRTC host inputs only for CUDA" {
    var environ: std.process.Environ.Map = .init(std.testing.allocator);
    defer environ.deinit();

    try std.testing.expectError(
        error.MissingLinker,
        CompileConfig.from_environ(&environ, .cpu),
    );
    try environ.put(linker.path_env, "/toolchain/bin/ld.lld");
    const cpu = try CompileConfig.from_environ(&environ, .cpu);
    try std.testing.expectEqual(TargetKind.cpu, cpu.target);
    try std.testing.expectEqualStrings("/toolchain/bin/ld.lld", cpu.linker.executable);
    try std.testing.expect(cpu.nvrtc == null);

    try std.testing.expectError(
        error.MissingCudaToolkitRoot,
        CompileConfig.from_environ(&environ, .cuda),
    );
    try environ.put(cuda_nvrtc.cuda_home_env, "/cuda");
    const cuda = try CompileConfig.from_environ(&environ, .cuda);
    try std.testing.expect(cuda.nvrtc != null);
}
