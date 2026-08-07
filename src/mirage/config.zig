const std = @import("std");
const cuda_nvrtc = @import("../cuda/nvrtc.zig");
const runtime = @import("../runtime.zig");

pub const adapter_path_env = "ZG_MIRAGE_ADAPTER_PATH";
pub const default_adapter_path = "libzigrad_mirage.so";
pub const CompileConfigError = cuda_nvrtc.ConfigError || error{MissingExternalSdkRoot};

/// Runtime inputs for the Zigrad Mirage adapter.
pub const RuntimeConfig = struct {
    adapter: runtime.RuntimeLibrary = .{ .path = default_adapter_path },

    /// Resolve the optional adapter override from an environment map.
    pub fn from_environ(environ: *const std.process.Environ.Map) RuntimeConfig {
        return .{
            .adapter = .from_environ(
                environ,
                adapter_path_env,
                default_adapter_path,
            ),
        };
    }
};

/// Inputs required to compile Mirage-generated CUDA source with NVRTC.
///
/// String fields are borrowed and must outlive any dispatch state that stores
///  this configuration.
pub const CompileConfig = struct {
    /// Root of the external compile inputs.
    sdk_root: []const u8,

    /// Shared CUDA toolkit and host include policy.
    cuda: cuda_nvrtc.Config,

    /// Optional include root containing Mirage and Cutlass headers.
    mirage_include_dir: ?[]const u8 = null,

    /// Resolve compile inputs from an explicit environment map.
    pub fn from_environ(
        environ: *const std.process.Environ.Map,
    ) CompileConfigError!CompileConfig {
        const sdk_root = environ.get("ZG_EXTERNAL_SDK_ROOT") orelse
            return error.MissingExternalSdkRoot;
        return .{
            .sdk_root = sdk_root,
            .cuda = try .from_environ(environ),
            .mirage_include_dir = environ.get("ZG_MIRAGE_INCLUDE_DIR"),
        };
    }
};

/// Complete application configuration for the current Mirage integration.
pub const Config = struct {
    runtime: RuntimeConfig,
    compile: CompileConfig,

    /// Resolve runtime loading and CUDA compilation from one environment map.
    pub fn from_environ(
        environ: *const std.process.Environ.Map,
    ) CompileConfigError!Config {
        return .{
            .runtime = .from_environ(environ),
            .compile = try .from_environ(environ),
        };
    }
};

test "RuntimeConfig resolves the Mirage adapter path" {
    var environ: std.process.Environ.Map = .init(std.testing.allocator);
    defer environ.deinit();

    try std.testing.expectEqualStrings(
        default_adapter_path,
        RuntimeConfig.from_environ(&environ).adapter.path,
    );

    try environ.put(adapter_path_env, "/runtime/libzigrad_mirage.so");
    try std.testing.expectEqualStrings(
        "/runtime/libzigrad_mirage.so",
        RuntimeConfig.from_environ(&environ).adapter.path,
    );
}

test "CompileConfig requires an external SDK root" {
    var environ: std.process.Environ.Map = .init(std.testing.allocator);
    defer environ.deinit();

    try std.testing.expectError(
        error.MissingExternalSdkRoot,
        CompileConfig.from_environ(&environ),
    );
}

test "CompileConfig requires the CUDA toolkit root" {
    var environ: std.process.Environ.Map = .init(std.testing.allocator);
    defer environ.deinit();

    try environ.put("ZG_EXTERNAL_SDK_ROOT", "/external");
    try std.testing.expectError(
        error.MissingCudaToolkitRoot,
        CompileConfig.from_environ(&environ),
    );
}
