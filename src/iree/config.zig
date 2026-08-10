//! Pure Zig configuration for the optional IREE integration.

const std = @import("std");

pub const compiler_path_env = "ZG_IREE_COMPILER_PATH";
pub const driver_env = "ZG_IREE_DRIVER";
pub const target_backend_env = "ZG_IREE_TARGET_BACKEND";
pub const temporary_directory_env = "TMPDIR";

pub const default_compiler_path = "iree-compile";
pub const default_driver = "local-sync";
pub const default_target_backend = "vmvx";
pub const default_input_type = "stablehlo";
pub const default_temporary_directory = "/tmp";

/// Inputs for one out-of-process IREE compilation.
pub const CompilerConfig = struct {
    /// Path or executable name for the IREE compiler CLI.
    executable: []const u8 = default_compiler_path,

    /// IREE HAL target backend passed to the compiler.
    target_backend: []const u8 = default_target_backend,

    /// IREE input dialect name passed to the compiler.
    input_type: []const u8 = default_input_type,

    /// Directory used for transient compiler inputs and outputs.
    temporary_directory: []const u8 = default_temporary_directory,

    /// Additional arguments appended to the IREE compiler invocation.
    extra_arguments: []const []const u8 = &.{},
};

/// Inputs for one IREE runtime instance.
pub const RuntimeConfig = struct {
    /// HAL driver used by registered-driver execution.
    driver: []const u8 = default_driver,
};

/// Compiler and runtime configuration for IREE.
pub const Config = struct {
    /// Compiler process configuration.
    compiler: CompilerConfig = .{},

    /// Runtime device configuration.
    runtime: RuntimeConfig = .{},

    /// Resolve optional overrides from an explicit environment map.
    pub fn from_environ(environ: *const std.process.Environ.Map) Config {
        return .{
            .compiler = .{
                .executable = environ.get(compiler_path_env) orelse
                    default_compiler_path,
                .target_backend = environ.get(target_backend_env) orelse
                    default_target_backend,
                .temporary_directory = environ.get(temporary_directory_env) orelse
                    default_temporary_directory,
            },
            .runtime = .{
                .driver = environ.get(driver_env) orelse default_driver,
            },
        };
    }
};

test "Config resolves IREE application inputs" {
    var environ: std.process.Environ.Map = .init(std.testing.allocator);
    defer environ.deinit();

    const defaults = Config.from_environ(&environ);
    try std.testing.expectEqualStrings(default_compiler_path, defaults.compiler.executable);
    try std.testing.expectEqualStrings(default_driver, defaults.runtime.driver);
    try std.testing.expectEqualStrings(default_target_backend, defaults.compiler.target_backend);
    try std.testing.expectEqualStrings(default_temporary_directory, defaults.compiler.temporary_directory);

    try environ.put(compiler_path_env, "/runtime/bin/iree-compile");
    try environ.put(driver_env, "local-task");
    try environ.put(target_backend_env, "llvm-cpu");
    try environ.put(temporary_directory_env, "/work/tmp");

    const configured = Config.from_environ(&environ);
    try std.testing.expectEqualStrings("/runtime/bin/iree-compile", configured.compiler.executable);
    try std.testing.expectEqualStrings("local-task", configured.runtime.driver);
    try std.testing.expectEqualStrings("llvm-cpu", configured.compiler.target_backend);
    try std.testing.expectEqualStrings("/work/tmp", configured.compiler.temporary_directory);
}
