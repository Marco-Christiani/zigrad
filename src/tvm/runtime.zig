//! Process-wide TVM runtime-library lifecycle.

const std = @import("std");

const dylib = @import("../c/dylib.zig");
const api = @import("../c/tvm/api.zig");
const raw = @import("../c/tvm/c.zig");
pub const Config = @import("config.zig").RuntimeConfig;
pub const Surface = @import("config.zig").RuntimeSurface;

const log = std.log.scoped(.@"zg/tvm_runtime_loader");

/// Errors returned when configuration changes after runtime initialization.
pub const ConfigureError = error{
    RuntimeAlreadyInitialized,
};

/// Stable runtime errors exposed by the TVM integration.
pub const Error = error{
    TvmCallFailed,
    TvmLoadFailed,
    TvmFunctionNotFound,
    TvmNotConfigured,
    UnexpectedTvmType,
    TvmSurfaceUnavailable,
    OutOfMemory,
};

const State = struct {
    ffi: dylib.Library,
    surface: ?dylib.Library = null,
};

var config: ?Config = null;
var state: ?State = null;

/// Configure TVM before the first runtime operation.
///
/// Borrowed paths must remain alive until their corresponding library loads.
pub fn configure(next: Config) ConfigureError!void {
    if (state != null) return error.RuntimeAlreadyInitialized;
    config = next;
}

/// Load the configured TVM runtime surface.
pub fn ensure_loaded(requirement: Surface) Error!void {
    const configured = config orelse return error.TvmNotConfigured;
    if (!configured.surface.satisfies(requirement)) {
        log.err(
            "TVM {t} operation exceeds configured {t} surface",
            .{ requirement, configured.surface },
        );
        return error.TvmSurfaceUnavailable;
    }

    try ensure_ffi_loaded();
    if (requirement != .ffi) try ensure_surface_loaded();
}

/// Return TVM global-function names allocated by `allocator` for diagnostics.
pub fn list_global_names(allocator: std.mem.Allocator, requirement: Surface) Error![][]const u8 {
    try ensure_loaded(requirement);
    return try api.list_global_names(allocator);
}

fn ensure_ffi_loaded() Error!void {
    if (state != null) return;
    const configured = config orelse return error.TvmNotConfigured;

    var library = dylib.Library.open(std.heap.smp_allocator, configured.ffi.path, .{
        .visibility = .global,
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.OpenFailed => {
            log.err(
                "failed to open TVM FFI library '{s}': {s}",
                .{ configured.ffi.path, dylib.error_message() },
            );
            return error.TvmLoadFailed;
        },
    };
    errdefer library.close();

    raw.install_symbols(library) catch |err| {
        log.err("failed to resolve TVM FFI symbols: {s}", .{@errorName(err)});
        return error.TvmLoadFailed;
    };

    state = .{ .ffi = library };
    log.info("loaded TVM FFI library '{s}'", .{configured.ffi.path});
}

fn ensure_surface_loaded() Error!void {
    if (state.?.surface != null) return;
    const configured = config orelse return error.TvmNotConfigured;

    const selected = switch (configured.surface) {
        .ffi => unreachable,
        .runtime => configured.runtime,
        .compiler => configured.compiler,
    };

    const library = dylib.Library.open(std.heap.smp_allocator, selected.path, .{
        .visibility = .global,
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.OpenFailed => {
            log.err(
                "failed to open TVM {t} library '{s}': {s}",
                .{ configured.surface, selected.path, dylib.error_message() },
            );
            return error.TvmLoadFailed;
        },
    };

    state.?.surface = library;
    log.info("loaded TVM {t} library '{s}'", .{ configured.surface, selected.path });
}

test "Config uses stable TVM sonames" {
    try std.testing.expectEqualStrings(
        @import("config.zig").default_ffi_path,
        (Config{ .surface = .ffi }).ffi.path,
    );
    try std.testing.expectEqualStrings(
        @import("config.zig").default_runtime_path,
        (Config{ .surface = .runtime }).runtime.path,
    );
    try std.testing.expectEqualStrings(
        @import("config.zig").default_compiler_path,
        (Config{ .surface = .compiler }).compiler.path,
    );
}
