const std = @import("std");
const kernel = @import("../kernel.zig");
const mirage_api = @import("../c/mirage/api.zig");
const mirage_c = @import("../c/mirage/c.zig");

const log = std.log.scoped(.@"zg/mirage_dispatch");

/// Dispatch state for Mirage kernels.
///
/// With the layered C API, Mirage produces CUDA source code (artifact data)
/// rather than compiled .so artifacts. The dispatch layer is responsible for
/// compiling the source to a loadable module and launching the kernels.
///
/// Current implementation: delegates to system nvcc for compilation and
/// dlopen for execution, matching the compilation model that was previously
/// internal to mirage_graph_compile.
pub const MirageDispatchState = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) mirage_api.MirageError!MirageDispatchState {
        return .{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MirageDispatchState) void {
        _ = self;
    }

    pub fn dispatch(
        provider_ctx: *anyopaque,
        artifact_data: []const u8,
        kernel_key: []const u8,
        ctx: kernel.DispatchContext,
    ) kernel.DispatchError!void {
        const self: *MirageDispatchState = @ptrCast(@alignCast(provider_ctx));
        self.dispatch_impl(artifact_data, kernel_key, ctx) catch |err| {
            log.err("mirage dispatch failed for '{s}': {s}", .{ kernel_key, @errorName(err) });
            return err;
        };
    }

    fn dispatch_impl(
        self: *MirageDispatchState,
        artifact_data: []const u8,
        kernel_key: []const u8,
        ctx: kernel.DispatchContext,
    ) kernel.DispatchError!void {
        _ = self;
        _ = artifact_data;
        _ = ctx;
        // The artifact data is now CUDA source code from mirage_transpile().
        // Compilation and execution of the source is the caller's responsibility.
        // This dispatch path will be re-implemented when the NVRTC/cuModule
        // compilation pipeline is integrated.
        log.err("mirage dispatch not yet implemented for source-code artifacts (kernel '{s}')", .{kernel_key});
        return error.MirageInternalError;
    }
};

fn map_mirage_api_error(err: mirage_api.MirageError) kernel.DispatchError {
    return switch (err) {
        error.MirageUnavailable => error.MirageLoadFailed,
        error.MirageInvalidArgument => error.MirageInvalidArgument,
        error.MirageInternalError => error.MirageInternalError,
        error.MirageApiUnsupported => error.MirageApiUnsupported,
        error.MirageNotFound => error.MirageInternalError,
        error.OutOfMemory => error.OutOfMemory,
    };
}

fn map_mirage_status(status: mirage_c.MirageStatus) kernel.DispatchError {
    return switch (status) {
        .ok => unreachable,
        .invalid_argument => error.MirageInvalidArgument,
        .internal_error => error.MirageInternalError,
        .unsupported => error.MirageApiUnsupported,
        .not_found => error.MirageInternalError,
        _ => error.MirageInternalError,
    };
}

test "map_mirage_status preserves runtime detail" {
    try std.testing.expectEqual(error.MirageInvalidArgument, map_mirage_status(.invalid_argument));
    try std.testing.expectEqual(error.MirageInternalError, map_mirage_status(.internal_error));
    try std.testing.expectEqual(error.MirageApiUnsupported, map_mirage_status(.unsupported));
}
