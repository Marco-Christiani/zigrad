const std = @import("std");
const kernel = @import("../kernel.zig");
const mirage_api = @import("../c/mirage/api.zig");
const mirage_c = @import("../c/mirage/c.zig");

const log = std.log.scoped(.@"zg/mirage_dispatch");

pub const MirageDispatchState = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) MirageDispatchState {
        return .{ .allocator = allocator };
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
        _ = kernel_key;
        _ = self;

        var mirage_ctx = mirage_api.Context.init() catch |err| return map_mirage_api_error(err);
        defer mirage_ctx.deinit();

        try require_workspace(ctx);

        var input_descs = try ctx.allocator.alloc(mirage_c.BufferDesc, ctx.inputs.len);
        defer ctx.allocator.free(input_descs);
        for (ctx.inputs, 0..) |buf, i| {
            input_descs[i] = .{
                .data = buf.data,
                .dtype = dtype_to_mirage(buf.dtype),
                .dims = buf.dims.ptr,
                .rank = buf.rank,
            };
        }

        var output_descs = try ctx.allocator.alloc(mirage_c.BufferDesc, ctx.outputs.len);
        defer ctx.allocator.free(output_descs);
        for (ctx.outputs, 0..) |buf, i| {
            output_descs[i] = .{
                .data = buf.data,
                .dtype = dtype_to_mirage(buf.dtype),
                .dims = buf.dims.ptr,
                .rank = buf.rank,
            };
        }

        const params: mirage_c.DispatchParams = .{
            .inputs = if (input_descs.len == 0) null else input_descs.ptr,
            .num_inputs = input_descs.len,
            .outputs = if (output_descs.len == 0) null else output_descs.ptr,
            .num_outputs = output_descs.len,
            .device_ordinal = ctx.device_ordinal,
            .stream = ctx.stream,
        };

        const valid_status = mirage_c.mirage_validate_artifact(
            mirage_ctx.raw,
            artifact_data.ptr,
            artifact_data.len,
        );
        if (valid_status != .ok) {
            log.err("mirage_validate_artifact returned {s}", .{mirage_api.status_name(valid_status)});
            return map_mirage_status(valid_status);
        }

        const status = mirage_c.mirage_execute_kernel(
            mirage_ctx.raw,
            artifact_data.ptr,
            artifact_data.len,
            &params,
            ctx.workspace,
        );
        if (status != .ok) {
            log.err("mirage_execute_kernel returned {s}", .{mirage_api.status_name(status)});
            return map_mirage_status(status);
        }
    }
};

fn map_mirage_api_error(err: mirage_api.MirageError) kernel.DispatchError {
    return switch (err) {
        error.MirageUnavailable => error.MirageLoadFailed,
        error.MirageInvalidArgument => error.MirageInvalidArgument,
        error.MirageInternalError => error.MirageInternalError,
        error.MirageApiUnsupported => error.MirageApiUnsupported,
        error.OutOfMemory => error.OutOfMemory,
    };
}

fn map_mirage_status(status: mirage_c.MirageStatus) kernel.DispatchError {
    return switch (status) {
        .ok => unreachable,
        .invalid_argument => error.MirageInvalidArgument,
        .internal_error => error.MirageInternalError,
        .unsupported => error.MirageApiUnsupported,
    };
}

fn require_workspace(ctx: kernel.DispatchContext) kernel.DispatchError!void {
    if (ctx.workspace_bytes_required != 0 and ctx.workspace == null) {
        return error.WorkspaceUnavailable;
    }
}

fn dtype_to_mirage(dtype: kernel.DType) mirage_c.MirageDType {
    return switch (dtype) {
        .f16 => .f16,
        .bf16 => .bf16,
        .f32 => .f32,
        .f64 => .f64,
        .i8 => .i8,
        .i32 => .i32,
        .i64 => .i64,
        .u32 => .u32,
        .u64 => .u64,
    };
}

test "map_mirage_status preserves runtime detail" {
    try std.testing.expectEqual(error.MirageInvalidArgument, map_mirage_status(.invalid_argument));
    try std.testing.expectEqual(error.MirageInternalError, map_mirage_status(.internal_error));
    try std.testing.expectEqual(error.MirageApiUnsupported, map_mirage_status(.unsupported));
}

test "dispatch requires workspace when artifact needs it" {
    const ctx: kernel.DispatchContext = .{
        .inputs = &.{},
        .outputs = &.{},
        .device_ordinal = 0,
        .platform = .cuda,
        .stream = null,
        .workspace = null,
        .workspace_bytes_required = 16,
        .allocator = std.testing.allocator,
    };
    try std.testing.expectError(error.WorkspaceUnavailable, require_workspace(ctx));
}
