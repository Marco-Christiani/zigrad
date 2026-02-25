const std = @import("std");
const kernel = @import("../kernel.zig");
const dispatch_mod = @import("dispatch.zig");
const mirage_api = @import("../c/mirage/api.zig");
const mirage_c = @import("../c/mirage/c.zig");

const log = std.log.scoped(.@"zg/mirage_provider");

pub const MirageProvider = struct {
    allocator: std.mem.Allocator,
    dispatch_state: *dispatch_mod.MirageDispatchState,
    launcher_so_path: ?[]const u8 = null,

    pub fn kernel_provider(self: *MirageProvider) kernel.KernelProvider {
        return .{
            .name = "mirage",
            .ptr = @ptrCast(self),
            .compile_fn = compile_impl,
        };
    }

    fn compile_impl(ptr: *anyopaque, desc: kernel.RegionDescriptor, allocator: std.mem.Allocator) kernel.CompileError!kernel.KernelArtifact {
        const self: *MirageProvider = @ptrCast(@alignCast(ptr));
        return self.compile(desc, allocator);
    }

    fn compile(self: *MirageProvider, desc: kernel.RegionDescriptor, allocator: std.mem.Allocator) kernel.CompileError!kernel.KernelArtifact {
        const launcher_path = self.launcher_so_path orelse blk: {
            const launcher_path_c = std.posix.getenv("MIRAGE_EXECUTE_MUGRAPH_SO");
            break :blk if (launcher_path_c) |v| std.mem.sliceTo(v, 0) else "";
        };

        const payload: mirage_c.PayloadV1 = .{
            .eqn_count = @intCast(desc.eqns.len),
            .num_inputs = @intCast(desc.inputs.len),
            .num_outputs = @intCast(desc.outputs.len),
            .launcher_so_path_len = @intCast(launcher_path.len),
        };
        const payload_header = std.mem.asBytes(&payload);
        var payload_bytes = try allocator.alloc(u8, payload_header.len + launcher_path.len);
        defer allocator.free(payload_bytes);
        @memcpy(payload_bytes[0..payload_header.len], payload_header);
        @memcpy(payload_bytes[payload_header.len..], launcher_path);

        var ctx = mirage_api.Context.init() catch |err| switch (err) {
            error.MirageUnavailable => return error.MirageLoadFailed,
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.MirageCompileFailed,
        };
        defer ctx.deinit();

        var artifact_ptr: [*]const u8 = undefined;
        var artifact_len: usize = 0;
        var launch_info: mirage_c.LaunchInfo = .{ .workspace_bytes = 0 };
        const status = mirage_c.mirage_compile_kernel(
            ctx.raw,
            payload_bytes.ptr,
            payload_bytes.len,
            &artifact_ptr,
            &artifact_len,
            &launch_info,
        );

        switch (status) {
            .ok => {},
            .unsupported => {
                log.debug("mirage compile unsupported for region '{s}'", .{desc.name});
                return error.Unsupported;
            },
            else => {
                log.err("mirage_compile_kernel failed for '{s}': {s}", .{ desc.name, mirage_api.status_name(status) });
                return error.MirageCompileFailed;
            },
        }

        if (launch_info.workspace_bytes != 0) {
            log.debug("mirage launch workspace for '{s}': {d} bytes", .{ desc.name, launch_info.workspace_bytes });
        }

        if (artifact_len == 0) return error.MirageContractError;
        defer mirage_c.mirage_release_buffer(ctx.raw, artifact_ptr, artifact_len);
        const artifact_bytes = try allocator.dupe(u8, artifact_ptr[0..artifact_len]);

        return .{
            .provider_name = "mirage",
            .data = artifact_bytes,
            .target_name = try allocator.dupe(u8, desc.name),
            .dispatch_fn = &dispatch_mod.MirageDispatchState.dispatch,
            .dispatch_ctx = @ptrCast(self.dispatch_state),
        };
    }
};
