const std = @import("std");
const kernel = @import("../kernel.zig");
const dispatch_mod = @import("dispatch.zig");
const mirage_api = @import("../c/mirage/api.zig");
const mirage_c = @import("../c/mirage/c.zig");

const log = std.log.scoped(.@"zg/mirage_provider");

pub const MirageProvider = struct {
    allocator: std.mem.Allocator,
    dispatch_state: *dispatch_mod.MirageDispatchState,

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
        const matmul = parse_matmul_region(desc) catch |err| switch (err) {
            error.Unsupported => return error.Unsupported,
        };

        var ctx = mirage_api.Context.init() catch |err| switch (err) {
            error.MirageUnavailable => return error.MirageLoadFailed,
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.MirageCompileFailed,
        };
        defer ctx.deinit();

        var graph: ?*mirage_c.MirageGraph = null;
        {
            const st = mirage_c.mirage_graph_create(ctx.raw, &graph);
            if (st != .ok) {
                log.err("mirage_graph_create failed for '{s}': {s}", .{ desc.name, mirage_api.status_name(st) });
                return error.MirageCompileFailed;
            }
        }
        defer mirage_c.mirage_graph_destroy(graph);

        var lhs_tensor: mirage_c.MirageTensor = 0;
        var rhs_tensor: mirage_c.MirageTensor = 0;
        var out_tensor: mirage_c.MirageTensor = 0;

        const lhs_dims = [_]i64{ @intCast(matmul.m), @intCast(matmul.k) };
        const rhs_dims = [_]i64{ @intCast(matmul.k), @intCast(matmul.n) };

        {
            const st = mirage_c.mirage_graph_new_input(graph, lhs_dims[0..].ptr, lhs_dims.len, .f32, &lhs_tensor);
            if (st != .ok) {
                log.err("mirage_graph_new_input(lhs) failed for '{s}': {s}", .{ desc.name, mirage_api.status_name(st) });
                return error.MirageCompileFailed;
            }
        }
        {
            const st = mirage_c.mirage_graph_new_input(graph, rhs_dims[0..].ptr, rhs_dims.len, .f32, &rhs_tensor);
            if (st != .ok) {
                log.err("mirage_graph_new_input(rhs) failed for '{s}': {s}", .{ desc.name, mirage_api.status_name(st) });
                return error.MirageCompileFailed;
            }
        }
        {
            const st = mirage_c.mirage_graph_matmul(graph, lhs_tensor, rhs_tensor, &out_tensor);
            if (st != .ok) {
                log.err("mirage_graph_matmul failed for '{s}': {s}", .{ desc.name, mirage_api.status_name(st) });
                return error.Unsupported;
            }
        }
        {
            const st = mirage_c.mirage_graph_mark_output(graph, out_tensor);
            if (st != .ok) {
                log.err("mirage_graph_mark_output failed for '{s}': {s}", .{ desc.name, mirage_api.status_name(st) });
                return error.MirageCompileFailed;
            }
        }
        {
            const st = mirage_c.mirage_graph_superoptimize(graph, null);
            if (st == .unsupported) {
                return error.Unsupported;
            }
            if (st != .ok) {
                log.err("mirage_graph_superoptimize failed for '{s}': {s}", .{ desc.name, mirage_api.status_name(st) });
                return error.MirageCompileFailed;
            }
        }

        var artifact_ptr: [*]const u8 = undefined;
        var artifact_len: usize = 0;
        var launch_info: mirage_c.LaunchInfo = .{ .workspace_bytes = 0 };
        const status = mirage_c.mirage_graph_compile(
            ctx.raw,
            graph,
            0,
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
                log.err("mirage_graph_compile failed for '{s}': {s}", .{ desc.name, mirage_api.status_name(status) });
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

const MatmulRegion = struct {
    m: usize,
    k: usize,
    n: usize,
};

fn parse_matmul_region(desc: kernel.RegionDescriptor) error{Unsupported}!MatmulRegion {
    if (desc.eqns.len != 1) return error.Unsupported;
    if (desc.inputs.len != 2 or desc.outputs.len != 1) return error.Unsupported;

    const eqn = desc.eqns[0];
    if (eqn.prim != .dot) return error.Unsupported;

    const lhs_id = desc.inputs[0];
    const rhs_id = desc.inputs[1];
    const out_id = desc.outputs[0];

    const lhs_aval = desc.aval_of(lhs_id) orelse return error.Unsupported;
    const rhs_aval = desc.aval_of(rhs_id) orelse return error.Unsupported;
    const out_aval = desc.aval_of(out_id) orelse return error.Unsupported;

    const lhs = lhs_aval.as_tensor() orelse return error.Unsupported;
    const rhs = rhs_aval.as_tensor() orelse return error.Unsupported;
    const out = out_aval.as_tensor() orelse return error.Unsupported;

    if (lhs.dtype != .f32 or rhs.dtype != .f32 or out.dtype != .f32) return error.Unsupported;
    if (lhs.shape.rank() != 2 or rhs.shape.rank() != 2 or out.shape.rank() != 2) return error.Unsupported;

    const m = lhs.shape.dims[0];
    const k = lhs.shape.dims[1];
    if (rhs.shape.dims[0] != k) return error.Unsupported;
    const n = rhs.shape.dims[1];
    if (out.shape.dims[0] != m or out.shape.dims[1] != n) return error.Unsupported;

    if (m > std.math.maxInt(u32) or k > std.math.maxInt(u32) or n > std.math.maxInt(u32)) return error.Unsupported;

    return .{ .m = m, .k = k, .n = n };
}
