const std = @import("std");
const kernel = @import("../kernel.zig");
const pr = @import("../pr/pr.zig");
const dispatch_mod = @import("dispatch.zig");
const mirage_api = @import("../c/mirage/api.zig");
const mirage_c = @import("../c/mirage/c.zig");

const log = std.log.scoped(.@"zg/mirage_provider");
const superopt_max_num_graphs: u32 = 64;

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
                return map_runtime_status(st);
            }
        }
        const graph_ptr = graph orelse {
            log.err("mirage_graph_create returned null graph for '{s}'", .{desc.name});
            return error.MirageContractError;
        };
        defer mirage_c.mirage_graph_destroy(graph_ptr);

        var tensor_map = std.AutoHashMap(pr.VarId, mirage_c.MirageTensor).init(allocator);
        defer tensor_map.deinit();

        try lower_region_graph(desc, graph_ptr, &tensor_map, allocator);

        {
            const superopt_opts = mirage_c.SuperoptOptions{
                .max_num_graphs = superopt_max_num_graphs,
                .imap_to_explore = null,
                .num_imaps = 0,
                .omap_to_explore = null,
                .num_omaps = 0,
                .grid_dim_to_explore = null,
                .num_grid_dims = 0,
                .block_dim_to_explore = null,
                .num_block_dims = 0,
                .fmap_to_explore = null,
                .num_fmaps = 0,
                .frange_to_explore = null,
                .num_franges = 0,
                .checkpoint_filename = null,
                .verbose = 0,
                .is_formal_verified = 0,
            };
            const st = mirage_c.mirage_graph_superoptimize_with_options(graph_ptr, null, &superopt_opts);
            if (st != .ok) {
                if (st == .unsupported) {
                    log.debug("mirage superopt unsupported for region '{s}'", .{desc.name});
                } else {
                    log.err("mirage_graph_superoptimize failed for '{s}': {s}", .{ desc.name, mirage_api.status_name(st) });
                }
                return map_region_status(st);
            }
        }

        var artifact_ptr: [*]const u8 = undefined;
        var artifact_len: usize = 0;
        var launch_info: mirage_c.LaunchInfo = .{ .workspace_bytes = 0 };
        const status = mirage_c.mirage_graph_compile(
            ctx.raw,
            graph_ptr,
            0,
            &artifact_ptr,
            &artifact_len,
            &launch_info,
        );

        if (status != .ok) {
            if (status == .unsupported) {
                log.debug("mirage compile unsupported for region '{s}'", .{desc.name});
            } else {
                log.err("mirage_graph_compile failed for '{s}': {s}", .{ desc.name, mirage_api.status_name(status) });
            }
            return map_region_status(status);
        }

        if (launch_info.workspace_bytes != 0) {
            log.debug("mirage launch workspace for '{s}': {d} bytes", .{ desc.name, launch_info.workspace_bytes });
        }

        if (artifact_len == 0) return error.MirageContractError;

        const validate_status = mirage_c.mirage_validate_artifact(ctx.raw, artifact_ptr, artifact_len);
        if (validate_status != .ok) {
            log.err("mirage_validate_artifact failed for '{s}': {s}", .{ desc.name, mirage_api.status_name(validate_status) });
            return error.MirageContractError;
        }

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

fn lower_region_graph(
    desc: kernel.RegionDescriptor,
    graph: *mirage_c.MirageGraph,
    tensor_map: *std.AutoHashMap(pr.VarId, mirage_c.MirageTensor),
    allocator: std.mem.Allocator,
) kernel.CompileError!void {
    for (desc.inputs) |in_id| {
        const aval = desc.aval_of(in_id) orelse return error.Unsupported;
        const tensor = aval.as_tensor() orelse return error.Unsupported;

        const dtype = dtype_to_mirage(tensor.dtype) orelse return error.Unsupported;
        const dims = try allocator.alloc(i64, tensor.shape.dims.len);
        defer allocator.free(dims);
        for (tensor.shape.dims, 0..) |dim, idx| {
            if (dim == 0 or dim > std.math.maxInt(i64)) return error.Unsupported;
            dims[idx] = @intCast(dim);
        }

        var handle: mirage_c.MirageTensor = 0;
        const st = mirage_c.mirage_graph_new_input(graph, dims.ptr, dims.len, dtype, &handle);
        if (st != .ok) return map_region_status(st);

        try tensor_map.put(in_id, handle);
    }

    for (desc.eqns) |eqn| {
        const inputs = eqn.inputs.slice(pr.VarId, desc.varids_store);
        const outputs = eqn.outputs.slice(pr.VarId, desc.varids_store);
        if (outputs.len != 1) return error.Unsupported;

        const out_tensor = try lower_eqn(graph, desc, eqn, inputs, tensor_map);
        try tensor_map.put(outputs[0], out_tensor);
    }

    for (desc.outputs) |out_id| {
        const out_tensor = tensor_map.get(out_id) orelse return error.Unsupported;
        const st = mirage_c.mirage_graph_mark_output(graph, out_tensor);
        if (st != .ok) return map_region_status(st);
    }
}

fn lower_eqn(
    graph: *mirage_c.MirageGraph,
    desc: kernel.RegionDescriptor,
    eqn: pr.Eqn,
    inputs: []const pr.VarId,
    tensor_map: *const std.AutoHashMap(pr.VarId, mirage_c.MirageTensor),
) kernel.CompileError!mirage_c.MirageTensor {
    switch (eqn.prim) {
        .dot => {
            if (inputs.len != 2) return error.Unsupported;
            const lhs = tensor_map.get(inputs[0]) orelse return error.Unsupported;
            const rhs = tensor_map.get(inputs[1]) orelse return error.Unsupported;
            return try emit_matmul(graph, lhs, rhs);
        },
        .dot_general => {
            if (inputs.len != 2) return error.Unsupported;
            if (!kernel.dot_general_is_matrix_matmul(eqn.params.slice(pr.Param, desc.params_store))) return error.Unsupported;
            const lhs = tensor_map.get(inputs[0]) orelse return error.Unsupported;
            const rhs = tensor_map.get(inputs[1]) orelse return error.Unsupported;
            return try emit_matmul(graph, lhs, rhs);
        },
        .exp => {
            if (inputs.len != 1) return error.Unsupported;
            const input = tensor_map.get(inputs[0]) orelse return error.Unsupported;
            return try emit_unary(graph, .exp, input);
        },
        .log => {
            if (inputs.len != 1) return error.Unsupported;
            const input = tensor_map.get(inputs[0]) orelse return error.Unsupported;
            return try emit_unary(graph, .log, input);
        },
        .add => {
            if (inputs.len != 2) return error.Unsupported;
            const lhs = tensor_map.get(inputs[0]) orelse return error.Unsupported;
            const rhs = tensor_map.get(inputs[1]) orelse return error.Unsupported;
            return try emit_binary(graph, .add, lhs, rhs);
        },
        .multiply => {
            if (inputs.len != 2) return error.Unsupported;
            const lhs = tensor_map.get(inputs[0]) orelse return error.Unsupported;
            const rhs = tensor_map.get(inputs[1]) orelse return error.Unsupported;
            return try emit_binary(graph, .mul, lhs, rhs);
        },
        .divide => {
            if (inputs.len != 2) return error.Unsupported;
            const lhs = tensor_map.get(inputs[0]) orelse return error.Unsupported;
            const rhs = tensor_map.get(inputs[1]) orelse return error.Unsupported;
            return try emit_binary(graph, .div, lhs, rhs);
        },
        else => return error.Unsupported,
    }
}

fn emit_matmul(graph: *mirage_c.MirageGraph, lhs: mirage_c.MirageTensor, rhs: mirage_c.MirageTensor) kernel.CompileError!mirage_c.MirageTensor {
    var out_tensor: mirage_c.MirageTensor = 0;
    const st = mirage_c.mirage_graph_matmul(graph, lhs, rhs, &out_tensor);
    if (st != .ok) return map_region_status(st);
    return out_tensor;
}

fn emit_unary(
    graph: *mirage_c.MirageGraph,
    op: mirage_c.MirageUnaryOp,
    input: mirage_c.MirageTensor,
) kernel.CompileError!mirage_c.MirageTensor {
    var out_tensor: mirage_c.MirageTensor = 0;
    const st = mirage_c.mirage_graph_unary(graph, op, input, &out_tensor);
    if (st != .ok) return map_region_status(st);
    return out_tensor;
}

fn emit_binary(
    graph: *mirage_c.MirageGraph,
    op: mirage_c.MirageBinaryOp,
    lhs: mirage_c.MirageTensor,
    rhs: mirage_c.MirageTensor,
) kernel.CompileError!mirage_c.MirageTensor {
    var out_tensor: mirage_c.MirageTensor = 0;
    const st = mirage_c.mirage_graph_binary(graph, op, lhs, rhs, &out_tensor);
    if (st != .ok) return map_region_status(st);
    return out_tensor;
}

fn dtype_to_mirage(dtype: pr.DType) ?mirage_c.MirageDType {
    return switch (dtype) {
        .bf16 => .bf16,
        .f32 => .f32,
        .f64 => .f64,
        .i32 => .i32,
        .i64 => .i64,
        .u32 => .u32,
        .u64 => .u64,
        else => null,
    };
}

fn map_region_status(status: mirage_c.MirageStatus) kernel.CompileError {
    return switch (status) {
        .ok => unreachable,
        .invalid_argument => error.MirageInvalidArgument,
        .internal_error => error.MirageInternalError,
        .unsupported => error.Unsupported,
    };
}

fn map_runtime_status(status: mirage_c.MirageStatus) kernel.CompileError {
    return switch (status) {
        .ok => unreachable,
        .invalid_argument => error.MirageInvalidArgument,
        .internal_error => error.MirageInternalError,
        .unsupported => error.MirageApiUnsupported,
    };
}
