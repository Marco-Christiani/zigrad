const std = @import("std");
const kernel = @import("../kernel.zig");
const pr = @import("../pr/pr.zig");
const dispatch_mod = @import("dispatch.zig");
const artifact_mod = @import("artifact.zig");
const mirage_api = @import("../c/mirage/api.zig");
const mirage_c = @import("../c/mirage/c.zig");

const log = std.log.scoped(.@"zg/mirage_provider");
const superopt_max_candidates: u32 = 1024;
const max_region_eqns: usize = 5;

pub const MirageProvider = struct {
    allocator: std.mem.Allocator,
    dispatch_state: *dispatch_mod.MirageDispatchState,

    pub fn kernel_provider(self: *MirageProvider) kernel.KernelProvider {
        return .{
            .name = "mirage",
            .ptr = @ptrCast(self),
            .compile_fn = compile_impl,
            .compile_mlir_fn = compile_mlir_impl,
            .finalize_fn = finalize_impl,
            .device_memory_info_fn = device_memory_info_impl,
        };
    }

    fn finalize_impl(_: *anyopaque) void {}

    fn device_memory_info_impl(_: *anyopaque) ?kernel.DeviceMemoryInfo {
        const info = mirage_c.mirage_device_mem_info() orelse return null;
        return .{ .free_bytes = info.free, .total_bytes = info.total };
    }

    fn compile_impl(ptr: *anyopaque, desc: kernel.RegionDescriptor, allocator: std.mem.Allocator) kernel.CompileError!kernel.KernelArtifact {
        const self: *MirageProvider = @ptrCast(@alignCast(ptr));
        return self.compile(desc, allocator);
    }

    fn compile_mlir_impl(
        ptr: *anyopaque,
        desc: kernel.MlirKernelDescriptor,
        allocator: std.mem.Allocator,
    ) kernel.CompileError!kernel.KernelArtifact {
        const self: *MirageProvider = @ptrCast(@alignCast(ptr));
        return self.compile_mlir(desc, allocator);
    }

    fn compile(self: *MirageProvider, desc: kernel.RegionDescriptor, allocator: std.mem.Allocator) kernel.CompileError!kernel.KernelArtifact {
        if (desc.eqns.len > max_region_eqns) {
            log.debug(
                "region '{s}' has {d} eqns (> {d}); skipping mirage compile",
                .{ desc.name, desc.eqns.len, max_region_eqns },
            );
            return error.Unsupported;
        }

        var graph = mirage_api.Graph.init() catch |err| return map_mirage_api_error(err);
        defer graph.deinit();

        var tensor_map = std.AutoHashMap(pr.VarId, mirage_c.MirageTensor).init(allocator);
        defer tensor_map.deinit();

        try lower_region_graph(desc, &graph, &tensor_map, allocator);

        return self.search_and_transpile(desc.name, allocator, &graph);
    }

    fn compile_mlir(
        self: *MirageProvider,
        desc: kernel.MlirKernelDescriptor,
        allocator: std.mem.Allocator,
    ) kernel.CompileError!kernel.KernelArtifact {
        if (desc.outputs.len != 1) return error.Unsupported;

        const required_inputs: usize = switch (desc.pattern) {
            .dot, .dot_general, .dot_log, .dot_exp => 2,
            .dot_add, .dot_add_mul => 3,
            .rms_norm => 1,
            .softmax_matmul => 2,
            .attention => 3,
        };
        if (desc.inputs.len != required_inputs) return error.Unsupported;

        var graph = mirage_api.Graph.init() catch |err| return map_mirage_api_error(err);
        defer graph.deinit();

        var handles = try std.ArrayList(mirage_c.MirageTensor).initCapacity(allocator, desc.inputs.len);
        defer handles.deinit(allocator);

        for (desc.inputs) |input_desc| {
            const handle = try emit_graph_input(&graph, input_desc);
            try handles.append(allocator, handle);
        }

        const out_tensor = build_mirage_graph(&graph, desc.pattern, handles.items, desc) catch |err| switch (err) {
            // Shape rejections from Mirage (e.g. non-canonical batched matmul
            // layout, rank mismatch) are not fatal — fall back to the expand
            // pass which reconstructs the original StableHLO for XLA.
            error.MirageInvalidArgument => {
                log.warn(
                    "mirage rejected shapes for '{s}' (pattern={s}); falling back",
                    .{ desc.name, @tagName(desc.pattern) },
                );
                return error.Unsupported;
            },
            inline else => return err,
        };

        graph.markOutput(out_tensor) catch |err| return map_mirage_api_error(err);

        return self.search_and_transpile(desc.name, allocator, &graph);
    }

    /// Build the Mirage graph for a given kernel pattern.
    ///
    /// Translates the pattern-specific op sequence into Mirage graph ops.
    /// Returns `MirageInvalidArgument` if Mirage rejects the tensor shapes
    /// (e.g. non-canonical matmul layout); the caller maps this to `Unsupported`.
    fn build_mirage_graph(
        graph: *mirage_api.Graph,
        pattern: kernel.MlirKernelPattern,
        input_handles: []const mirage_c.MirageTensor,
        desc: kernel.MlirKernelDescriptor,
    ) kernel.CompileError!mirage_c.MirageTensor {
        switch (pattern) {
            .dot, .dot_general => {
                return try emit_matmul(graph, input_handles[0], input_handles[1]);
            },
            .dot_add => {
                const dot = try emit_matmul(graph, input_handles[0], input_handles[1]);
                return try emit_binary(graph, mirage_c.binary_add, dot, input_handles[2]);
            },
            .dot_add_mul => {
                const dot = try emit_matmul(graph, input_handles[0], input_handles[1]);
                const sum = try emit_binary(graph, mirage_c.binary_add, dot, input_handles[2]);
                return try emit_binary(graph, mirage_c.binary_mul, sum, input_handles[2]);
            },
            .dot_log => {
                const dot = try emit_matmul(graph, input_handles[0], input_handles[1]);
                return try emit_unary(graph, mirage_c.unary_log, dot);
            },
            .dot_exp => {
                const dot = try emit_matmul(graph, input_handles[0], input_handles[1]);
                return try emit_unary(graph, mirage_c.unary_exp, dot);
            },
            .rms_norm => {
                return graph.rmsNorm(input_handles[0], desc.normalized_size) catch |err|
                    return map_mirage_api_error(err);
            },
            .softmax_matmul => {
                const exp_result = try emit_unary(graph, mirage_c.unary_exp, input_handles[0]);
                const sum_result = graph.reduction(exp_result, desc.reduction_dim, desc.reduction_factor) catch |err|
                    return map_mirage_api_error(err);
                const attn_probs = try emit_binary(graph, mirage_c.binary_div, exp_result, sum_result);
                return try emit_matmul(graph, attn_probs, input_handles[1]);
            },
            .attention => {
                // Mirage graph omits scaling — the superoptimizer works on the
                // structural pattern. If Mirage returns Unsupported, the expand
                // pass reconstructs the full chain WITH scale correctly.
                const scores = try emit_matmul(graph, input_handles[0], input_handles[1]);
                const exp_result = try emit_unary(graph, mirage_c.unary_exp, scores);
                const sum_result = graph.reduction(exp_result, desc.reduction_dim, desc.reduction_factor) catch |err|
                    return map_mirage_api_error(err);
                const attn_probs = try emit_binary(graph, mirage_c.binary_div, exp_result, sum_result);
                return try emit_matmul(graph, attn_probs, input_handles[2]);
            },
        }
    }

    /// Run search on the graph and transpile the best candidate to CUDA source.
    fn search_and_transpile(
        self: *MirageProvider,
        target_name: []const u8,
        allocator: std.mem.Allocator,
        graph: *mirage_api.Graph,
    ) kernel.CompileError!kernel.KernelArtifact {
        // Search for optimized candidates.
        var device = mirage_api.Device.init(0) catch |err| return map_mirage_api_error(err);
        defer device.deinit();

        const search_opts = mirage_c.SearchOptions{
            .max_candidates = superopt_max_candidates,
        };

        var result = mirage_api.search(&device, graph, &search_opts) catch |err| {
            if (err == error.MirageApiUnsupported) {
                log.debug("mirage search unsupported for region '{s}'", .{target_name});
                return error.Unsupported;
            }
            return map_mirage_api_error(err);
        };
        defer result.deinit();

        // Use the best candidate if search found improvements, otherwise the original graph.
        const transpile_graph: ?*const mirage_c.MirageGraph = if (result.count() > 0)
            result.get(0)
        else
            graph.raw;

        // Transpile to CUDA source.
        var source = mirage_api.transpile(transpile_graph, null) catch |err| {
            if (err == error.MirageApiUnsupported) {
                log.debug("mirage transpile unsupported for region '{s}'", .{target_name});
                return error.Unsupported;
            }
            return map_mirage_api_error(err);
        };
        defer source.deinit();

        const cuda_code = source.code();
        if (cuda_code.len == 0) {
            log.err("mirage transpile returned empty source for '{s}'", .{target_name});
            return error.MirageContractError;
        }

        const buf_size = source.bufSize();
        if (buf_size != 0) {
            log.debug("mirage workspace for '{s}': {d} bytes", .{ target_name, buf_size });
        }

        // If the transpiled source has no custom kernels (only library ops
        // like standalone matmul → cuBLAS), we can't launch via NVRTC.
        // Return Unsupported so the backend handles this natively.
        const num_kernels = source.numKernels();
        if (num_kernels == 0) {
            log.debug("mirage transpile produced 0 custom kernels for '{s}'; falling back to backend", .{target_name});
            return error.Unsupported;
        }

        // Filter source for NVRTC (strip host code, replace runtime.h).
        const filtered = dispatch_mod.filter_source_for_nvrtc(allocator, cuda_code) catch {
            log.err("failed to filter source for '{s}'", .{target_name});
            return error.MirageInternalError;
        };
        defer allocator.free(filtered);

        // Build kernel descriptors from Layer 2 metadata.
        var kernel_descs = try std.ArrayList(artifact_mod.KernelDesc).initCapacity(allocator, num_kernels);
        defer {
            for (kernel_descs.items) |k| {
                allocator.free(k.args);
                allocator.free(k.func_name);
            }
            kernel_descs.deinit(allocator);
        }

        for (0..num_kernels) |ki| {
            const meta = source.kernelMeta(ki) catch |err| return map_mirage_api_error(err);
            const num_args = source.kernelNumArgs(ki);

            var args = try allocator.alloc(artifact_mod.KernelArg, num_args);
            for (0..num_args) |ai| {
                const arg = source.kernelArg(ki, ai) catch |err| {
                    allocator.free(args);
                    return map_mirage_api_error(err);
                };
                args[ai] = .{
                    .source = @enumFromInt(arg.source),
                    .index_or_offset = arg.index_or_offset,
                };
            }

            const func_name = if (meta.func_name) |ptr|
                @as([*]const u8, @ptrCast(ptr))[0..meta.func_name_len]
            else
                "";

            try kernel_descs.append(allocator, .{
                .func_name = try allocator.dupe(u8, func_name),
                .smem_bytes = @intCast(meta.smem_bytes),
                .grid_dim = meta.grid_dim,
                .block_dim = meta.block_dim,
                .args = args,
            });
        }

        // Serialize the artifact.
        const artifact_data = artifact_mod.encode(allocator, .{
            .source = filtered,
            .buf_size = @intCast(buf_size),
            .kernels = kernel_descs.items,
        }) catch {
            log.err("failed to encode artifact for '{s}'", .{target_name});
            return error.MirageInternalError;
        };

        return .{
            .provider_name = "mirage",
            .data = artifact_data,
            .target_name = try allocator.dupe(u8, target_name),
            .workspace_bytes = buf_size,
            .dispatch_fn = &dispatch_mod.MirageDispatchState.dispatch,
            .dispatch_ctx = @ptrCast(self.dispatch_state),
        };
    }
};

/// Release Mirage device memory. With the new API this destroys the
/// device handle; currently a no-op since device handles are scoped to
/// search_and_transpile calls.
pub fn release_device_memory() void {}

fn emit_graph_input(
    graph: *mirage_api.Graph,
    input_desc: kernel.MlirTensorDesc,
) kernel.CompileError!mirage_c.MirageTensor {
    const dtype = dtype_to_mirage(input_desc.dtype) orelse return error.Unsupported;

    var dims: [mirage_c.max_rank]i64 = .{ 0, 0, 0, 0 };
    for (input_desc.dims, 0..) |dim, idx| {
        if (dim == 0 or dim > std.math.maxInt(i64)) return error.Unsupported;
        dims[idx] = @intCast(dim);
    }

    const spec = mirage_c.TensorSpec{
        .dtype = dtype,
        .rank = @intCast(input_desc.dims.len),
        .dims = dims,
        .strides = .{ 0, 0, 0, 0 },
    };

    return graph.newInput(&spec) catch |err| return map_mirage_api_error(err);
}

fn lower_region_graph(
    desc: kernel.RegionDescriptor,
    graph: *mirage_api.Graph,
    tensor_map: *std.AutoHashMap(pr.VarId, mirage_c.MirageTensor),
    allocator: std.mem.Allocator,
) kernel.CompileError!void {
    for (desc.inputs) |in_id| {
        const aval = desc.aval_of(in_id) orelse return error.Unsupported;
        const tensor = aval.as_tensor() orelse return error.Unsupported;

        const dtype = dtype_to_mirage(tensor.dtype) orelse return error.Unsupported;

        var dims: [mirage_c.max_rank]i64 = .{ 0, 0, 0, 0 };
        for (tensor.shape.dims, 0..) |dim, idx| {
            if (dim == 0 or dim > std.math.maxInt(i64)) return error.Unsupported;
            dims[idx] = @intCast(dim);
        }

        const spec = mirage_c.TensorSpec{
            .dtype = dtype,
            .rank = @intCast(tensor.shape.dims.len),
            .dims = dims,
            .strides = .{ 0, 0, 0, 0 },
        };

        const handle = graph.newInput(&spec) catch |err| return map_mirage_api_error(err);
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
        graph.markOutput(out_tensor) catch |err| return map_mirage_api_error(err);
    }

    _ = allocator;
}

fn lower_eqn(
    graph: *mirage_api.Graph,
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

            const lhs_aval = desc.aval_of(inputs[0]) orelse return error.Unsupported;
            const rhs_aval = desc.aval_of(inputs[1]) orelse return error.Unsupported;
            const lhs_tensor = lhs_aval.as_tensor() orelse return error.Unsupported;
            const rhs_tensor = rhs_aval.as_tensor() orelse return error.Unsupported;

            if (!kernel.dot_general_is_canonical_batched_matmul(
                eqn.params.slice(pr.Param, desc.params_store),
                lhs_tensor.shape.rank(),
                rhs_tensor.shape.rank(),
            )) return error.Unsupported;

            const lhs = tensor_map.get(inputs[0]) orelse return error.Unsupported;
            const rhs = tensor_map.get(inputs[1]) orelse return error.Unsupported;
            return try emit_matmul(graph, lhs, rhs);
        },
        .exp => {
            if (inputs.len != 1) return error.Unsupported;
            const input = tensor_map.get(inputs[0]) orelse return error.Unsupported;
            return try emit_unary(graph, mirage_c.unary_exp, input);
        },
        .log => {
            if (inputs.len != 1) return error.Unsupported;
            const input = tensor_map.get(inputs[0]) orelse return error.Unsupported;
            return try emit_unary(graph, mirage_c.unary_log, input);
        },
        .add => {
            if (inputs.len != 2) return error.Unsupported;
            const lhs = tensor_map.get(inputs[0]) orelse return error.Unsupported;
            const rhs = tensor_map.get(inputs[1]) orelse return error.Unsupported;
            return try emit_binary(graph, mirage_c.binary_add, lhs, rhs);
        },
        .multiply => {
            if (inputs.len != 2) return error.Unsupported;
            const lhs = tensor_map.get(inputs[0]) orelse return error.Unsupported;
            const rhs = tensor_map.get(inputs[1]) orelse return error.Unsupported;
            return try emit_binary(graph, mirage_c.binary_mul, lhs, rhs);
        },
        .divide => {
            if (inputs.len != 2) return error.Unsupported;
            const lhs = tensor_map.get(inputs[0]) orelse return error.Unsupported;
            const rhs = tensor_map.get(inputs[1]) orelse return error.Unsupported;
            return try emit_binary(graph, mirage_c.binary_div, lhs, rhs);
        },
        else => return error.Unsupported,
    }
}

fn emit_matmul(graph: *mirage_api.Graph, lhs: mirage_c.MirageTensor, rhs: mirage_c.MirageTensor) kernel.CompileError!mirage_c.MirageTensor {
    return graph.matmul(lhs, rhs) catch |err| return map_mirage_api_error(err);
}

fn emit_unary(
    graph: *mirage_api.Graph,
    op: mirage_c.MirageUnaryOp,
    input: mirage_c.MirageTensor,
) kernel.CompileError!mirage_c.MirageTensor {
    return graph.unary(op, input) catch |err| return map_mirage_api_error(err);
}

fn emit_binary(
    graph: *mirage_api.Graph,
    op: mirage_c.MirageBinaryOp,
    lhs: mirage_c.MirageTensor,
    rhs: mirage_c.MirageTensor,
) kernel.CompileError!mirage_c.MirageTensor {
    return graph.binary(op, lhs, rhs) catch |err| return map_mirage_api_error(err);
}

fn dtype_to_mirage(dtype: pr.DType) ?mirage_c.MirageDType {
    return switch (dtype) {
        .bf16 => mirage_c.dtype_bf16,
        .f32 => mirage_c.dtype_f32,
        .f64 => mirage_c.dtype_f64,
        else => null,
    };
}

fn map_mirage_api_error(err: mirage_api.MirageError) kernel.CompileError {
    return switch (err) {
        error.MirageUnavailable => error.MirageLoadFailed,
        error.MirageInvalidArgument => error.MirageInvalidArgument,
        error.MirageInternalError => error.MirageInternalError,
        error.MirageApiUnsupported => error.MirageApiUnsupported,
        error.MirageNotFound => error.Unsupported,
        error.OutOfMemory => error.OutOfMemory,
    };
}

const StatusContext = enum {
    region,
    runtime,
};

fn map_mirage_status(status: mirage_c.MirageStatus, ctx: StatusContext) kernel.CompileError {
    if (status == mirage_c.status_invalid_argument) return error.MirageInvalidArgument;
    if (status == mirage_c.status_internal_error) return error.MirageInternalError;
    if (status == mirage_c.status_unsupported) return switch (ctx) {
        .region => error.Unsupported,
        .runtime => error.MirageApiUnsupported,
    };
    if (status == mirage_c.status_not_found) return error.Unsupported;
    return error.MirageInternalError;
}

test map_mirage_status {
    try std.testing.expectEqual(error.MirageInvalidArgument, map_mirage_status(mirage_c.status_invalid_argument, .region));
    try std.testing.expectEqual(error.MirageInternalError, map_mirage_status(mirage_c.status_internal_error, .region));
    try std.testing.expectEqual(error.Unsupported, map_mirage_status(mirage_c.status_unsupported, .region));

    try std.testing.expectEqual(error.MirageInvalidArgument, map_mirage_status(mirage_c.status_invalid_argument, .runtime));
    try std.testing.expectEqual(error.MirageInternalError, map_mirage_status(mirage_c.status_internal_error, .runtime));
    try std.testing.expectEqual(error.MirageApiUnsupported, map_mirage_status(mirage_c.status_unsupported, .runtime));
}
