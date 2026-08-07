const std = @import("std");
const device = @import("../device.zig");
const region_view = @import("../pr/region_view.zig");
const kernel = @import("../pr/kernel.zig");
const pr = @import("../pr/pr.zig");
const dispatch_mod = @import("dispatch.zig");
const artifact_mod = @import("artifact.zig");
const config = @import("config.zig");
const mirage = @import("../c/mirage/api.zig");
const mlir_types = @import("mlir.zig");
const TypedPtr = @import("../utils/rtti.zig").TypedPtr;

const log = std.log.scoped(.@"zg/mirage_provider");
const max_region_eqns: usize = 5;

// Alignment required by offsets in Mirage DTensor workspace plans.
//
// TODO(mirage): Read this value from transpile metadata when the C API exposes
//  it.
const workspace_alignment: usize = 128;

pub const MirageProvider = struct {
    dispatch_state: *dispatch_mod.MirageDispatchState,

    pub const InitOptions = struct {
        /// Mirage adapter loading policy.
        runtime: config.RuntimeConfig = .{},
    };

    /// Initialize the provider and load its adapter dependency.
    pub fn init(
        dispatch_state: *dispatch_mod.MirageDispatchState,
        options: InitOptions,
    ) mirage.MirageError!MirageProvider {
        try mirage.load_runtime(options.runtime.adapter.path);
        return .{
            .dispatch_state = dispatch_state,
        };
    }

    pub fn kernel_provider(self: *MirageProvider) kernel.KernelProvider {
        return .{
            .name = "mirage",
            .ptr = @ptrCast(self),
            .compile_fn = compile_impl,
            .dispatch_fn = &dispatch_mod.MirageDispatchState.dispatch,
            .dispatch_ctx = TypedPtr.init(self.dispatch_state),
        };
    }

    fn compile_impl(ptr: *anyopaque, desc: region_view.RegionView, selected_device: device.Device, allocator: std.mem.Allocator) kernel.CompileError!kernel.Artifact {
        _ = ptr;
        return try compile(desc, selected_device, allocator);
    }

    fn compile(
        desc: region_view.RegionView,
        selected_device: device.Device,
        allocator: std.mem.Allocator,
    ) kernel.CompileError!kernel.Artifact {
        if (!selected_device.platform.eql(.cuda)) return error.Unsupported;
        if (desc.ops.len > max_region_eqns) {
            log.debug(
                "region '{s}' has {d} ops (> {d}); skipping mirage compile",
                .{ desc.name, desc.ops.len, max_region_eqns },
            );
            return error.Unsupported;
        }

        const graph = mirage.Graph.init() catch |err| return map_mirage_api_error(err);
        defer graph.deinit();

        var tensor_map = std.AutoHashMap(*const pr.Var, mirage.Tensor).init(allocator);
        defer tensor_map.deinit();

        try lower_region_graph(desc, graph, &tensor_map);

        return try optimize_and_transpile(
            desc.name,
            selected_device,
            allocator,
            graph,
        );
    }

    /// Compile one MLIR pattern descriptor for the selected CUDA device.
    pub fn compile_mlir(
        desc: mlir_types.MlirKernelDescriptor,
        selected_device: device.Device,
        allocator: std.mem.Allocator,
    ) kernel.CompileError!kernel.Artifact {
        if (!selected_device.platform.eql(.cuda)) return error.Unsupported;
        if (desc.outputs.len != 1) return error.Unsupported;

        const required_inputs: usize = switch (desc.pattern) {
            .dot, .dot_general, .dot_log, .dot_exp => 2,
            .dot_add, .dot_add_mul => 3,
            .rms_norm => 1,
            .rms_norm_matmul => 2,
            .softmax_matmul => 2,
            .attention => 3,
        };
        if (desc.inputs.len != required_inputs) return error.Unsupported;

        log.debug("compile_mlir '{s}' pattern={s} inputs:", .{ desc.name, @tagName(desc.pattern) });
        for (desc.inputs, 0..) |input_desc, i| {
            log.debug("  in{d}: {s}{any}", .{ i, @tagName(input_desc.dtype), input_desc.dims });
        }

        const graph = mirage.Graph.init() catch |err| return map_mirage_api_error(err);
        defer graph.deinit();

        var handles = try std.ArrayList(mirage.Tensor).initCapacity(allocator, desc.inputs.len);
        defer handles.deinit(allocator);

        for (desc.inputs) |input_desc| {
            const handle = try emit_graph_input(graph, input_desc);
            try handles.append(allocator, handle);
        }

        const out_tensor = build_mirage_graph(graph, desc.pattern, handles.items, desc) catch |err| {
            log.err("graph construction failed for '{s}' (pattern={s}): {s}", .{
                desc.name, @tagName(desc.pattern), @errorName(err),
            });
            return err;
        };

        graph.mark_output(out_tensor) catch |err| return map_mirage_api_error(err);

        return try optimize_and_transpile(
            desc.name,
            selected_device,
            allocator,
            graph,
        );
    }

    /// Build the Mirage graph for a given kernel pattern.
    ///
    /// Translates the pattern-specific op sequence into Mirage graph ops.
    /// Returns `Unsupported` if Mirage rejects the tensor shapes (e.g.
    ///  non-canonical matmul layout).
    ///
    /// Callers can fall back to baseline lowering.
    fn build_mirage_graph(
        graph: *mirage.Graph,
        pattern: mlir_types.MlirKernelPattern,
        input_handles: []const mirage.Tensor,
        desc: mlir_types.MlirKernelDescriptor,
    ) kernel.CompileError!mirage.Tensor {
        switch (pattern) {
            .dot, .dot_general => {
                return try emit_matmul(graph, input_handles[0], input_handles[1]);
            },
            .dot_add => {
                const dot = try emit_matmul(graph, input_handles[0], input_handles[1]);
                return try emit_binary(graph, .add, dot, input_handles[2]);
            },
            .dot_add_mul => {
                const dot = try emit_matmul(graph, input_handles[0], input_handles[1]);
                const sum = try emit_binary(graph, .add, dot, input_handles[2]);
                return try emit_binary(graph, .mul, sum, input_handles[2]);
            },
            .dot_log => {
                const dot = try emit_matmul(graph, input_handles[0], input_handles[1]);
                return try emit_unary(graph, .log, dot);
            },
            .dot_exp => {
                const dot = try emit_matmul(graph, input_handles[0], input_handles[1]);
                return try emit_unary(graph, .exp, dot);
            },
            .rms_norm => {
                return graph.rms_norm(input_handles[0], desc.normalized_size) catch |err|
                    return map_mirage_api_error(err);
            },
            .rms_norm_matmul => {
                const rms_result = graph.rms_norm(input_handles[0], desc.normalized_size) catch |err|
                    return map_mirage_api_error(err);
                return try emit_matmul(graph, rms_result, input_handles[1]);
            },
            .softmax_matmul => {
                const exp_result = try emit_unary(graph, .exp, input_handles[0]);
                const sum_result = graph.reduction(exp_result, desc.reduction_dim, desc.reduction_factor) catch |err|
                    return map_mirage_api_error(err);
                const attn_probs = try emit_binary(graph, .div, exp_result, sum_result);
                return try emit_matmul(graph, attn_probs, input_handles[1]);
            },
            .attention => {
                // Mirage graph omits scaling because the superoptimizer works on
                //  the structural pattern.
                //
                // The expand pass reconstructs the scaled chain when Mirage
                //  returns Unsupported.
                const scores = try emit_matmul(graph, input_handles[0], input_handles[1]);
                const exp_result = try emit_unary(graph, .exp, scores);
                const sum_result = graph.reduction(exp_result, desc.reduction_dim, desc.reduction_factor) catch |err|
                    return map_mirage_api_error(err);
                const attn_probs = try emit_binary(graph, .div, exp_result, sum_result);
                return try emit_matmul(graph, attn_probs, input_handles[2]);
            },
        }
    }

    /// Optimize the graph and transpile the selected graph to CUDA source.
    fn optimize_and_transpile(
        target_name: []const u8,
        selected_device: device.Device,
        allocator: std.mem.Allocator,
        graph: *mirage.Graph,
    ) kernel.CompileError!kernel.Artifact {
        const mirage_device = mirage.Device.init(selected_device.ordinal) catch |err|
            return map_mirage_api_error(err);
        defer mirage_device.deinit();

        const optimize_opts: mirage.OptimizeOptions = .{};

        const optimized = mirage.optimize(mirage_device, graph, &optimize_opts) catch |err| {
            if (err == error.MirageApiUnsupported) {
                log.debug("mirage symbolic optimization is unsupported for region '{s}'", .{target_name});
                return error.Unsupported;
            }
            if (err == error.MirageNotFound) {
                log.debug("mirage found no optimized graph for region '{s}'", .{target_name});
                return error.Unsupported;
            }
            return map_mirage_api_error(err);
        };
        defer optimized.deinit();

        var source = mirage.transpile(allocator, optimized, null) catch |err| {
            if (err == error.MirageApiUnsupported) {
                log.debug("mirage transpile unsupported for region '{s}'", .{target_name});
                return error.Unsupported;
            }
            return map_mirage_api_error(err);
        };
        defer source.deinit();

        const cuda_code = source.code;
        if (cuda_code.len == 0) {
            log.err("mirage transpile returned empty source for '{s}'", .{target_name});
            return error.ProviderCallFailed;
        }

        const buf_size = source.workspace_size;
        if (buf_size != 0) {
            log.debug("mirage workspace for '{s}': {d} bytes", .{ target_name, buf_size });
        }

        // A source with no custom kernels cannot launch through NVRTC.
        //
        // This occurs for library operations such as a standalone cuBLAS matmul.
        //  Returning Unsupported leaves the operation with the backend.
        const num_kernels = source.kernels.len;
        if (num_kernels == 0) {
            log.debug("mirage transpile produced 0 custom kernels for '{s}'; falling back to backend", .{target_name});
            return error.Unsupported;
        }

        // Filter source for NVRTC (strip host code, replace runtime.h).
        const filtered = try dispatch_mod.filter_source_for_nvrtc(allocator, cuda_code);
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

        for (source.kernels) |meta| {
            var args = try allocator.alloc(artifact_mod.KernelArg, meta.args.len);
            errdefer allocator.free(args);
            for (meta.args, 0..) |arg, ai| {
                args[ai] = .{
                    .source = switch (arg.source) {
                        .input => .input,
                        .output => .output,
                        .workspace => .buf,
                    },
                    .index_or_offset = arg.index_or_offset,
                };
            }

            const func_name = try allocator.dupe(u8, meta.func_name);
            errdefer allocator.free(func_name);
            try kernel_descs.append(allocator, .{
                .func_name = func_name,
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
            return error.ProviderCallFailed;
        };

        return .{
            .data = artifact_data,
            .workspace_bytes = buf_size,
            .workspace_alignment = workspace_alignment,
        };
    }
};

fn emit_graph_input(
    graph: *mirage.Graph,
    input_desc: mlir_types.MlirTensorDesc,
) kernel.CompileError!mirage.Tensor {
    const dtype = dtype_to_mirage(input_desc.dtype) orelse {
        log.debug("unsupported dtype for mirage input: {s}", .{@tagName(input_desc.dtype)});
        return error.Unsupported;
    };
    if (input_desc.dims.len == 0 or input_desc.dims.len > mirage.max_rank)
        return error.Unsupported;

    var dims: [mirage.max_rank]i64 = .{ 0, 0, 0, 0 };
    for (input_desc.dims, 0..) |dim, idx| {
        if (dim == 0 or dim > std.math.maxInt(i64)) return error.Unsupported;
        dims[idx] = @intCast(dim);
    }

    const spec = mirage.TensorSpec{
        .dtype = dtype,
        .dims = dims[0..input_desc.dims.len],
    };

    return graph.new_input(&spec) catch |err| {
        log.debug("mirage graph.new_input rejected ({s} rank={d}): {s}", .{
            @tagName(dtype), input_desc.dims.len, @errorName(err),
        });
        return map_mirage_api_error(err);
    };
}

fn lower_region_graph(
    desc: region_view.RegionView,
    graph: *mirage.Graph,
    tensor_map: *std.AutoHashMap(*const pr.Var, mirage.Tensor),
) kernel.CompileError!void {
    for (desc.inputs) |in_var| {
        const tensor = in_var.as_tensor();

        const dtype = dtype_to_mirage(tensor.dtype) orelse return error.Unsupported;
        if (tensor.shape.dims.len == 0 or tensor.shape.dims.len > mirage.max_rank)
            return error.Unsupported;

        var dims: [mirage.max_rank]i64 = .{ 0, 0, 0, 0 };
        for (tensor.shape.dims, 0..) |dim, idx| {
            if (dim == 0 or dim > std.math.maxInt(i64)) return error.Unsupported;
            dims[idx] = @intCast(dim);
        }

        const spec = mirage.TensorSpec{
            .dtype = dtype,
            .dims = dims[0..tensor.shape.dims.len],
        };

        const handle = graph.new_input(&spec) catch |err| return map_mirage_api_error(err);
        try tensor_map.put(in_var, handle);
    }

    for (desc.ops) |op| {
        if (op.outputs.len != 1) return error.Unsupported;

        const out_tensor = try lower_op(graph, op, tensor_map);
        try tensor_map.put(op.outputs[0], out_tensor);
    }

    for (desc.outputs) |out_var| {
        const out_tensor = tensor_map.get(out_var) orelse return error.Unsupported;
        graph.mark_output(out_tensor) catch |err| return map_mirage_api_error(err);
    }
}

fn lower_op(
    graph: *mirage.Graph,
    op: *const pr.Op,
    tensor_map: *const std.AutoHashMap(*const pr.Var, mirage.Tensor),
) kernel.CompileError!mirage.Tensor {
    switch (op.params) {
        .dot => {
            if (op.inputs.len != 2) return error.Unsupported;
            const lhs = tensor_map.get(op.inputs[0].value) orelse return error.Unsupported;
            const rhs = tensor_map.get(op.inputs[1].value) orelse return error.Unsupported;
            return try emit_matmul(graph, lhs, rhs);
        },
        .dot_general => |dg| {
            if (op.inputs.len != 2) return error.Unsupported;

            const lhs_tensor = op.inputs[0].value.as_tensor();
            const rhs_tensor = op.inputs[1].value.as_tensor();

            if (!kernel.dot_general_is_canonical_batched_matmul(
                dg,
                lhs_tensor.shape.rank(),
                rhs_tensor.shape.rank(),
            )) return error.Unsupported;

            const lhs = tensor_map.get(op.inputs[0].value) orelse return error.Unsupported;
            const rhs = tensor_map.get(op.inputs[1].value) orelse return error.Unsupported;
            return try emit_matmul(graph, lhs, rhs);
        },
        .exp => {
            if (op.inputs.len != 1) return error.Unsupported;
            const input = tensor_map.get(op.inputs[0].value) orelse return error.Unsupported;
            return try emit_unary(graph, .exp, input);
        },
        .log => {
            if (op.inputs.len != 1) return error.Unsupported;
            const input = tensor_map.get(op.inputs[0].value) orelse return error.Unsupported;
            return try emit_unary(graph, .log, input);
        },
        .add => {
            if (op.inputs.len != 2) return error.Unsupported;
            const lhs = tensor_map.get(op.inputs[0].value) orelse return error.Unsupported;
            const rhs = tensor_map.get(op.inputs[1].value) orelse return error.Unsupported;
            return try emit_binary(graph, .add, lhs, rhs);
        },
        .multiply => {
            if (op.inputs.len != 2) return error.Unsupported;
            const lhs = tensor_map.get(op.inputs[0].value) orelse return error.Unsupported;
            const rhs = tensor_map.get(op.inputs[1].value) orelse return error.Unsupported;
            return try emit_binary(graph, .mul, lhs, rhs);
        },
        .divide => {
            if (op.inputs.len != 2) return error.Unsupported;
            const lhs = tensor_map.get(op.inputs[0].value) orelse return error.Unsupported;
            const rhs = tensor_map.get(op.inputs[1].value) orelse return error.Unsupported;
            return try emit_binary(graph, .div, lhs, rhs);
        },
        else => return error.Unsupported,
    }
}

fn emit_matmul(graph: *mirage.Graph, lhs: mirage.Tensor, rhs: mirage.Tensor) kernel.CompileError!mirage.Tensor {
    return graph.matmul(lhs, rhs) catch |err| {
        log.err("mirage graph.matmul rejected (lhs={d}, rhs={d}): {s}", .{ lhs.index, rhs.index, @errorName(err) });
        return map_mirage_api_error(err);
    };
}

fn emit_unary(
    graph: *mirage.Graph,
    op: mirage.UnaryOp,
    input: mirage.Tensor,
) kernel.CompileError!mirage.Tensor {
    return graph.unary(op, input) catch |err| {
        log.err("mirage graph.unary({s}) rejected (input={d}): {s}", .{ @tagName(op), input.index, @errorName(err) });
        return map_mirage_api_error(err);
    };
}

fn emit_binary(
    graph: *mirage.Graph,
    op: mirage.BinaryOp,
    lhs: mirage.Tensor,
    rhs: mirage.Tensor,
) kernel.CompileError!mirage.Tensor {
    return graph.binary(op, lhs, rhs) catch |err| {
        log.err("mirage graph.binary({s}) rejected (lhs={d}, rhs={d}): {s}", .{ @tagName(op), lhs.index, rhs.index, @errorName(err) });
        return map_mirage_api_error(err);
    };
}

fn dtype_to_mirage(dtype: pr.DType) ?mirage.DType {
    return switch (dtype) {
        .bf16 => .bf16,
        .f32 => .f32,
        .f64 => .f64,
        else => null,
    };
}

fn map_mirage_api_error(err: mirage.MirageError) kernel.CompileError {
    return switch (err) {
        error.MirageUnavailable => error.ProviderLoadFailed,
        error.MirageInvalidArgument => error.Unsupported,
        error.MirageInternalError => error.ProviderCallFailed,
        error.MirageApiUnsupported => error.Unsupported,
        error.MirageNotFound => error.Unsupported,
        error.OutOfMemory => error.OutOfMemory,
    };
}
