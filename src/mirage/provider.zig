const std = @import("std");
const kernel = @import("../kernel.zig");
const pr = @import("../pr/pr.zig");
const dispatch_mod = @import("dispatch.zig");
const artifact_mod = @import("artifact.zig");
const mirage = @import("../c/mirage/api.zig");
const mlir_types = @import("mlir.zig");
const TypedPtr = @import("../utils/rtti.zig").TypedPtr;

const log = std.log.scoped(.@"zg/mirage_provider");
const superopt_max_candidates: u32 = 1024;
const max_region_eqns: usize = 5;

pub const MirageProvider = struct {
    allocator: std.mem.Allocator,
    dispatch_state: *dispatch_mod.MirageDispatchState,
    device_ordinal: i32 = 0,

    pub fn kernel_provider(self: *MirageProvider) kernel.KernelProvider {
        return .{
            .name = "mirage",
            .ptr = @ptrCast(self),
            .compile_fn = compile_impl,
            .dispatch_fn = &dispatch_mod.MirageDispatchState.dispatch,
            .dispatch_ctx = TypedPtr.init(self.dispatch_state),
        };
    }

    fn compile_impl(ptr: *anyopaque, desc: kernel.RegionDescriptor, _: kernel.CompileContext, allocator: std.mem.Allocator) kernel.CompileError!kernel.KernelArtifact {
        const self: *MirageProvider = @ptrCast(@alignCast(ptr));
        return self.compile(desc, allocator);
    }

    fn compile(self: *MirageProvider, desc: kernel.RegionDescriptor, allocator: std.mem.Allocator) kernel.CompileError!kernel.KernelArtifact {
        if (desc.ops.len > max_region_eqns) {
            log.debug(
                "region '{s}' has {d} ops (> {d}); skipping mirage compile",
                .{ desc.name, desc.ops.len, max_region_eqns },
            );
            return error.Unsupported;
        }

        var graph = mirage.Graph.init() catch |err| return map_mirage_api_error(err);
        defer graph.deinit();

        var tensor_map = std.AutoHashMap(*const pr.Var, mirage.Tensor).init(allocator);
        defer tensor_map.deinit();

        try lower_region_graph(desc, &graph, &tensor_map);

        return self.search_and_transpile(desc.name, allocator, &graph);
    }

    pub fn compile_mlir(
        self: *MirageProvider,
        desc: mlir_types.MlirKernelDescriptor,
        allocator: std.mem.Allocator,
    ) kernel.CompileError!kernel.KernelArtifact {
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

        var graph = mirage.Graph.init() catch |err| return map_mirage_api_error(err);
        defer graph.deinit();

        var handles = try std.ArrayList(mirage.Tensor).initCapacity(allocator, desc.inputs.len);
        defer handles.deinit(allocator);

        for (desc.inputs) |input_desc| {
            const handle = try emit_graph_input(&graph, input_desc);
            try handles.append(allocator, handle);
        }

        const out_tensor = build_mirage_graph(&graph, desc.pattern, handles.items, desc) catch |err| {
            log.err("graph construction failed for '{s}' (pattern={s}): {s}", .{
                desc.name, @tagName(desc.pattern), @errorName(err),
            });
            return err;
        };

        graph.mark_output(out_tensor) catch |err| return map_mirage_api_error(err);

        return self.search_and_transpile(desc.name, allocator, &graph);
    }

    /// Build the Mirage graph for a given kernel pattern.
    ///
    /// Translates the pattern-specific op sequence into Mirage graph ops.
    /// Returns `Unsupported` if Mirage rejects the tensor shapes (e.g.
    /// non-canonical matmul layout); callers can fall back to baseline lowering.
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
                // Mirage graph omits scaling -- the superoptimizer works on the
                // structural pattern. If Mirage returns Unsupported, the expand
                // pass reconstructs the full chain WITH scale correctly.
                const scores = try emit_matmul(graph, input_handles[0], input_handles[1]);
                const exp_result = try emit_unary(graph, .exp, scores);
                const sum_result = graph.reduction(exp_result, desc.reduction_dim, desc.reduction_factor) catch |err|
                    return map_mirage_api_error(err);
                const attn_probs = try emit_binary(graph, .div, exp_result, sum_result);
                return try emit_matmul(graph, attn_probs, input_handles[2]);
            },
        }
    }

    /// Run search on the graph and transpile the best candidate to CUDA source.
    fn search_and_transpile(
        self: *MirageProvider,
        target_name: []const u8,
        allocator: std.mem.Allocator,
        graph: *mirage.Graph,
    ) kernel.CompileError!kernel.KernelArtifact {
        // Search for optimized candidates.
        var device = mirage.Device.init(self.device_ordinal) catch |err| return map_mirage_api_error(err);
        defer device.deinit();

        // TODO: proper init or .empty-style pattern
        var search_opts: mirage.SearchOptions = std.mem.zeroes(mirage.SearchOptions);
        search_opts.max_candidates = superopt_max_candidates;

        var result = mirage.search(&device, graph, &search_opts) catch |err| {
            if (err == error.MirageApiUnsupported) {
                log.debug("mirage search unsupported for region '{s}'", .{target_name});
                return error.Unsupported;
            }
            return map_mirage_api_error(err);
        };
        defer result.deinit();

        // If search found no valid execution strategies, skip transpile.
        // The raw graph is not directly transpilable -- it needs a search-
        // discovered threadblock decomposition to produce a kernel.
        const num_candidates = result.count();
        if (num_candidates == 0) {
            log.debug("mirage search found 0 valid graphs for '{s}'; falling back", .{target_name});
            return error.Unsupported;
        }

        // Mirage's transpiler has unimplemented code paths that abort() the
        // process. Probe each top candidate in a forked child to detect
        // crashes before committing to a transpile in the parent.
        const max_probes = @min(num_candidates, 5);
        const transpile_graph = for (0..max_probes) |i| {
            const g = result.get(i) orelse continue;
            if (probe_transpile_safe(g)) break g;
            log.warn("mirage transpile probe crashed for '{s}' (candidate {d}); trying next", .{ target_name, i });
        } else {
            log.warn("top {d} mirage candidates all crashed for '{s}'; falling back", .{ max_probes, target_name });
            return error.Unsupported;
        };

        // Transpile in parent -- safe because the same deterministic graph
        // passed the fork-canary probe above.
        var source = mirage.transpile(transpile_graph, null) catch |err| {
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
            return error.ProviderCallFailed;
        }

        const buf_size = source.buf_size();
        if (buf_size != 0) {
            log.debug("mirage workspace for '{s}': {d} bytes", .{ target_name, buf_size });
        }

        // If the transpiled source has no custom kernels (only library ops
        // like standalone matmul -> cuBLAS), we can't launch via NVRTC.
        // Return Unsupported so the backend handles this natively.
        const num_kernels = source.num_kernels();
        if (num_kernels == 0) {
            log.debug("mirage transpile produced 0 custom kernels for '{s}'; falling back to backend", .{target_name});
            return error.Unsupported;
        }

        // Filter source for NVRTC (strip host code, replace runtime.h).
        const filtered = dispatch_mod.filter_source_for_nvrtc(allocator, cuda_code) catch {
            log.err("failed to filter source for '{s}'", .{target_name});
            return error.ProviderCallFailed;
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
            const meta = source.kernel_meta(ki) catch |err| return map_mirage_api_error(err);
            const num_args = source.kernel_num_args(ki);

            var args = try allocator.alloc(artifact_mod.KernelArg, num_args);
            for (0..num_args) |ai| {
                const arg = source.kernel_arg(ki, ai) catch |err| {
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
            return error.ProviderCallFailed;
        };

        return .{
            .provider_name = "mirage",
            .data = artifact_data,
            .target_name = try allocator.dupe(u8, target_name),
            .workspace_bytes = buf_size,
        };
    }
};

/// Test if transpiling a graph is safe by running it in a forked child.
///
/// Mirage's transpiler has unimplemented code paths that call `assert(false)`,
/// triggering `abort()` which kills the process. Since `abort()` cannot be
/// caught from Zig, we fork a canary process: the child attempts the transpile
/// and exits with status 0 on success. If the child is killed by a signal
/// (SIGABRT from the assert), the parent detects this via `waitpid` and
/// returns false. The parent then re-transpiles the same deterministic graph
/// when it is confirmed safe.
///
/// Uses raw posix `fork`/`dup2`/`waitpid` because the canary semantics need
///  direct kernel-level fork to share the parent's address space (the C++
///  state inside libmirage is not re-creatable from `std.process.run`). An
///  `io`-threaded variant of `fork` does not exist.
fn probe_transpile_safe(graph: ?*const mirage.RawGraph) bool {
    const pid = std.posix.fork() catch |err| {
        log.warn("fork failed for transpile probe: {}", .{err});
        return false;
    };

    if (pid == 0) {
        // Child: suppress stderr (hides the assert message) and attempt
        // transpile. Use exit_group to avoid running atexit handlers.
        if (std.posix.open("/dev/null", .{ .ACCMODE = .WRONLY }, 0)) |devnull| {
            std.posix.dup2(devnull, std.posix.STDERR_FILENO) catch {};
            std.posix.close(devnull);
        } else |_| {}

        var source = mirage.transpile(graph, null) catch {
            std.os.linux.exit_group(1);
        };
        source.deinit();
        std.os.linux.exit_group(0);
    }

    // Parent: wait for the canary child.
    const wait = std.posix.waitpid(pid, 0);
    const W = std.os.linux.W;
    return W.IFEXITED(wait.status) and W.EXITSTATUS(wait.status) == 0;
}

fn emit_graph_input(
    graph: *mirage.Graph,
    input_desc: mlir_types.MlirTensorDesc,
) kernel.CompileError!mirage.Tensor {
    const dtype = dtype_to_mirage(input_desc.dtype) orelse {
        log.debug("unsupported dtype for mirage input: {s}", .{@tagName(input_desc.dtype)});
        return error.Unsupported;
    };

    var dims: [mirage.max_rank]i64 = .{ 0, 0, 0, 0 };
    for (input_desc.dims, 0..) |dim, idx| {
        if (dim == 0 or dim > std.math.maxInt(i64)) return error.Unsupported;
        dims[idx] = @intCast(dim);
    }

    const spec = mirage.TensorSpec{
        .dtype = @intFromEnum(dtype),
        .rank = @intCast(input_desc.dims.len),
        .dims = dims,
        .strides = .{ 0, 0, 0, 0 },
    };

    return graph.new_input(&spec) catch |err| {
        log.debug("mirage graph.new_input rejected ({s} rank={d}): {s}", .{
            @tagName(dtype), input_desc.dims.len, @errorName(err),
        });
        return map_mirage_api_error(err);
    };
}

fn lower_region_graph(
    desc: kernel.RegionDescriptor,
    graph: *mirage.Graph,
    tensor_map: *std.AutoHashMap(*const pr.Var, mirage.Tensor),
) kernel.CompileError!void {
    for (desc.inputs) |in_var| {
        const tensor = in_var.as_tensor();

        const dtype = dtype_to_mirage(tensor.dtype) orelse return error.Unsupported;

        var dims: [mirage.max_rank]i64 = .{ 0, 0, 0, 0 };
        for (tensor.shape.dims, 0..) |dim, idx| {
            if (dim == 0 or dim > std.math.maxInt(i64)) return error.Unsupported;
            dims[idx] = @intCast(dim);
        }

        const spec = mirage.TensorSpec{
            .dtype = @intFromEnum(dtype),
            .rank = @intCast(tensor.shape.dims.len),
            .dims = dims,
            .strides = .{ 0, 0, 0, 0 },
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
        log.err("mirage graph.matmul rejected (lhs={d}, rhs={d}): {s}", .{ lhs, rhs, @errorName(err) });
        return map_mirage_api_error(err);
    };
}

fn emit_unary(
    graph: *mirage.Graph,
    op: mirage.UnaryOp,
    input: mirage.Tensor,
) kernel.CompileError!mirage.Tensor {
    return graph.unary(op, input) catch |err| {
        log.err("mirage graph.unary({s}) rejected (input={d}): {s}", .{ @tagName(op), input, @errorName(err) });
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
        log.err("mirage graph.binary({s}) rejected (lhs={d}, rhs={d}): {s}", .{ @tagName(op), lhs, rhs, @errorName(err) });
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
        error.MirageUnavailable => {
            log.warn("remapping {s} -> ProviderLoadFailed", .{@errorName(err)});
            return error.ProviderLoadFailed;
        },
        error.MirageInvalidArgument => error.Unsupported,
        error.MirageInternalError => {
            log.warn("remapping {s} -> ProviderCallFailed", .{@errorName(err)});
            return error.ProviderCallFailed;
        },
        error.MirageApiUnsupported => {
            log.warn("remapping {s} -> ProviderCallFailed", .{@errorName(err)});
            return error.ProviderCallFailed;
        },
        error.MirageNotFound => error.Unsupported,
        error.OutOfMemory => error.OutOfMemory,
    };
}

fn map_mirage_status(status: mirage.Status, ctx: StatusContext) kernel.CompileError {
    return switch (status) {
        .invalid_argument => error.Unsupported,
        .internal_error => {
            if (!@import("builtin").is_test) log.warn("status_internal_error -> ProviderCallFailed", .{});
            return error.ProviderCallFailed;
        },
        .unsupported => switch (ctx) {
            .region => error.Unsupported,
            .runtime => {
                if (!@import("builtin").is_test) log.warn("status_unsupported (runtime) -> ProviderCallFailed", .{});
                return error.ProviderCallFailed;
            },
        },
        .not_found => error.Unsupported,
        else => {
            log.warn("unknown status {d} -> ProviderCallFailed", .{@intFromEnum(status)});
            return error.ProviderCallFailed;
        },
    };
}

const StatusContext = enum {
    region,
    runtime,
};

test map_mirage_status {
    try std.testing.expectEqual(error.Unsupported, map_mirage_status(.invalid_argument, .region));
    try std.testing.expectEqual(error.ProviderCallFailed, map_mirage_status(.internal_error, .region));
    try std.testing.expectEqual(error.Unsupported, map_mirage_status(.unsupported, .region));

    try std.testing.expectEqual(error.Unsupported, map_mirage_status(.invalid_argument, .runtime));
    try std.testing.expectEqual(error.ProviderCallFailed, map_mirage_status(.internal_error, .runtime));
    try std.testing.expectEqual(error.ProviderCallFailed, map_mirage_status(.unsupported, .runtime));
}
