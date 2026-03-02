const std = @import("std");

const kernel = @import("../kernel.zig");
const pr = @import("../pr/pr.zig");
const mlir = @import("../c/mlir/mlir.zig");
const pass_mod = @import("pass.zig");
const mlir_passes = @import("mlir_passes.zig");

const log = std.log.scoped(.@"zg/mlir_materialize");

/// MLIR -> MLIR pass that materializes executable-scoped kernel artifacts from
/// pre-legalize `zigrad.kernel_call` operations.
///
/// Responsibilities:
/// 1. Parse `artifact.mlir.bytes` (the current MLIR state, post-select).
/// 2. Discover selected `zigrad.kernel_call` operations and extract
///    provider/key/pattern/type descriptors.
/// 3. Ensure each selected key has an artifact:
///    - prefer an existing registry entry,
///    - otherwise compile from the MLIR descriptor via provider `compile_mlir`.
/// 4. Populate `KernelPackage` by deterministic kernel id.
///
/// This pass must run after `MlirSelectPass` (which emits `zigrad.kernel_call`
/// ops) and before `MlirLegalizePass` (which converts them to
/// `stablehlo.custom_call`).
pub const MlirKernelMaterializePass = struct {
    registry: *kernel.KernelRegistry,
    package: *kernel.KernelPackage,
    providers: []const kernel.KernelProvider,
    /// Print a summary table of materialized kernel calls after the pass.
    dump_kernels: bool = false,

    pub fn pass(self: *MlirKernelMaterializePass) pass_mod.Pass {
        return .{
            .ptr = @ptrCast(self),
            .run_fn = run_impl,
            .name = "mlir_kernel_materialize",
            .input_kind = .mlir,
            .output_kind = .mlir,
        };
    }

    fn run_impl(
        ptr: *anyopaque,
        artifact: *pass_mod.Artifact,
        ctx: *pass_mod.PassContext,
    ) pass_mod.PassError!void {
        if (artifact.kind() != .mlir) return error.ArtifactKindMismatch;

        const self: *MlirKernelMaterializePass = @ptrCast(@alignCast(ptr));
        const mlir_artifact = &artifact.mlir;

        const kernel_calls = collect_kernel_calls_from_mlir(ctx.allocator, mlir_artifact.bytes) catch {
            log.err("failed to parse MLIR artifact for kernel materialization ({s} encoding, {d} bytes)", .{
                @tagName(mlir_artifact.encoding), mlir_artifact.bytes.len,
            });
            return error.InvalidMlir;
        };
        defer {
            for (kernel_calls) |*call| call.deinit(ctx.allocator);
            ctx.allocator.free(kernel_calls);
        }

        // Shape cache: maps structural signature (pattern + shapes) → compiled artifact
        // bytes. Regions with identical shapes across layers share one compiled binary,
        // avoiding redundant superoptimization passes.
        var shape_cache = std.StringHashMap(ShapeCacheEntry).init(ctx.allocator);
        defer {
            var it = shape_cache.iterator();
            while (it.next()) |entry| {
                ctx.allocator.free(entry.key_ptr.*);
                ctx.allocator.free(entry.value_ptr.data);
            }
            shape_cache.deinit();
        }

        var had_unsupported = false;
        for (kernel_calls) |*call| {
            self.materialize_selected_call(call, &shape_cache, ctx.allocator) catch |err| {
                if (err == error.Unsupported) {
                    log.warn(
                        "provider '{s}' returned Unsupported for key '{s}' (pattern={s}); expanding back to StableHLO",
                        .{ call.provider, call.kernel_key, @tagName(call.pattern) },
                    );
                    had_unsupported = true;
                    continue;
                }
                return err;
            };
        }

        // If any kernel_calls were unsupported, run the expand pass to revert
        // those kernel_call ops back to their original StableHLO patterns so
        // the backend can handle them natively.
        if (had_unsupported) {
            mlir_passes.run_pipeline_on_artifact(
                ctx.allocator,
                mlir_artifact,
                "func.func(zg-kernel-call-expand),canonicalize,cse",
            ) catch {
                log.err("failed to run kernel-call-expand pipeline", .{});
                return error.InvalidMlir;
            };
        }

        if (self.dump_kernels and kernel_calls.len > 0) {
            var buf: [8192]u8 = undefined;
            var stdout_writer = std.fs.File.stdout().writer(&buf);
            const out = &stdout_writer.interface;
            dump_mlir_kernel_calls(out, kernel_calls) catch {};
            out.flush() catch {};
        }

        mlir_artifact.kernel_package = self.package;
    }

    fn materialize_selected_call(
        self: *MlirKernelMaterializePass,
        call: *const KernelCallPlan,
        shape_cache: *std.StringHashMap(ShapeCacheEntry),
        temp_allocator: std.mem.Allocator,
    ) pass_mod.PassError!void {
        const kernel_id = kernel.kernel_id_from_key(call.kernel_key);
        if (self.package.get(kernel_id) != null) return;

        var compiled = self.registry.get(call.kernel_key);
        if (compiled == null) {
            const provider = self.find_provider(call.provider) orelse {
                log.err(
                    "selected provider '{s}' for key '{s}' is not registered",
                    .{ call.provider, call.kernel_key },
                );
                return error.ValidationFailed;
            };

            const shape_key = try compute_call_shape_key(temp_allocator, call);
            defer temp_allocator.free(shape_key);

            if (shape_cache.get(shape_key)) |cached| {
                log.debug("dedup cache hit: MLIR key '{s}' reuses compiled artifact", .{call.kernel_key});
                const reg_alloc = self.registry.allocator();
                const target_name = try reg_alloc.dupe(u8, call.kernel_key);
                errdefer reg_alloc.free(target_name);
                const cloned = kernel.KernelArtifact{
                    .provider_name = cached.provider_name,
                    .data = try reg_alloc.dupe(u8, cached.data),
                    .target_name = target_name,
                    .workspace_bytes = cached.workspace_bytes,
                    .dispatch_fn = cached.dispatch_fn,
                    .dispatch_ctx = cached.dispatch_ctx,
                };
                self.registry.put(cloned.target_name, cloned) catch |err| switch (err) {
                    error.DuplicateKey => {
                        log.err("duplicate registry key while materializing '{s}'", .{call.kernel_key});
                        return error.DuplicateKey;
                    },
                    error.OutOfMemory => return error.OutOfMemory,
                };
            } else {
                const desc = call.descriptor();
                log.debug(
                    "compiling MLIR-selected key '{s}' via provider '{s}' (pattern={s})",
                    .{ call.kernel_key, call.provider, @tagName(call.pattern) },
                );
                var compiled_from_mlir = provider.compile_mlir(desc, self.registry.allocator()) catch |err| {
                    if (err == error.Unsupported) {
                        log.err(
                            "provider '{s}' does not support MLIR compile for key '{s}' (pattern={s})",
                            .{ call.provider, call.kernel_key, @tagName(call.pattern) },
                        );
                    } else {
                        log.err(
                            "provider '{s}' failed MLIR compile for key '{s}': {s}",
                            .{ call.provider, call.kernel_key, @errorName(err) },
                        );
                    }
                    return err;
                };
                errdefer compiled_from_mlir.deinit(self.registry.allocator());

                if (!std.mem.eql(u8, compiled_from_mlir.provider_name, call.provider)) {
                    log.err(
                        "provider mismatch for key '{s}': selected '{s}', compiled '{s}'",
                        .{ call.kernel_key, call.provider, compiled_from_mlir.provider_name },
                    );
                    return error.ValidationFailed;
                }

                if (!std.mem.eql(u8, compiled_from_mlir.target_name, call.kernel_key)) {
                    log.err(
                        "compiled key mismatch: expected '{s}', got '{s}'",
                        .{ call.kernel_key, compiled_from_mlir.target_name },
                    );
                    return error.ValidationFailed;
                }

                // Populate shape cache so subsequent same-shape regions skip compilation.
                const cache_key = try temp_allocator.dupe(u8, shape_key);
                errdefer temp_allocator.free(cache_key);
                const cache_data = try temp_allocator.dupe(u8, compiled_from_mlir.data);
                errdefer temp_allocator.free(cache_data);
                try shape_cache.put(cache_key, .{
                    .data = cache_data,
                    .workspace_bytes = compiled_from_mlir.workspace_bytes,
                    .dispatch_fn = compiled_from_mlir.dispatch_fn,
                    .dispatch_ctx = compiled_from_mlir.dispatch_ctx,
                    .provider_name = compiled_from_mlir.provider_name,
                });

                self.registry.put(compiled_from_mlir.target_name, compiled_from_mlir) catch |err| switch (err) {
                    error.DuplicateKey => {
                        log.err("duplicate registry key while materializing '{s}'", .{call.kernel_key});
                        return error.DuplicateKey;
                    },
                    error.OutOfMemory => return error.OutOfMemory,
                };
            }
            compiled = self.registry.get(call.kernel_key);
        }

        const compiled_artifact = compiled orelse {
            log.err("selected key '{s}' could not be materialized", .{call.kernel_key});
            return error.ValidationFailed;
        };

        if (!std.mem.eql(u8, compiled_artifact.provider_name, call.provider)) {
            log.err(
                "provider mismatch for key '{s}': selected '{s}', compiled '{s}'",
                .{ call.kernel_key, call.provider, compiled_artifact.provider_name },
            );
            return error.ValidationFailed;
        }

        const pkg_artifact = clone_artifact_for_package(self.package.allocator(), compiled_artifact) catch
            return error.OutOfMemory;

        self.package.put(kernel_id, pkg_artifact) catch |err| switch (err) {
            error.DuplicateKey => {
                var owned = pkg_artifact;
                owned.deinit(self.package.allocator());
                log.err(
                    "duplicate kernel id {d} while materializing key '{s}'",
                    .{ kernel_id, call.kernel_key },
                );
                return error.DuplicateKey;
            },
            error.OutOfMemory => return error.OutOfMemory,
        };
    }

    fn find_provider(self: *const MlirKernelMaterializePass, name: []const u8) ?kernel.KernelProvider {
        for (self.providers) |provider| {
            if (std.mem.eql(u8, provider.name, name)) return provider;
        }
        return null;
    }

    fn dump_mlir_kernel_calls(out: *std.Io.Writer, calls: []const KernelCallPlan) !void {
        const fields = std.meta.fields(kernel.MlirKernelPattern);
        var counts: [fields.len]usize = [_]usize{0} ** fields.len;
        for (calls) |call| counts[@intFromEnum(call.pattern)] += 1;

        try out.print("kernels ({d} total):", .{calls.len});
        inline for (fields, 0..) |field, i| {
            if (counts[i] > 0) try out.print(" {s}={d}", .{ field.name, counts[i] });
        }
        try out.writeByte('\n');
        try out.print("  {s:<55} {s:<12} {s:<16} {s}\n", .{ "key", "provider", "pattern", "in0" });
        try out.writeAll("  " ++ ("-" ** 100) ++ "\n");
        for (calls) |call| {
            try out.print("  {s:<55} {s:<12} {s:<16} ", .{ call.kernel_key, call.provider, @tagName(call.pattern) });
            if (call.inputs.len > 0) {
                const d = call.inputs[0];
                try out.print("{s}[", .{@tagName(d.dtype)});
                for (d.dims, 0..) |dim, i| {
                    if (i > 0) try out.writeByte(',');
                    try out.print("{d}", .{dim});
                }
                try out.writeByte(']');
            } else {
                try out.writeAll("?");
            }
            try out.writeByte('\n');
        }
    }

    const KernelCallPlan = struct {
        provider: []const u8,
        kernel_key: []const u8,
        pattern: kernel.MlirKernelPattern,
        inputs: []kernel.MlirTensorDesc,
        outputs: []kernel.MlirTensorDesc,

        fn descriptor(self: *const KernelCallPlan) kernel.MlirKernelDescriptor {
            return .{
                .name = self.kernel_key,
                .provider_name = self.provider,
                .pattern = self.pattern,
                .inputs = self.inputs,
                .outputs = self.outputs,
            };
        }

        fn deinit(self: *KernelCallPlan, allocator: std.mem.Allocator) void {
            allocator.free(self.provider);
            allocator.free(self.kernel_key);

            for (self.inputs) |input| allocator.free(input.dims);
            allocator.free(self.inputs);

            for (self.outputs) |output| allocator.free(output.dims);
            allocator.free(self.outputs);

            self.* = undefined;
        }
    };

    /// Parse artifact bytes and collect all `zigrad.kernel_call` operations.
    ///
    /// Registers all required dialects including zigrad extensions so that
    /// both text and bytecode encodings can be parsed. The zigrad extension
    /// shim is already loaded by earlier passes (singleton with mutex guard),
    /// so the cost here is just dialect registration on this context.
    fn collect_kernel_calls_from_mlir(
        allocator: std.mem.Allocator,
        bytes: []const u8,
    ) ![]KernelCallPlan {
        var registry = mlir.Registry.init() catch return error.OutOfMemory;
        defer registry.deinit();

        mlir.DialectHandle.from_string("func").insert_dialect(registry);
        mlir.DialectHandle.from_string("stablehlo").insert_dialect(registry);

        var mlir_ctx = mlir.Context.init_with_registry(registry, false) catch return error.OutOfMemory;
        defer mlir_ctx.deinit();
        mlir_ctx.allow_unregistered_dialects(false);

        mlir.register_zigrad_extensions(mlir_ctx) catch {
            log.err("missing MLIR extension shim; cannot parse zigrad dialect ops", .{});
            return error.InvalidMlir;
        };

        const func_handle = mlir.DialectHandle.from_string("func");
        func_handle.register_dialect(mlir_ctx);
        _ = func_handle.load_dialect(mlir_ctx);

        const stablehlo_handle = mlir.DialectHandle.from_string("stablehlo");
        stablehlo_handle.register_dialect(mlir_ctx);
        _ = stablehlo_handle.load_dialect(mlir_ctx);

        var module = mlir.Module.parse_bytes(mlir_ctx, bytes) catch return error.InvalidMlir;
        defer module.deinit();

        var calls = try std.ArrayList(KernelCallPlan).initCapacity(allocator, 8);
        errdefer {
            for (calls.items) |*call| call.deinit(allocator);
            calls.deinit(allocator);
        }

        var seen = std.StringHashMap(void).init(allocator);
        defer {
            var key_it = seen.keyIterator();
            while (key_it.next()) |key| allocator.free(key.*);
            seen.deinit();
        }

        const WalkCtx = struct {
            allocator: std.mem.Allocator,
            calls: *std.ArrayList(KernelCallPlan),
            seen: *std.StringHashMap(void),
            err: ?anyerror = null,
        };

        var walk_ctx = WalkCtx{
            .allocator = allocator,
            .calls = &calls,
            .seen = &seen,
        };

        module.op().walk(.pre_order, &walk_ctx, struct {
            fn callback(ctx: anytype, op: mlir.Operation) mlir.Operation.WalkResult {
                const walk_ctx_ptr: *WalkCtx = @ptrCast(@constCast(ctx));
                if (walk_ctx_ptr.err != null) return .interrupt;
                if (!std.mem.eql(u8, op.name().str(), "zigrad.kernel_call")) return .advance;

                var plan = parse_kernel_call(walk_ctx_ptr.allocator, op) catch |err| {
                    walk_ctx_ptr.err = err;
                    return .interrupt;
                };

                const seen_key = std.fmt.allocPrint(walk_ctx_ptr.allocator, "{s}|{s}", .{ plan.provider, plan.kernel_key }) catch |err| {
                    plan.deinit(walk_ctx_ptr.allocator);
                    walk_ctx_ptr.err = err;
                    return .interrupt;
                };

                if (walk_ctx_ptr.seen.contains(seen_key)) {
                    walk_ctx_ptr.allocator.free(seen_key);
                    plan.deinit(walk_ctx_ptr.allocator);
                    return .advance;
                }

                walk_ctx_ptr.seen.put(seen_key, {}) catch |err| {
                    walk_ctx_ptr.allocator.free(seen_key);
                    plan.deinit(walk_ctx_ptr.allocator);
                    walk_ctx_ptr.err = err;
                    return .interrupt;
                };

                walk_ctx_ptr.calls.append(walk_ctx_ptr.allocator, plan) catch |err| {
                    plan.deinit(walk_ctx_ptr.allocator);
                    walk_ctx_ptr.err = err;
                    return .interrupt;
                };

                return .advance;
            }
        }.callback);

        if (walk_ctx.err) |err| return err;
        return calls.toOwnedSlice(allocator);
    }

    fn parse_kernel_call(allocator: std.mem.Allocator, op: mlir.Operation) !KernelCallPlan {
        const backend_config_attr = op.get_attribute_by_name("backend_config") orelse return error.InvalidMlir;
        const backend_config = attr_as_dict(backend_config_attr) orelse return error.InvalidMlir;

        const kernel_key = get_dict_string(backend_config, "zigrad.kernel_key") orelse return error.InvalidMlir;
        const provider = get_dict_string(backend_config, "zigrad.provider") orelse return error.InvalidMlir;

        const pattern = if (get_dict_string(backend_config, "zigrad.pattern")) |pattern_name|
            parse_pattern_name(pattern_name) orelse return error.Unsupported
        else
            infer_pattern_from_arity(op.num_operands()) orelse return error.Unsupported;

        var inputs = try allocator.alloc(kernel.MlirTensorDesc, op.num_operands());
        errdefer {
            for (inputs) |input| allocator.free(input.dims);
            allocator.free(inputs);
        }

        for (0..op.num_operands()) |idx| {
            inputs[idx] = try parse_tensor_desc(allocator, op.operand(idx).get_type());
        }

        var outputs = try allocator.alloc(kernel.MlirTensorDesc, op.num_results());
        errdefer {
            for (outputs) |output| allocator.free(output.dims);
            allocator.free(outputs);
        }

        for (0..op.num_results()) |idx| {
            outputs[idx] = try parse_tensor_desc(allocator, op.result(idx).get_type());
        }

        return .{
            .provider = try allocator.dupe(u8, provider),
            .kernel_key = try allocator.dupe(u8, kernel_key),
            .pattern = pattern,
            .inputs = inputs,
            .outputs = outputs,
        };
    }

    fn attr_as_dict(attr: mlir.Attribute) ?mlir.DictionaryAttribute {
        if (!attr.is_a(mlir.DictionaryAttribute)) return null;
        return .{ ._inner = attr._inner };
    }

    fn attr_as_string(attr: mlir.Attribute) ?mlir.StringAttribute {
        if (!attr.is_a(mlir.StringAttribute)) return null;
        return .{ ._inner = attr._inner };
    }

    fn get_dict_string(dict: mlir.DictionaryAttribute, key: [:0]const u8) ?[]const u8 {
        const attr = dict.get_by_name(key) orelse return null;
        const str_attr = attr_as_string(attr) orelse return null;
        return str_attr.value();
    }

    fn parse_pattern_name(name: []const u8) ?kernel.MlirKernelPattern {
        return if (std.mem.eql(u8, name, "dot"))
            .dot
        else if (std.mem.eql(u8, name, "dot_general"))
            .dot_general
        else if (std.mem.eql(u8, name, "dot_add"))
            .dot_add
        else if (std.mem.eql(u8, name, "dot_add_mul"))
            .dot_add_mul
        else if (std.mem.eql(u8, name, "dot_log"))
            .dot_log
        else if (std.mem.eql(u8, name, "dot_exp"))
            .dot_exp
        else
            null;
    }

    fn infer_pattern_from_arity(arity: usize) ?kernel.MlirKernelPattern {
        return switch (arity) {
            2 => .dot,
            3 => .dot_add,
            else => null,
        };
    }

    fn parse_tensor_desc(allocator: std.mem.Allocator, typ: mlir.Type) !kernel.MlirTensorDesc {
        const ranked = typ.as(mlir.RankedTensorType) orelse return error.Unsupported;
        const dtype = mlir_type_to_pr_dtype(ranked.get_element_type()) orelse return error.Unsupported;

        const dims = try allocator.alloc(usize, ranked.get_rank());
        errdefer allocator.free(dims);

        for (0..ranked.get_rank()) |idx| {
            const dim = ranked.get_dimension(idx);
            if (dim <= 0) return error.Unsupported;
            dims[idx] = @intCast(dim);
        }

        return .{
            .dtype = dtype,
            .dims = dims,
        };
    }

    fn mlir_type_to_pr_dtype(typ: mlir.Type) ?pr.DType {
        if (typ.as(mlir.FloatType(.bf16)) != null) return .bf16;
        if (typ.as(mlir.FloatType(.f32)) != null) return .f32;
        if (typ.as(mlir.FloatType(.f64)) != null) return .f64;

        if (typ.as(mlir.IntegerType(.i32)) != null or typ.as(mlir.IntegerType(.si32)) != null) return .i32;
        if (typ.as(mlir.IntegerType(.i64)) != null or typ.as(mlir.IntegerType(.si64)) != null) return .i64;
        if (typ.as(mlir.IntegerType(.u32)) != null) return .u32;
        if (typ.as(mlir.IntegerType(.u64)) != null) return .u64;

        return null;
    }

    /// Cached artifact bytes keyed by structural shape signature.
    ///
    /// Stores enough to clone a `KernelArtifact` for a new region with the same
    /// pattern and tensor shapes, avoiding redundant superoptimization passes.
    /// `data` is owned by the shape cache (allocated from `temp_allocator` in
    /// `run_impl`) and freed when the cache is torn down.
    const ShapeCacheEntry = struct {
        data: []const u8,
        workspace_bytes: usize,
        dispatch_fn: ?kernel.DispatchFn,
        dispatch_ctx: ?*anyopaque,
        /// Non-owning; points into the provider's own name storage.
        provider_name: []const u8,
    };

    /// Build a deterministic shape-signature string for a `KernelCallPlan`.
    ///
    /// Encodes `pattern;dtype[d0,d1,...]x...>dtype[d0,d1,...]x...` so that two
    /// plans with the same fusion pattern and tensor shapes produce the same key,
    /// regardless of their `kernel_key` (region name).
    fn compute_call_shape_key(allocator: std.mem.Allocator, call: *const KernelCallPlan) ![]const u8 {
        var buf = try std.ArrayList(u8).initCapacity(allocator, 64);
        errdefer buf.deinit(allocator);
        const w = buf.writer(allocator);
        try w.writeAll(@tagName(call.pattern));
        for (call.inputs) |inp| {
            try w.writeByte(';');
            try w.writeAll(@tagName(inp.dtype));
            try w.writeByte('[');
            for (inp.dims, 0..) |d, i| {
                if (i > 0) try w.writeByte(',');
                try w.print("{d}", .{d});
            }
            try w.writeByte(']');
        }
        try w.writeByte('>');
        for (call.outputs) |out| {
            try w.writeByte(';');
            try w.writeAll(@tagName(out.dtype));
            try w.writeByte('[');
            for (out.dims, 0..) |d, i| {
                if (i > 0) try w.writeByte(',');
                try w.print("{d}", .{d});
            }
            try w.writeByte(']');
        }
        return buf.toOwnedSlice(allocator);
    }

    fn clone_artifact_for_package(
        dst_allocator: std.mem.Allocator,
        artifact: kernel.KernelArtifact,
    ) !kernel.KernelArtifact {
        return .{
            .provider_name = artifact.provider_name,
            .data = try dst_allocator.dupe(u8, artifact.data),
            .target_name = artifact.target_name,
            .workspace_bytes = artifact.workspace_bytes,
            .dispatch_fn = artifact.dispatch_fn,
            .dispatch_ctx = artifact.dispatch_ctx,
        };
    }
};

test "mlir materialize parser extracts kernel call pattern and shapes" {
    const testing = std.testing;

    const text =
        "module {\n" ++
        "  func.func @main(%a: tensor<2x3xf32>, %b: tensor<3x2xf32>, %c: tensor<2x2xf32>) -> tensor<2x2xf32> {\n" ++
        "    %0 = \"zigrad.kernel_call\"(%a, %b, %c) {api_version = 4 : i32, call_target_name = \"zigrad.kernel.dispatch\", has_side_effect = false, backend_config = {zigrad.kernel_key = \"k0\", zigrad.provider = \"mirage\", zigrad.pattern = \"dot_add\"}} : (tensor<2x3xf32>, tensor<3x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>\n" ++
        "    return %0 : tensor<2x2xf32>\n" ++
        "  }\n" ++
        "}\n";

    const calls = try MlirKernelMaterializePass.collect_kernel_calls_from_mlir(testing.allocator, text);
    defer {
        for (calls) |*call| call.deinit(testing.allocator);
        testing.allocator.free(calls);
    }

    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expect(std.mem.eql(u8, calls[0].kernel_key, "k0"));
    try testing.expect(std.mem.eql(u8, calls[0].provider, "mirage"));
    try testing.expectEqual(kernel.MlirKernelPattern.dot_add, calls[0].pattern);
    try testing.expectEqual(@as(usize, 3), calls[0].inputs.len);
    try testing.expectEqual(@as(usize, 2), calls[0].inputs[0].dims[0]);
    try testing.expectEqual(@as(usize, 3), calls[0].inputs[0].dims[1]);
}

test "mlir materialize pass populates package from registry" {
    const testing = std.testing;

    var registry = kernel.KernelRegistry.init(testing.allocator);
    defer registry.deinit();

    var package = kernel.KernelPackage.init(testing.allocator);
    defer package.deinit();

    const data0 = try testing.allocator.dupe(u8, "abc");
    try registry.put("k0", .{
        .provider_name = "mirage",
        .data = data0,
        .target_name = "k0",
    });

    var pass_state = MlirKernelMaterializePass{
        .registry = &registry,
        .package = &package,
        .providers = &.{},
    };

    // Artifact bytes contain zigrad.kernel_call ops (post-select, pre-legalize).
    var artifact = pass_mod.Artifact{ .mlir = .{
        .bytes = try testing.allocator.dupe(
            u8,
            "module { func.func @main(%a: tensor<2x3xf32>, %b: tensor<3x2xf32>) -> tensor<2x2xf32> { %0 = \"zigrad.kernel_call\"(%a, %b) {api_version = 4 : i32, call_target_name = \"zigrad.kernel.dispatch\", has_side_effect = false, backend_config = {zigrad.kernel_key = \"k0\", zigrad.provider = \"mirage\", zigrad.pattern = \"dot\"}} : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32> return %0 : tensor<2x2xf32> } }\n",
        ),
        .encoding = .text,
    } };
    defer artifact.deinit(testing.allocator);

    var ctx = pass_mod.PassContext{ .allocator = testing.allocator };
    try pass_state.pass().run(&artifact, &ctx);

    const kernel_id = kernel.kernel_id_from_key("k0");
    try testing.expect(package.get(kernel_id) != null);
    try testing.expect(artifact.mlir.kernel_package == &package);
}

test "mlir materialize compiles missing key via provider compile_mlir" {
    const testing = std.testing;

    var registry = kernel.KernelRegistry.init(testing.allocator);
    defer registry.deinit();

    var package = kernel.KernelPackage.init(testing.allocator);
    defer package.deinit();

    var provider_state: u8 = 0;
    const TestProvider = struct {
        fn compile_pr(_: *anyopaque, _: kernel.RegionDescriptor, _: std.mem.Allocator) kernel.CompileError!kernel.KernelArtifact {
            return error.Unsupported;
        }

        fn compile_mlir(
            _: *anyopaque,
            desc: kernel.MlirKernelDescriptor,
            allocator: std.mem.Allocator,
        ) kernel.CompileError!kernel.KernelArtifact {
            return .{
                .provider_name = "mock",
                .data = try allocator.dupe(u8, "blob"),
                .target_name = try allocator.dupe(u8, desc.name),
            };
        }
    };

    const providers = [_]kernel.KernelProvider{.{
        .name = "mock",
        .ptr = @ptrCast(&provider_state),
        .compile_fn = TestProvider.compile_pr,
        .compile_mlir_fn = TestProvider.compile_mlir,
    }};

    var pass_state = MlirKernelMaterializePass{
        .registry = &registry,
        .package = &package,
        .providers = providers[0..],
    };

    var artifact = pass_mod.Artifact{ .mlir = .{
        .bytes = try testing.allocator.dupe(
            u8,
            "module { func.func @main(%a: tensor<2x3xf32>, %b: tensor<3x2xf32>) -> tensor<2x2xf32> { %0 = \"zigrad.kernel_call\"(%a, %b) {api_version = 4 : i32, call_target_name = \"zigrad.kernel.dispatch\", has_side_effect = false, backend_config = {zigrad.kernel_key = \"k_mlir\", zigrad.provider = \"mock\", zigrad.pattern = \"dot\"}} : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32> return %0 : tensor<2x2xf32> } }\n",
        ),
        .encoding = .text,
    } };
    defer artifact.deinit(testing.allocator);

    var ctx = pass_mod.PassContext{ .allocator = testing.allocator };
    try pass_state.pass().run(&artifact, &ctx);

    try testing.expect(registry.get("k_mlir") != null);
    const kernel_id = kernel.kernel_id_from_key("k_mlir");
    try testing.expect(package.get(kernel_id) != null);
}
