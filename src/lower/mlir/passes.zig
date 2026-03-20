/// Shared MLIR-stage passes for the pipeline.
///
/// These passes operate on serialized MLIR artifacts: they parse the current
/// artifact bytes, run an MLIR pass pipeline, verify the result, and
/// re-serialize back into the artifact. Each pass is a standalone
/// `.mlir -> .mlir` transformation in the pipeline.
///
/// This module contains dialect-agnostic passes that work on zigrad dialect
/// ops regardless of the target MLIR dialect. StableHLO-specific passes
/// (e.g. legalize) live in `stablehlo/legalize.zig`.
const std = @import("std");

const mlir = @import("../../c/mlir/mlir.zig");
const pass_mod = @import("../../pipeline/pass.zig");

const log = std.log.scoped(.@"zg/mlir_passes");

pub const zigrad_kernel_select_pipeline: [:0]const u8 = "canonicalize,cse,func.func(zg-mirage-kernel-select),canonicalize,cse";

/// Run an MLIR pass pipeline on the current artifact bytes.
///
/// Handles context setup, dialect/extension registration, parsing (text or
/// bytecode), pipeline execution, verification, and re-serialization. The
/// artifact encoding is preserved across the transformation.
pub fn run_pipeline_on_artifact(
    allocator: std.mem.Allocator,
    mlir_artifact: *pass_mod.MlirArtifact,
    pipeline_str: [:0]const u8,
) pass_mod.PassError!void {
    var registry = mlir.Registry.init() catch return error.OutOfMemory;
    defer registry.deinit();

    mlir.DialectHandle.from_string("func").insert_dialect(registry);
    mlir.DialectHandle.from_string("stablehlo").insert_dialect(registry);

    var ctx = mlir.Context.init_with_registry(registry, false) catch return error.OutOfMemory;
    defer ctx.deinit();
    ctx.allow_unregistered_dialects(false);

    mlir.register_zigrad_extensions(ctx) catch {
        log.err("missing MLIR extension shim; cannot run MLIR passes", .{});
        return error.InvalidMlir;
    };

    const func_handle = mlir.DialectHandle.from_string("func");
    func_handle.register_dialect(ctx);
    _ = func_handle.load_dialect(ctx);

    const stablehlo_handle = mlir.DialectHandle.from_string("stablehlo");
    stablehlo_handle.register_dialect(ctx);
    _ = stablehlo_handle.load_dialect(ctx);

    var module = mlir.Module.parse_bytes(ctx, mlir_artifact.bytes) catch {
        log.err("failed to parse MLIR artifact ({s} encoding, {d} bytes)", .{
            @tagName(mlir_artifact.encoding), mlir_artifact.bytes.len,
        });
        return error.InvalidMlir;
    };
    defer module.deinit();

    var pm = mlir.PassManager.init(ctx) catch return error.InvalidMlir;
    defer pm.deinit();

    var op_pm = pm.as_op_pass_manager();
    op_pm.add_pipeline(pipeline_str) catch {
        log.err("failed to add MLIR pass pipeline: {s}", .{pipeline_str});
        return error.InvalidMlir;
    };
    pm.run_on_op(module.op()) catch {
        log.err("MLIR pass pipeline failed: {s}", .{pipeline_str});
        return error.InvalidMlir;
    };

    if (!module.op().verify()) {
        log.err("MLIR verification failed after pipeline: {s}", .{pipeline_str});
        return error.InvalidMlir;
    }

    var writer_state = std.Io.Writer.Allocating.init(allocator);
    defer writer_state.deinit();

    switch (mlir_artifact.encoding) {
        .bytecode => module.op().write_bytecode(&writer_state.writer) catch return error.OutOfMemory,
        .text => module.op().print(&writer_state.writer, .{}) catch return error.OutOfMemory,
    }

    const new_bytes = writer_state.toOwnedSlice() catch return error.OutOfMemory;
    allocator.free(mlir_artifact.bytes);
    mlir_artifact.bytes = new_bytes;
}

// ============================================================================
// Select Pass
// ============================================================================

/// MLIR -> MLIR pass: kernel selection via greedy pattern matching.
///
/// Runs `zg-mirage-kernel-select` which structurally matches fuseable
/// StableHLO patterns (dot+add, dot+exp, etc.) and rewrites them into
/// `zigrad.kernel_call` operations for the Mirage kernel provider.
/// Canonicalize and CSE run before and after selection.
///
/// After this pass, the artifact contains `zigrad.kernel_call` ops that
/// represent selected kernel candidates. These must be legalized (converted
/// to `stablehlo.custom_call`) before backend compilation.
pub const MlirSelectPass = struct {
    pub fn pass() pass_mod.Pass {
        return .{
            .ptr = undefined,
            .run_fn = run_impl,
            .name = "mlir_kernel_select",
            .input_kind = .mlir,
            .output_kind = .mlir,
        };
    }

    fn run_impl(
        _: *anyopaque,
        artifact: *pass_mod.Artifact,
        ctx: *pass_mod.PassContext,
    ) pass_mod.PassError!void {
        if (artifact.kind() != .mlir) return error.ArtifactKindMismatch;
        try run_pipeline_on_artifact(ctx.allocator, &artifact.mlir, zigrad_kernel_select_pipeline);
    }
};
