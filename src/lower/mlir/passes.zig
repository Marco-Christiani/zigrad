//! Shared MLIR-stage passes for the pipeline.
//!
//! These passes operate on serialized MLIR artifacts: they parse the current
//!  artifact bytes, run an MLIR pass pipeline, verify the result, and
//!  re-serialize back into the artifact.
//!
//! Each pass is a standalone `.mlir -> .mlir` transformation in the pipeline.
//!
//! This module contains dialect-agnostic passes that work on zigrad dialect
//!  ops regardless of the target MLIR dialect. StableHLO-specific passes
//!  (e.g. legalize) live in `stablehlo/legalize.zig`.
const std = @import("std");
const mlir = @import("../../c/mlir/mlir.zig");
const pass = @import("../../pipeline/pass.zig");
const MlirSession = @import("session.zig").MlirSession;
const Writer = std.Io.Writer;
const MlirArtifact = pass.MlirArtifact;
const PassError = pass.PassError;
const Pass = pass.Pass;
const PassContext = pass.PassContext;
const Artifact = pass.Artifact;

const log = std.log.scoped(.@"zg/mlir_passes");

pub const zigrad_kernel_select_pipeline: [:0]const u8 = "canonicalize,cse,func.func(zg-mirage-kernel-select),canonicalize,cse";

/// Run an MLIR pass pipeline on the current artifact bytes.
///
/// The caller provides a fully-configured `MlirSession` with all required
///  dialects already loaded (the session owns the MLIR context lifetime).
///  This function handles parsing, pipeline execution, verification, and
///  re-serialization.
///
/// The artifact encoding is preserved across the transformation.
/// TODO: consider if this is clear as a method.
pub fn run_pipeline_on_artifact(
    allocator: std.mem.Allocator,
    session: MlirSession,
    mlir_artifact: *pass.MlirArtifact,
    pipeline_str: [:0]const u8,
) PassError!void {
    const ctx = session.ctx;

    var module = mlir.Module.parse_bytes(ctx, mlir_artifact.bytes) catch {
        log.err("failed to parse MLIR artifact ({s} encoding, {d} bytes)", .{
            @tagName(mlir_artifact.encoding), mlir_artifact.bytes.len,
        });
        return error.InvalidMlir;
    };
    defer module.deinit();

    var pm = mlir.PassManager.init(ctx) catch @panic("Failed to init PassManager in run_pipeline_on_artifact().");
    defer pm.deinit();

    var op_pm = pm.as_op_pass_manager();
    op_pm.add_pipeline(pipeline_str) catch |e| {
        log.err("failed to add MLIR pass pipeline: {s}", .{pipeline_str});
        return e;
    };
    pm.run_on_op(module.op()) catch |e| {
        log.err("MLIR pass pipeline failed: {s}", .{pipeline_str});
        return e;
    };

    if (!module.op().verify()) {
        log.err("MLIR verification failed after pipeline: {s}", .{pipeline_str});
        return error.InvalidMlir;
    }

    var writer_state = Writer.Allocating.init(allocator);
    defer writer_state.deinit();

    switch (mlir_artifact.encoding) {
        .bytecode => module.op().write_bytecode(&writer_state.writer) catch |e| switch (e) {
            inline else => |ee| @panic("Serialization failed got " ++ @errorName(ee) ++ " in run_pipeline_on_artifact()."),
        },
        .text => module.op().print(&writer_state.writer, .{}) catch |e| switch (e) {
            inline else => |ee| @panic("Serialization failed got " ++ @errorName(ee) ++ " in run_pipeline_on_artifact()."),
        },
    }

    const new_bytes = try writer_state.toOwnedSlice();
    allocator.free(mlir_artifact.bytes);
    mlir_artifact.bytes = new_bytes;
}

// ============================================================================
// Select Pass
// ============================================================================

/// MLIR -> MLIR pass: kernel selection via greedy pattern matching.
///
/// Runs `zg-mirage-kernel-select` which structurally matches fuseable
///  StableHLO patterns (dot+add, dot+exp, etc.) and rewrites them into
///  `zigrad.kernel_call` operations for the Mirage kernel provider.
/// Canonicalize and CSE run before and after selection.
///
/// After this pass, the artifact contains `zigrad.kernel_call` ops that
///  represent selected kernel candidates. These must be legalized (converted
///  to `stablehlo.custom_call`) before backend compilation.
/// TODO: this does NOT belong here its both KP specific and dialect specific
pub const MlirSelectPass = struct {
    pub fn pass() Pass {
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
        artifact: *Artifact,
        ctx: *PassContext,
    ) PassError!void {
        if (artifact.kind() != .mlir) return error.ArtifactKindMismatch;
        // TODO: this is a misleading error but stems from a shortcoming in our interface design,
        //  see comments in pass.zig for ideas to fix this.
        var session = MlirSession.init() catch |e| {
            log.err("MLIR context initialization failed: {s}", .{@errorName(e)});
            return error.InvalidMlir;
        };
        defer session.deinit();
        session.load_dialect("stablehlo");
        try run_pipeline_on_artifact(ctx.allocator, session, &artifact.mlir, zigrad_kernel_select_pipeline);
    }
};
