//! Pass-pipeline execution over serialized MLIR representations.
//!
//! `run_pipeline` parses the artifact, runs a caller-supplied pipeline,
//!  verifies the result, and serializes it using the input encoding.
const std = @import("std");
const mlir = @import("../c/mlir/mlir.zig");
const stablehlo = @import("../stablehlo.zig");
const MlirSession = @import("session.zig").Session;
const Writer = std.Io.Writer;

const log = std.log.scoped(.@"zg/mlir_passes");

/// Failures while parsing, running, or serializing an MLIR pass pipeline.
pub const PipelineError = mlir.Error || std.Io.Writer.Error;

/// Run an MLIR pass pipeline on a StableHLO artifact.
///
/// Replaces `artifact.bytes` only after the pipeline and serialization succeed.
pub fn run_pipeline(
    allocator: std.mem.Allocator,
    session: MlirSession,
    artifact: *stablehlo.Artifact,
    pipeline_str: [:0]const u8,
) PipelineError!void {
    const ctx = session.ctx;

    var module = mlir.Module.parse_bytes(ctx, artifact.bytes) catch {
        log.err("failed to parse stablehlo artifact ({s} encoding, {d} bytes)", .{
            @tagName(artifact.encoding), artifact.bytes.len,
        });
        return error.InvalidMlir;
    };
    defer module.deinit();

    var pm = try mlir.PassManager.init(ctx);
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

    switch (artifact.encoding) {
        .binary => try module.op().write_bytecode(&writer_state.writer),
        .text => try module.op().print(&writer_state.writer, .{}),
    }

    const new_bytes = try writer_state.toOwnedSlice();
    allocator.free(artifact.bytes);
    artifact.bytes = new_bytes;
}
