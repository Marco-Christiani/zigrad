/// StableHLO Legalize Pass
///
/// Converts `zigrad.kernel_call` operations to `stablehlo.custom_call`,
/// which the XLA/PJRT backend understands. This pass must run after
/// MLIR-level selection or PR-level lowering and before backend compilation.
const std = @import("std");

const pass_mod = @import("../../../pipeline/pass.zig");
const mlir_passes = @import("../passes.zig");

const zigrad_kernel_legalize_pipeline: [:0]const u8 = "func.func(zg-kernel-legalize),canonicalize,cse";

/// MLIR -> MLIR pass: legalize `zigrad.kernel_call` to `stablehlo.custom_call`.
///
/// For programs with no `zigrad.kernel_call` ops, this pass is a no-op.
pub const MlirLegalizePass = struct {
    pub fn pass() pass_mod.Pass {
        return .{
            .ptr = undefined,
            .run_fn = run_impl,
            .name = "mlir_kernel_legalize",
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
        try mlir_passes.run_pipeline_on_artifact(ctx.allocator, &artifact.mlir, zigrad_kernel_legalize_pipeline);
    }
};
