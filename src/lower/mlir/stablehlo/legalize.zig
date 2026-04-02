/// StableHLO Legalize Pass
///
/// Converts `zigrad` dialect operations to `stablehlo`.
/// When using the kernelize feature, this pass must run after MLIR-level
///  selection or PR-level lowering and before backend compilation.
/// NOTE: given recent changes this is actually deserving of some scrutiny,
///  we can keep this logic in Zigrad almost certainly for the PR case and
///  there is a good chance we can do so for the other case although that
///  would be a larger lift and requires osme investigation
const std = @import("std");

const pass_mod = @import("../../../pipeline/pass.zig");
const mlir_passes = @import("../passes.zig");
const MlirSession = @import("../session.zig").MlirSession;

const log = std.log.scoped(.@"zg/legalize_stablehlo");

// TODO: should we rename the zg-kernel-legalize pass? have to look into pass configurability.
const stablehlo_legalize_pipeline: [:0]const u8 = "func.func(zg-kernel-legalize),canonicalize,cse";

/// MLIR -> MLIR pass
///
/// Currently, this legalizes `zigrad.kernel_call` to `stablehlo.custom_call`,
///  but expansion is likely. For as long as that is true, then for programs
///  with no `zigrad.kernel_call` ops, this pass is a no-op.
pub const StablehloLegalizePass = struct {
    pub fn pass() pass_mod.Pass {
        return .{
            .ptr = undefined,
            .run_fn = run_impl,
            .name = "stablehlo_legalize",
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
        var session = MlirSession.init() catch |e| {
            log.err("MLIR context initialization failed: {s}", .{@errorName(e)});
            return error.InvalidMlir;
        };
        defer session.deinit();
        session.load_dialect("stablehlo");
        try mlir_passes.run_pipeline_on_artifact(ctx.allocator, session, &artifact.mlir, stablehlo_legalize_pipeline);
    }
};
