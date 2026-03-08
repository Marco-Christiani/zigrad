/// MLIR Lowering Infrastructure
///
/// Shared MLIR passes (dialect-agnostic, work on zigrad dialect ops)
/// and dialect-specific sub-modules.
///
/// Shared:
/// - passes: run_pipeline_on_artifact, MlirSelectPass
/// - materialize: MlirKernelMaterializePass
///
/// Dialect-specific:
/// - stablehlo: PR -> StableHLO lowering + legalize pass
pub const passes = @import("passes.zig");
pub const materialize = @import("materialize.zig");

pub const MlirSelectPass = passes.MlirSelectPass;
pub const MlirKernelMaterializePass = materialize.MlirKernelMaterializePass;
pub const run_pipeline_on_artifact = passes.run_pipeline_on_artifact;

pub const stablehlo = @import("stablehlo/root.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
