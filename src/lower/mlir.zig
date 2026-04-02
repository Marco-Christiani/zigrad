//! MLIR Lowering Infrastructure
//!
//! Dialect-agnostic scaffold and shared MLIR passes, plus dialect-specific
//!  sub-modules.
//!
//! Shared:
//!  - context: generic mlir lowering scaffold (LowerContext, LowerOpFn, type mapping)
//!  - session: MLIR registry + context lifecycle
//!  - passes: run_pipeline_on_artifact, MlirSelectPass
//!
//! Dialect-specific:
//! - stablehlo: PR -> StableHLO lowering + legalize pass
pub const context = @import("mlir/context.zig");
pub const session = @import("mlir/session.zig");
pub const passes = @import("mlir/passes.zig");

// TODO: do we want this to be dialect-agnostic? as of now, it isnt, so this is problematic.
pub const MlirSelectPass = passes.MlirSelectPass;
pub const run_pipeline_on_artifact = passes.run_pipeline_on_artifact;

pub const stablehlo = @import("mlir/stablehlo.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
