/// Lower Module
///
/// Lowering passes that transform PR into target-specific IR.
///
/// Currently provides:
/// - stablehlo: PR -> StableHLO/MLIR lowering
///
/// See: .internal/2026-01-16-03_PASS_BASED_PIPELINE.md
pub const stablehlo = @import("stablehlo.zig");

// Re-export pass functions
pub const lowerPass = stablehlo.lowerPass;
pub const LowerPassConfig = stablehlo.LowerPassConfig;
pub const lowerPassWithConfig = stablehlo.lowerPassWithConfig;
pub const validatePass = stablehlo.validatePass;
pub const validate_pass = stablehlo.validate_pass;

// Re-export direct lowering API
pub const lower = stablehlo.lower;
pub const lowerProgramToMlir = stablehlo.lowerProgramToMlir;
pub const lowerFunctionToMlir = stablehlo.lowerFunctionToMlir;
pub const OutputFormat = stablehlo.OutputFormat;
