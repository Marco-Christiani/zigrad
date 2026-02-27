/// Lower Module
///
/// Lowering passes that transform PR into target-specific IR.
///
/// Currently provides:
/// - stablehlo: PR -> StableHLO/MLIR lowering
pub const stablehlo = @import("stablehlo.zig");

// Re-export pass functions
pub const lower_pass = stablehlo.lower_pass;
pub const LowerPassConfig = stablehlo.LowerPassConfig;
pub const lower_pass_with_config = stablehlo.lower_pass_with_config;
pub const validate_pass = stablehlo.validate_pass;

// Re-export direct lowering API
pub const lower = stablehlo.lower;
pub const lower_program_to_mlir = stablehlo.lower_program_to_mlir;
pub const lower_function_to_mlir = stablehlo.lower_function_to_mlir;
pub const OutputFormat = stablehlo.OutputFormat;
pub const KernelizationLane = stablehlo.KernelizationLane;
