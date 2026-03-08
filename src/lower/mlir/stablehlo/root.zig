/// StableHLO Dialect Module
///
/// StableHLO-specific lowering and MLIR passes:
/// - lower: PR -> StableHLO translation
/// - legalize: zigrad.kernel_call -> stablehlo.custom_call
const lower_mod = @import("lower.zig");
const legalize_mod = @import("legalize.zig");

pub const lower = lower_mod.lower;
pub const lower_program_to_mlir = lower_mod.lower_program_to_mlir;
pub const lower_function_to_mlir = lower_mod.lower_function_to_mlir;
pub const lower_pass_with_config = lower_mod.lower_pass_with_config;
pub const lower_pass = lower_mod.lower_pass;
pub const validate_pass = lower_mod.validate_pass;
pub const LowerPassConfig = lower_mod.LowerPassConfig;
pub const OutputFormat = lower_mod.OutputFormat;
pub const KernelizationLane = lower_mod.KernelizationLane;

pub const MlirLegalizePass = legalize_mod.MlirLegalizePass;

test {
    @import("std").testing.refAllDecls(@This());
}
