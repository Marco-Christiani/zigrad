//! PR to StableHLO lowering through the optional MLIR integration.
const lower_mod = @import("stablehlo/lower.zig");

pub const lower = lower_mod.lower;
pub const Lower = lower_mod.Lower;
pub const lower_program_to_mlir = lower_mod.lower_program_to_mlir;
pub const lower_function_to_mlir = lower_mod.lower_function_to_mlir;
pub const LowerConfig = lower_mod.LowerConfig;

test {
    @import("std").testing.refAllDecls(@This());
}
