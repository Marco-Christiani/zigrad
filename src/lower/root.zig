/// Lower Module
///
/// Lowering passes that transform PR into target-specific IR.
///
/// Sub-modules:
/// - mlir: MLIR infrastructure + dialect-specific lowering (gated by has_mlir)
/// - types: Pure data types (always available, no link dependency)
const build_options = @import("build_options");

pub const types = @import("types.zig");

/// Pure data types - always available regardless of MLIR enablement.
pub const LowerPassConfig = types.LowerPassConfig;
pub const OutputFormat = types.OutputFormat;

/// MLIR lowering infrastructure (gated by -Dmlir build option).
pub const mlir = if (build_options.has_mlir) @import("mlir/root.zig") else struct {};

// Re-export common StableHLO pass functions when MLIR is enabled.
pub const lower_pass = if (build_options.has_mlir) mlir.stablehlo.lower_pass else {};
pub const lower_pass_with_config = if (build_options.has_mlir) mlir.stablehlo.lower_pass_with_config else {};
pub const validate_pass = if (build_options.has_mlir) mlir.stablehlo.validate_pass else {};
pub const lower = if (build_options.has_mlir) mlir.stablehlo.lower else {};
pub const lower_program_to_mlir = if (build_options.has_mlir) mlir.stablehlo.lower_program_to_mlir else {};
pub const lower_function_to_mlir = if (build_options.has_mlir) mlir.stablehlo.lower_function_to_mlir else {};

test {
    if (build_options.has_mlir) {
        @import("std").testing.refAllDecls(@This());
    }
}
