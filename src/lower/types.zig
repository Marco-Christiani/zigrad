/// Lowering types shared across the pipeline.
///
/// These are pure data types with no MLIR C link dependency. They can be
/// imported unconditionally regardless of whether MLIR is enabled.
const pass = @import("../pipeline/pass.zig");

pub const OutputFormat = enum {
    mlir_text,
    mlir_bytecode,
};

pub const KernelizationLane = enum {
    pr,
    mlir,
};

pub const LowerPassConfig = struct {
    encoding: pass.MlirEncoding = .bytecode,

    /// Selects which PR function is the compilation entry point.
    /// The selected function is always renamed to "@main" in MLIR output (XLA requirement).
    entry_name: ?[]const u8 = null,

    kernelization_lane: KernelizationLane = .pr,
};
