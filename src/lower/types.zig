//! Lowering types shared across the pipeline.
//!
//! Pure data types with no MLIR C link dependency. Importable unconditionally
//! regardless of whether MLIR is enabled.
const pass = @import("../pipeline/pass.zig");

pub const OutputFormat = enum {
    mlir_text,
    mlir_bytecode,
};

/// Configuration for the StableHLO lowering pass.
///
/// Controls output encoding and entry point selection. The lowering pass
/// itself is stateless -- it reads a PR program and produces MLIR bytes.
pub const LowerPassConfig = struct {
    /// Wire format for the MLIR output. Bytecode is smaller and faster to
    /// parse; text is human-readable (useful with `--dump-mlir`).
    encoding: pass.MlirEncoding = .bytecode,

    /// Selects which PR function is the compilation entry point.
    /// The selected function is always renamed to "@main" in MLIR output (XLA requirement).
    entry_name: ?[]const u8 = null,
};
