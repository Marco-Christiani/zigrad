// MLIR C API Bindings Module
// Adapted from ZML (https://github.com/zml/zml)
// Apache License 2.0

// This module provides Zig bindings to the MLIR C API by importing the
// aggregated C header. It is imported as `const c = @import("mlir/c.zig")`
// throughout the MLIR wrapper code.

pub usingnamespace @cImport({
    @cInclude("mlir/c_api/mlir.h");
});
