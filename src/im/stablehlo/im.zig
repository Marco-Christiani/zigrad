/// StableHLO Interchange Module (IM)
///
/// Represents a realized IM artifact ready for toolchain consumption.
/// This is the explicit boundary between PR (program representation) and
/// toolchain compilation. The IM is toolchain-facing; PR is Zigrad-internal.
///
/// Current form: MLIR (StableHLO dialect) in either text or bytecode form.
const std = @import("std");

const pr = @import("../../pr/pr.zig");
const lower = @import("lower.zig");

/// Realization options for PR -> IM lowering.
/// These are IM-level options, not toolchain compile options.
pub const RealizeOptions = struct {
    /// Emit MLIR text instead of bytecode (for debugging/inspection).
    emit_text: bool = false,
};

pub const Encoding = enum {
    mlir_text,
    mlir_bytecode,
};

/// StableHLO IM artifact.
pub const IM = struct {
    /// The serialized MLIR (text or bytecode depending on realization options).
    bytes: []u8,
    encoding: Encoding,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *IM) void {
        self.allocator.free(self.bytes);
    }
};

/// Realize a PR function into a StableHLO IM.
///
/// This is the PR -> IM realization step. The resulting IM can be passed to
/// a toolchain for compilation into an executable artifact (EA).
pub fn realize(allocator: std.mem.Allocator, func: pr.Function, options: RealizeOptions) !IM {
    if (options.emit_text) {
        const text = try lower.lowerFunctionToMlir(allocator, func, .mlir_text);
        return IM{ .bytes = text, .encoding = .mlir_text, .allocator = allocator };
    }
    const bytecode = try lower.lowerFunctionToMlir(allocator, func, .mlir_bytecode);
    return IM{ .bytes = bytecode, .encoding = .mlir_bytecode, .allocator = allocator };
}
