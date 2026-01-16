/// StableHLO Interchange Module (IM)
///
/// Represents a realized IM artifact ready for toolchain consumption.
/// This is the explicit boundary between PR (program representation) and
/// toolchain compilation. The IM is toolchain-facing; PR is Zigrad-internal.
///
/// Current form: MLIR bytecode (StableHLO dialect).
const std = @import("std");

const pr = @import("../../pr/pr.zig");
const lower = @import("lower.zig");

/// Realization options for PR -> IM lowering.
/// These are IM-level options, not toolchain compile options.
pub const RealizeOptions = struct {
    /// Emit MLIR text instead of bytecode (for debugging/inspection).
    emit_text: bool = false,
};

/// StableHLO IM artifact.
pub const IM = struct {
    /// The serialized MLIR (bytecode or text depending on realization options).
    bytecode: []u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *IM) void {
        self.allocator.free(self.bytecode);
    }
};

/// Realize a PR function into a StableHLO IM.
///
/// This is the PR -> IM realization step. The resulting IM can be passed to
/// a toolchain for compilation into an executable artifact (EA).
pub fn realize(allocator: std.mem.Allocator, func: pr.Function, options: RealizeOptions) !IM {
    _ = options; // TODO: emit_text support
    const bytecode = try lower.lowerFunctionToMlirBytecode(allocator, func);
    return IM{ .bytecode = bytecode, .allocator = allocator };
}
