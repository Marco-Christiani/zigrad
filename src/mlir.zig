//! Public surface for the optional MLIR integration.

pub const Session = @import("mlir/session.zig").Session;
pub const lowering = @import("mlir/lowering.zig");
pub const passes = @import("mlir/passes.zig");
pub const stablehlo = @import("mlir/stablehlo.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
