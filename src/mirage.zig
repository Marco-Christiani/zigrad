//! Optional Mirage kernel-provider integration.

pub const provider = @import("mirage/provider.zig");
pub const dispatch = @import("mirage/dispatch.zig");
pub const config = @import("mirage/config.zig");
pub const artifact = @import("mirage/artifact.zig");
pub const mlir = @import("mirage/mlir.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
