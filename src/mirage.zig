pub const provider = @import("mirage/provider.zig");
pub const dispatch = @import("mirage/dispatch.zig");
pub const artifact = @import("mirage/artifact.zig");
pub const mlir = @import("mirage/mlir.zig");

pub const ffi = @import("c/mirage/api.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
