//! Optional CUDA driver and runtime compilation capabilities.

pub const driver = @import("cuda/driver.zig");
pub const nvrtc = @import("cuda/nvrtc.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
