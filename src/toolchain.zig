//! External tool capabilities used by compilation integrations.

/// Native object linker capability.
pub const linker = @import("toolchain/linker.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
