//! Read-only analyses over PR programs and functions.

pub const contraction = @import("analysis/contraction.zig");
pub const effects = @import("analysis/effects.zig");
pub const fingerprint = @import("analysis/fingerprint.zig");
pub const region_view = @import("analysis/region_view.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
