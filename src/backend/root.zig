pub const interface = @import("interface.zig");
pub const pjrt = @import("pjrt.zig");
pub const PjrtBackend = interface.AsBackend(pjrt);

test {
    @setEvalBranchQuota(10000);
    @import("std").testing.refAllDecls(@This());
}
