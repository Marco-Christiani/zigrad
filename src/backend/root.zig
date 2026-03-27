const build_options = @import("build_options");

pub const Backend = @import("Backend.zig");
pub const pjrt = @import("pjrt.zig");

/// IREE backend module.  Only compiled when `-Diree-backend=true`.
pub const iree = if (build_options.has_iree) @import("iree.zig") else struct {};

test {
    @setEvalBranchQuota(10000);
    @import("std").testing.refAllDecls(@This());
}
