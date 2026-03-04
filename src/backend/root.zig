const build_options = @import("build_options");

pub const interface = @import("interface.zig");
pub const pjrt = @import("pjrt.zig");
pub const PjrtBackend = interface.AsBackend(pjrt);

/// IREE backend module.  Only compiled when `-Diree-backend=true`.
pub const iree = if (build_options.iree_backend) @import("iree.zig") else struct {};

/// Concrete IREE backend type.  `void` when the backend is disabled.
pub const IreeBackend = if (build_options.iree_backend)
    interface.AsBackend(@import("iree.zig"))
else
    void;

test {
    @setEvalBranchQuota(10000);
    @import("std").testing.refAllDecls(@This());
}
