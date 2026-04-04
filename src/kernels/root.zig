//! TODO: again, outdated scratch, see benchmark/root.zig, same thing.
pub const gemm = @import("gemm.zig");
pub const gemm_naive = @import("gemm_naive.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
