pub const gemm = @import("gemm.zig");
pub const gemm_naive = @import("gemm_naive.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
