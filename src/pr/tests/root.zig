
test{
    @import("std").testing.refAllDecls(@This());
    _ = @import("eval.zig");
    _ = @import("grad_check.zig");
}
