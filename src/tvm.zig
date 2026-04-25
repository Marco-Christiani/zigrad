// High-level modules
pub const module = @import("tvm/module.zig");
pub const provider = @import("tvm/provider.zig");
pub const dispatch = @import("tvm/dispatch.zig");
pub const tune = @import("tvm/tune.zig");

// C binding subsystem re-exports (named to match TVM namespaces)
// TODO: should consider keeping these under a c namespace so reaching in here is explicit, maybe dont export these here at all? Could have a zg.c namespace.
pub const tir = @import("c/tvm/tir.zig");
pub const runtime = @import("c/tvm/runtime.zig");
pub const ffi = @import("c/tvm/api.zig");
pub const meta_schedule = @import("c/tvm/meta_schedule.zig");
pub const compile = @import("c/tvm/compile.zig");
pub const dlpack = @import("c/dlpack.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
