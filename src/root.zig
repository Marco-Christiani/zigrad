const std = @import("std");

pub const pr = @import("pr/mod.zig");
pub const frontend = @import("frontend/frontend.zig");

pub const im = struct {
    pub const stablehlo = struct {
        pub const lower = @import("im/stablehlo/lower.zig");
        pub const verify = @import("im/stablehlo/verify.zig");

        // Re-export main functions for convenience
        pub const lowerFunctionToMlirBytecode = lower.lowerFunctionToMlirBytecode;
        pub const registerCustomCallTarget = verify.registerCustomCallTarget;
        pub const clearCustomCallTargets = verify.clearCustomCallTargets;
        pub const isCustomCallTargetRegistered = verify.isCustomCallTargetRegistered;
    };
};

pub const runtime = struct {
    pub const HostBuffer = @import("runtime/host_buffer.zig").HostBuffer;
    pub const DType = @import("runtime/host_buffer.zig").DType;
    pub const Shape = @import("runtime/host_buffer.zig").Shape;

    pub const pjrt = @import("runtime/pjrt/runtime.zig");
};

pub const toolchain = struct {
    pub const xla = @import("toolchain/xla/compile.zig");
};

test {
    std.testing.refAllDecls(@This());
}
