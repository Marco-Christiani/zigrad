const std = @import("std");

pub const pr = @import("pr/mod.zig");
pub const frontend = @import("frontend/frontend.zig");
pub const pipeline = @import("pipeline/mod.zig");

pub const im = struct {
    pub const stablehlo = struct {
        pub const im_mod = @import("im/stablehlo/im.zig");
        pub const lower = @import("im/stablehlo/lower.zig");
        pub const verify = @import("im/stablehlo/verify.zig");

        // IM type and realization (PR -> IM boundary)
        pub const IM = im_mod.IM;
        pub const Encoding = im_mod.Encoding;
        pub const RealizeOptions = im_mod.RealizeOptions;
        pub const realize = im_mod.realize;

        // Re-export other functions for convenience
        pub const lowerFunctionToMlir = lower.lowerFunctionToMlir;
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
