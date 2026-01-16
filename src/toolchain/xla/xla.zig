const std = @import("std");

const pr = @import("../../pr/pr.zig");
const stablehlo = @import("../../im/stablehlo/lower.zig");
const pjrt_rt = @import("../../runtime/pjrt/runtime.zig");
const pjrt_types = @import("../../bridge/pjrt/types.zig");

pub fn compileMain(
    allocator: std.mem.Allocator,
    runtime: *pjrt_rt.Runtime,
    device: *const pjrt_types.Device,
    func: pr.Function,
) !pjrt_types.LoadedExecutable {
    const bytecode = try stablehlo.lowerFunctionToMlirBytecode(allocator, func);
    defer allocator.free(bytecode);
    return runtime.client.compile(device, .mlir_bytecode, bytecode, null);
}

