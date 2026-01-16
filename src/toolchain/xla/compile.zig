/// XLA Toolchain - Compilation
///
/// Compiles PR functions to executable artifacts via StableHLO IM and PJRT.
/// This module owns the compilation path; runtime owns execution.
///
/// Note: This toolchain only supports JIT compilation. The compiled executable
/// is immediately loaded and ready to run.
const std = @import("std");

const pr = @import("../../pr/pr.zig");
const stablehlo = @import("../../im/stablehlo/lower.zig");
const pjrt_types = @import("../../ffi/pjrt/types.zig");

/// Compile options for the XLA toolchain
pub const CompileOptions = struct {
    num_replicas: u32 = 1,
    num_partitions: u32 = 1,
};

/// Compile a PR function to a loaded executable (JIT).
///
/// Takes a PR function, lowers it to StableHLO IM, and compiles via PJRT.
pub fn compile(
    allocator: std.mem.Allocator,
    client: *pjrt_types.Client,
    device: *const pjrt_types.Device,
    func: pr.Function,
    options: CompileOptions,
) !pjrt_types.LoadedExecutable {
    _ = options; // TODO: wire up replicas/partitions to compile options

    // Lower PR to StableHLO IM (bytecode)
    const bytecode = try stablehlo.lowerFunctionToMlirBytecode(allocator, func);
    defer allocator.free(bytecode);

    // Compile via PJRT
    return client.compile(device, .mlir_bytecode, bytecode, null);
}

/// Compile with default options
pub fn compileJit(
    allocator: std.mem.Allocator,
    client: *pjrt_types.Client,
    device: *const pjrt_types.Device,
    func: pr.Function,
) !pjrt_types.LoadedExecutable {
    return compile(allocator, client, device, func, .{});
}
