/// XLA Toolchain - Compilation
///
/// Compiles PR functions to executable artifacts via StableHLO IM and PJRT.
/// This module owns the compilation path; runtime owns execution.
///
/// Note: This toolchain compiles via PJRT (JIT). It can optionally serialize the
/// resulting executable for cache-style reuse, but this is not “true XLA AOT”.
const std = @import("std");

const pr = @import("../../pr/pr.zig");
const stablehlo = @import("../../im/stablehlo/lower.zig");
const pjrt_types = @import("../../ffi/pjrt/types.zig");

/// Compile options for the XLA toolchain
pub const CompileOptions = struct {
    num_replicas: u32 = 1,
    num_partitions: u32 = 1,
};

fn writeVarint(writer: anytype, value: u64) !void {
    var v = value;
    while (true) {
        const byte: u8 = @intCast(v & 0x7F);
        v >>= 7;
        if (v == 0) {
            try writer.writeByte(byte);
            return;
        }
        try writer.writeByte(byte | 0x80);
    }
}

/// Build a minimal CompileOptionsProto for PJRT (protobuf wire format).
///
/// This is PS-level configuration and should not be synthesized in PR/IM.
pub fn buildCompileOptionsProto(allocator: std.mem.Allocator, options: CompileOptions) ![]u8 {
    var build_opts = try std.ArrayList(u8).initCapacity(allocator, 16);
    defer build_opts.deinit(allocator);
    const b = build_opts.writer(allocator);

    // ExecutableBuildOptionsProto:
    //   int64 num_replicas = 4;
    //   int64 num_partitions = 5;
    try b.writeByte((4 << 3) | 0);
    try writeVarint(b, options.num_replicas);
    try b.writeByte((5 << 3) | 0);
    try writeVarint(b, options.num_partitions);

    var out = try std.ArrayList(u8).initCapacity(allocator, 32);
    errdefer out.deinit(allocator);
    const w = out.writer(allocator);

    // CompileOptionsProto:
    //   ExecutableBuildOptionsProto executable_build_options = 3;
    try w.writeByte((3 << 3) | 2);
    try writeVarint(w, build_opts.items.len);
    try w.writeAll(build_opts.items);

    return out.toOwnedSlice(allocator);
}

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
    // Lower PR to StableHLO IM (bytecode)
    const bytecode = try stablehlo.lowerFunctionToMlirBytecode(allocator, func);
    defer allocator.free(bytecode);

    const compile_opts_pb = try buildCompileOptionsProto(allocator, options);
    defer allocator.free(compile_opts_pb);

    // Compile via PJRT
    return client.compile(device, .mlir_bytecode, bytecode, compile_opts_pb);
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

/// Compile and serialize the resulting executable (JIT cache path).
pub fn compileSerialized(
    allocator: std.mem.Allocator,
    client: *pjrt_types.Client,
    device: *const pjrt_types.Device,
    func: pr.Function,
    options: CompileOptions,
) ![]u8 {
    var exe = try compile(allocator, client, device, func, options);
    defer exe.deinit();
    return exe.serialize(allocator);
}

/// Compile+serialize with default options.
pub fn compileSerializedDefault(
    allocator: std.mem.Allocator,
    client: *pjrt_types.Client,
    device: *const pjrt_types.Device,
    func: pr.Function,
) ![]u8 {
    return compileSerialized(allocator, client, device, func, .{});
}
