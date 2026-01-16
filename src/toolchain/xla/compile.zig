/// XLA Toolchain - Compilation
///
/// Compiles StableHLO IM to executable artifacts via PJRT.
/// This module owns the compilation path; runtime owns execution.
///
/// Note: This toolchain compiles via PJRT (JIT). It can optionally serialize the
/// resulting executable for cache-style reuse, but this is not "true XLA AOT".
///
/// The toolchain accepts IM (interchange module), not PR (program representation).
/// Use `zg.im.stablehlo.realize(...)` to produce IM from PR before calling this.
const std = @import("std");

const im_stablehlo = @import("../../im/stablehlo/im.zig");
const pjrt_types = @import("../../ffi/pjrt/types.zig");

pub const IM = im_stablehlo.IM;

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

/// Compile an IM to a loaded executable (JIT).
///
/// Takes a StableHLO IM and compiles via PJRT.
pub fn compile(
    allocator: std.mem.Allocator,
    client: *pjrt_types.Client,
    device: *const pjrt_types.Device,
    im: IM,
    options: CompileOptions,
) !pjrt_types.LoadedExecutable {
    const compile_opts_pb = try buildCompileOptionsProto(allocator, options);
    defer allocator.free(compile_opts_pb);

    // Compile via PJRT
    return client.compile(device, .mlir_bytecode, im.bytecode, compile_opts_pb);
}

/// Compile and serialize the resulting executable (JIT cache path).
pub fn compileSerialized(
    allocator: std.mem.Allocator,
    client: *pjrt_types.Client,
    device: *const pjrt_types.Device,
    im: IM,
    options: CompileOptions,
) ![]u8 {
    var exe = try compile(allocator, client, device, im, options);
    defer exe.deinit();
    return exe.serialize(allocator);
}
