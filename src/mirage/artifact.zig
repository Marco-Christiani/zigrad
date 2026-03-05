//! Mirage kernel artifact binary format.
//!
//! Encodes everything the dispatch layer needs to compile and launch
//! a Mirage kernel: filtered CUDA source, workspace size, and per-kernel
//! launch metadata with argument mappings.
//!
//! Layout:
//!   header (16 bytes)
//!   source (source_len bytes, already filtered for NVRTC)
//!   kernel descriptors (variable)

const std = @import("std");
const mirage_c = @import("../c/mirage/c.zig");

const log = std.log.scoped(.@"zg/mirage_artifact");

pub const magic: u32 = 0x4D495241; // "MIRA"
pub const version: u32 = 1;

/// Describes where a kernel argument pointer comes from at dispatch time.
pub const ArgSource = enum(u32) {
    /// input_tensors[index] from the dispatch call.
    input = 0,
    /// output_tensors[index] from the dispatch call.
    output = 1,
    /// (char*)workspace + byte_offset.
    buf = 2,
};

pub const KernelArg = struct {
    source: ArgSource,
    index_or_offset: u64,
};

pub const KernelDesc = struct {
    func_name: []const u8,
    smem_bytes: u32,
    grid_dim: [3]u32,
    block_dim: [3]u32,
    args: []const KernelArg,
};

pub const Artifact = struct {
    source: []const u8,
    buf_size: u32,
    kernels: []const KernelDesc,
};

/// Serialize an artifact to a byte buffer.
pub fn encode(allocator: std.mem.Allocator, artifact: Artifact) ![]const u8 {
    var buf = std.ArrayList(u8){};
    errdefer buf.deinit(allocator);

    const writer = buf.writer(allocator);

    // Header
    try writer.writeInt(u32, magic, .little);
    try writer.writeInt(u32, version, .little);
    try writer.writeInt(u32, @intCast(artifact.source.len), .little);
    try writer.writeInt(u32, artifact.buf_size, .little);

    // Source
    try writer.writeAll(artifact.source);

    // Number of kernels
    try writer.writeInt(u32, @intCast(artifact.kernels.len), .little);

    // Kernel descriptors
    for (artifact.kernels) |k| {
        try writer.writeInt(u32, @intCast(k.func_name.len), .little);
        try writer.writeAll(k.func_name);
        try writer.writeInt(u32, k.smem_bytes, .little);
        for (k.grid_dim) |d| try writer.writeInt(u32, d, .little);
        for (k.block_dim) |d| try writer.writeInt(u32, d, .little);
        try writer.writeInt(u32, @intCast(k.args.len), .little);
        for (k.args) |arg| {
            try writer.writeInt(u32, @intFromEnum(arg.source), .little);
            try writer.writeInt(u64, arg.index_or_offset, .little);
        }
    }

    return try buf.toOwnedSlice(allocator);
}

/// Decode an artifact from a byte buffer. All returned slices point into
/// `data` or are allocated via `allocator`.
pub fn decode(allocator: std.mem.Allocator, data: []const u8) !Artifact {
    if (data.len < 16) return error.InvalidArtifact;

    var pos: usize = 0;

    const m = readU32(data, &pos);
    if (m != magic) return error.InvalidArtifact;
    const v = readU32(data, &pos);
    if (v != version) return error.InvalidArtifact;
    const source_len = readU32(data, &pos);
    const buf_size = readU32(data, &pos);

    if (pos + source_len > data.len) return error.InvalidArtifact;
    const source = data[pos .. pos + source_len];
    pos += source_len;

    const num_kernels = readU32(data, &pos);
    const kernels = try allocator.alloc(KernelDesc, num_kernels);
    errdefer allocator.free(kernels);

    for (kernels) |*k| {
        const name_len = readU32(data, &pos);
        if (pos + name_len > data.len) return error.InvalidArtifact;
        k.func_name = data[pos .. pos + name_len];
        pos += name_len;

        k.smem_bytes = readU32(data, &pos);
        for (&k.grid_dim) |*d| d.* = readU32(data, &pos);
        for (&k.block_dim) |*d| d.* = readU32(data, &pos);

        const num_args = readU32(data, &pos);
        const args = try allocator.alloc(KernelArg, num_args);
        for (args) |*arg| {
            const src_val = readU32(data, &pos);
            arg.source = @enumFromInt(src_val);
            arg.index_or_offset = readU64(data, &pos);
        }
        k.args = args;
    }

    return .{
        .source = source,
        .buf_size = buf_size,
        .kernels = kernels,
    };
}

fn readU32(data: []const u8, pos: *usize) u32 {
    if (pos.* + 4 > data.len) return 0;
    const val = std.mem.readInt(u32, data[pos.*..][0..4], .little);
    pos.* += 4;
    return val;
}

fn readU64(data: []const u8, pos: *usize) u64 {
    if (pos.* + 8 > data.len) return 0;
    const val = std.mem.readInt(u64, data[pos.*..][0..8], .little);
    pos.* += 8;
    return val;
}

test "encode-decode round trip" {
    const allocator = std.testing.allocator;

    const args = [_]KernelArg{
        .{ .source = .input, .index_or_offset = 0 },
        .{ .source = .input, .index_or_offset = 1 },
        .{ .source = .output, .index_or_offset = 0 },
    };
    const kernel_desc = KernelDesc{
        .func_name = "custom_kernel_0",
        .smem_bytes = 49152,
        .grid_dim = .{ 1, 1, 1 },
        .block_dim = .{ 128, 1, 1 },
        .args = &args,
    };
    const kernels = [_]KernelDesc{kernel_desc};

    const artifact = Artifact{
        .source = "__global__ void custom_kernel_0() {}",
        .buf_size = 4096,
        .kernels = &kernels,
    };

    const encoded = try encode(allocator, artifact);
    defer allocator.free(encoded);

    const decoded = try decode(allocator, encoded);
    defer allocator.free(decoded.kernels);
    defer allocator.free(decoded.kernels[0].args);

    try std.testing.expectEqualStrings(artifact.source, decoded.source);
    try std.testing.expectEqual(artifact.buf_size, decoded.buf_size);
    try std.testing.expectEqual(artifact.kernels.len, decoded.kernels.len);
    try std.testing.expectEqualStrings("custom_kernel_0", decoded.kernels[0].func_name);
    try std.testing.expectEqual(@as(u32, 49152), decoded.kernels[0].smem_bytes);
    try std.testing.expectEqual(@as(usize, 3), decoded.kernels[0].args.len);
    try std.testing.expectEqual(ArgSource.input, decoded.kernels[0].args[0].source);
    try std.testing.expectEqual(ArgSource.output, decoded.kernels[0].args[2].source);
}
