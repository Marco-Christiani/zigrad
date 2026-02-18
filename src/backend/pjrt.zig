/// Unified PJRT Backend
///
/// Merges the Toolchain (compilation) and Runtime (execution) into a single
/// Backend abstraction that owns the PJRT plugin/client/device lifecycle.
///
/// This design reflects PJRT's natural model where compilation is runtime-hosted:
/// the Client does both compile and execute. Separating them into Toolchain/Runtime
/// created artificial boundaries that don't fit JIT compilation well.
///
/// The Backend provides:
/// - Plugin loading and client creation (lifecycle)
/// - Device enumeration and selection
/// - Compilation (MLIR -> LoadedExecutable)
/// - Execution (LoadedExecutable + buffers -> outputs)
/// - Buffer management (host <-> device transfers)
const std = @import("std");

const pr = @import("../pr/pr.zig");
const plugin = @import("../ffi/pjrt/plugin.zig");
const pjrt_api = @import("../ffi/pjrt/api.zig");
const pjrt_types = @import("../ffi/pjrt/types.zig");
// Re-export handle types for callers
pub const LoadedExecutable = pjrt_types.LoadedExecutable;
pub const Buffer = pjrt_types.Buffer;
pub const RawBuffer = pjrt_types.RawBuffer;
pub const Device = pjrt_types.Device;
pub const Event = pjrt_types.Event;
pub const ExecuteResult = pjrt_types.ExecuteResult;

/// Compile options for the PJRT backend.
pub const CompileOptions = struct {
    num_replicas: u32 = 1,
    num_partitions: u32 = 1,
};

/// Unified PJRT Backend.
///
/// Owns the complete PJRT lifecycle: plugin, client, and provides both
/// compilation and execution capabilities.
pub const Backend = struct {
    api: *pjrt_api.Api,
    client: pjrt_types.Client,
    allocator: std.mem.Allocator,

    /// Initialize the backend by loading a PJRT plugin.
    ///
    /// The plugin_path should point to a PJRT plugin DSO (e.g., CPU or GPU plugin).
    pub fn init(allocator: std.mem.Allocator, plugin_path: []const u8) !Backend {
        const api_ptr = try allocator.create(pjrt_api.Api);
        errdefer allocator.destroy(api_ptr);

        api_ptr.* = try plugin.load_plugin(plugin_path);
        errdefer plugin.unload_plugin(api_ptr.*);

        const is_cpu_plugin = std.mem.endsWith(u8, plugin_path, "pjrt_c_api_cpu_plugin.so");
        var client = if (is_cpu_plugin) blk: {
            const env_count = cpu_device_count_from_env();
            if (env_count) |count| break :blk try pjrt_types.Client.create_cpu_with_device_count(api_ptr, count);
            break :blk try pjrt_types.Client.create(api_ptr);
        } else try pjrt_types.Client.create(api_ptr);
        errdefer client.deinit();

        return .{
            .api = api_ptr,
            .client = client,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Backend) void {
        self.client.deinit();
        plugin.unload_plugin(self.api.*);
        self.allocator.destroy(self.api);
    }

    // ========================================================================
    // Device Management
    // ========================================================================

    /// Get all available devices.
    pub fn get_devices(self: *Backend, allocator: std.mem.Allocator) ![]Device {
        return self.client.get_devices(allocator);
    }

    /// Get the underlying client (for advanced use cases).
    pub fn get_client(self: *Backend) *pjrt_types.Client {
        return &self.client;
    }

    // ========================================================================
    // Compilation (MLIR -> EA)
    // ========================================================================

    /// Compile MLIR bytes to a loaded executable.
    ///
    /// This is the core compilation entry point. The MLIR should be in
    /// StableHLO dialect (text or bytecode format).
    pub fn compile(
        self: *Backend,
        device: *const Device,
        mlir_bytes: []const u8,
        is_bytecode: bool,
        options: CompileOptions,
    ) !LoadedExecutable {
        const compile_opts_pb = try build_compile_options_proto(self.allocator, options);
        defer self.allocator.free(compile_opts_pb);

        const format: pjrt_types.ProgramFormat = if (is_bytecode) .mlir_bytecode else .mlir_text;
        return self.client.compile(device, format, mlir_bytes, compile_opts_pb);
    }

    /// Compile and serialize the resulting executable (for caching).
    pub fn compile_serialized(
        self: *Backend,
        device: *const Device,
        mlir_bytes: []const u8,
        is_bytecode: bool,
        options: CompileOptions,
    ) ![]u8 {
        var exe = try self.compile(device, mlir_bytes, is_bytecode, options);
        defer exe.deinit(self.api);
        return exe.serialize(self.api, self.allocator);
    }

    /// Load a previously serialized executable.
    pub fn load_serialized_executable(
        self: *Backend,
        serialized_executable: []const u8,
        overridden_compile_options: ?[]const u8,
    ) !LoadedExecutable {
        return self.client.deserialize_and_load(serialized_executable, overridden_compile_options);
    }

    // ========================================================================
    // Buffer Management
    // ========================================================================

    /// Create a buffer on device from host data.
    pub fn buffer_from_host(
        self: *Backend,
        device: *const Device,
        data: []const u8,
        dtype: pr.DType,
        shape: []const i64,
    ) !Buffer {
        const buf_type = dtype_to_buffer_type(dtype);
        return self.client.buffer_from_host(device, data, buf_type, shape);
    }

    // ========================================================================
    // Execution
    // ========================================================================

    pub fn execute(self: *Backend, exe: *LoadedExecutable, allocator: std.mem.Allocator, inputs: []const Buffer) !ExecuteResult {
        return exe.execute(self.api, allocator, inputs);
    }

    pub fn execute_into(self: *Backend, exe: *LoadedExecutable, input_ptrs: []const RawBuffer, output_ptrs: []?RawBuffer, non_donatable: ?[]const i64) !?Event {
        return exe.execute_into_opts(self.api, input_ptrs, output_ptrs, non_donatable);
    }

    // ========================================================================
    // Handle Lifecycle
    // ========================================================================

    pub fn deinit_buffer(self: *Backend, buf: *Buffer) void {
        buf.deinit(self.api);
    }

    pub fn deinit_event(self: *Backend, ev: *Event) void {
        ev.deinit(self.api);
    }

    pub fn deinit_executable(self: *Backend, exe: *LoadedExecutable) void {
        exe.deinit(self.api);
    }

    // ========================================================================
    // Data Transfer
    // ========================================================================

    pub fn buffer_to_host(self: *Backend, buf: *Buffer, dst: []u8) !Event {
        return buf.to_host(self.api, dst);
    }

    pub fn await_event(self: *Backend, ev: *Event) !void {
        return ev.await_(self.api);
    }

    // ========================================================================
    // Extended (PJRT-specific, not part of AsBackend contract)
    // ========================================================================

    pub fn buffer_unsafe_pointer(self: *Backend, buf: *const Buffer) !usize {
        return buf.unsafe_pointer(self.api);
    }

    pub fn buffer_is_on_cpu(self: *Backend, buf: *const Buffer) !bool {
        return buf.is_on_cpu(self.api);
    }

    pub fn executable_memory_stats(self: *Backend, exe: *LoadedExecutable) !LoadedExecutable.CompiledMemoryStats {
        return exe.get_compiled_memory_stats(self.api);
    }

    pub fn device_memory_stats(self: *Backend, device: *const Device) !Device.MemoryStats {
        return device.get_memory_stats(self.api);
    }
};

fn cpu_device_count_from_env() ?usize {
    const env = std.posix.getenv("ZG_CPU_DEVICE_COUNT") orelse return null;
    const text = std.mem.sliceTo(env, 0);
    if (text.len == 0) return null;
    return std.fmt.parseInt(usize, text, 10) catch null;
}

// ============================================================================
// Helpers
// ============================================================================

fn write_varint(writer: anytype, value: u64) !void {
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

fn dtype_to_buffer_type(dtype: pr.DType) pjrt_types.BufferType {
    return switch (dtype) {
        .bf16 => .bf16,
        .f32 => .f32,
        .f64 => .f64,
        .i32 => .i32,
        .i64 => .i64,
        .u32 => .u32,
        .u64 => .u64,
        .bool => .i32,
    };
}

/// Build a minimal CompileOptionsProto for PJRT (protobuf wire format).
fn build_compile_options_proto(allocator: std.mem.Allocator, options: CompileOptions) ![]u8 {
    var build_opts = try std.ArrayList(u8).initCapacity(allocator, 16);
    defer build_opts.deinit(allocator);
    const b = build_opts.writer(allocator);

    // ExecutableBuildOptionsProto:
    //   int64 num_replicas = 4;
    //   int64 num_partitions = 5;
    try b.writeByte((4 << 3) | 0);
    try write_varint(b, options.num_replicas);
    try b.writeByte((5 << 3) | 0);
    try write_varint(b, options.num_partitions);

    var out = try std.ArrayList(u8).initCapacity(allocator, 32);
    errdefer out.deinit(allocator);
    const w = out.writer(allocator);

    // CompileOptionsProto:
    //   ExecutableBuildOptionsProto executable_build_options = 3;
    try w.writeByte((3 << 3) | 2);
    try write_varint(w, build_opts.items.len);
    try w.writeAll(build_opts.items);

    return out.toOwnedSlice(allocator);
}
