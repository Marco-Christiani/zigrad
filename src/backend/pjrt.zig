//! Unified PJRT Backend
//!
//! Merges the Toolchain (compilation) and Runtime (execution) into a single
//!  Backend abstraction that owns the PJRT plugin/client/device lifecycle.
//!
//! This design reflects PJRT's natural model where compilation is runtime-hosted:
//!  the Client does both compile and execute. Separating them into Toolchain/Runtime
//!  created artificial boundaries that don't fit JIT compilation well.
//!
//! The Backend provides:
//!  - Plugin loading and client creation (lifecycle)
//!  - Device enumeration and selection
//!  - Compilation (MLIR -> LoadedExecutable)
//!  - Execution (LoadedExecutable + buffers -> outputs)
//!  - Buffer management (host <-> device transfers)
// TODO: pjrt async manager apis, also we likely assumed the wrong async contract
//  throughout. See:
//   1. https://openxla.org/stablehlo/spec#async_start
//   2. https://openxla.org/stablehlo/spec#execution
//   3. https://github.com/openxla/stablehlo/issues/484
//   4. https://openxla.org/stablehlo/spec#optimization_barrier
const std = @import("std");

const pr = @import("../pr/pr.zig");
const kernel = @import("../kernel.zig");
const BackendInterface = @import("Backend.zig");
const plugin = @import("../c/pjrt/plugin.zig");
const pjrt_api = @import("../c/pjrt/api.zig");
const pjrt_types = @import("../c/pjrt/types.zig");
const c = @import("../c/pjrt/c.zig").c;
const log = std.log.scoped(.@"zg/pjrt_backend");

const dispatch_target_name = "zigrad.kernel.dispatch";

const Platform = enum {
    host,
    cuda,
    unknown,
};

var kernel_dispatch_target_registered: bool = false;

const DispatchTypeReg = struct {
    id: i64 = 0,
    registered: bool = false,
};

const dispatch_store_type_name = "zigrad.kernel.store.v1";
const dispatch_dispatch_registry_type_name = "zigrad.kernel.dispatch_registry.v1";
const dispatch_platform_type_name = "zigrad.kernel.platform.v1";

var dispatch_store_type: DispatchTypeReg = .{};
var dispatch_dispatch_registry_type: DispatchTypeReg = .{};
var dispatch_platform_type: DispatchTypeReg = .{};

// Re-export handle types for callers
pub const LoadedExecutable = pjrt_types.LoadedExecutable;
pub const Buffer = pjrt_types.Buffer;
pub const RawBuffer = pjrt_types.RawBuffer;
pub const Device = pjrt_types.Device;
pub const Event = pjrt_types.Event;
pub const ExecuteResult = pjrt_types.ExecuteResult;

pub const CompileOptions = BackendInterface.CompileOptions;
pub const ExecuteOptions = BackendInterface.ExecuteOptions;

/// Unified PJRT Backend.
///
/// Owns the complete PJRT lifecycle: plugin, client, and provides both
/// compilation and execution capabilities.
pub const Backend = struct {
    api: *pjrt_api.Api,
    client: pjrt_types.Client,
    allocator: std.mem.Allocator,
    platform: Platform,
    /// Kernel dispatch platform, derived from `platform`. Stored here so that
    /// its address can be passed as user data in the execute context (must
    /// outlive individual execute calls).
    dispatch_platform: kernel.DispatchPlatform,
    kernel_dispatch_registered: bool,
    interface: BackendInterface,

    /// Initialize the backend by loading a PJRT plugin.
    ///
    /// The plugin_path should point to a PJRT plugin DSO (e.g., CPU or GPU plugin).
    pub fn init(allocator: std.mem.Allocator, plugin_path: []const u8) !Backend {
        const api_ptr = try allocator.create(pjrt_api.Api);
        errdefer allocator.destroy(api_ptr);

        api_ptr.* = try plugin.load_plugin(plugin_path);
        errdefer plugin.unload_plugin(api_ptr.*);

        const platform = detect_platform(plugin_path);
        try register_dispatch_target(api_ptr, platform);

        const is_cpu_plugin = std.mem.endsWith(u8, plugin_path, "pjrt_c_api_cpu_plugin.so");
        var client = if (is_cpu_plugin) blk: {
            const env_count = cpu_device_count_from_env();
            if (env_count) |count| break :blk try pjrt_types.Client.create_cpu_with_device_count(api_ptr, count);
            break :blk try pjrt_types.Client.create(api_ptr);
        } else try pjrt_types.Client.create(api_ptr);
        errdefer client.deinit();

        const self = Backend{
            .api = api_ptr,
            .client = client,
            .allocator = allocator,
            .platform = platform,
            .dispatch_platform = switch (platform) {
                .cuda => .cuda,
                .host, .unknown => .host,
            },
            .kernel_dispatch_registered = false,
            .interface = .{ .vtable = &iface_vtable },
        };
        log.info("plugin loaded: {s}", .{plugin_path});
        log.info("platform: {s}, typed-ffi: {s}", .{
            @tagName(self.platform),
            if (self.api.ffi_extension() != null) "available" else "unavailable",
        });
        return self;
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
        const format_tag: []const u8 = if (is_bytecode) "bytecode" else "text";
        log.info("compile: {d:.1}KB {s}, platform={s}", .{
            @as(f64, @floatFromInt(mlir_bytes.len)) / 1024.0,
            format_tag,
            @tagName(self.platform),
        });

        var timer = std.time.Timer.start() catch null;

        const compile_opts_pb = try build_compile_options_proto(self.allocator, options);
        defer self.allocator.free(compile_opts_pb);

        const format: pjrt_types.ProgramFormat = if (is_bytecode) .mlir_bytecode else .mlir_text;
        const executable = try self.client.compile(device, format, mlir_bytes, compile_opts_pb);

        if (timer) |*t| {
            const elapsed_ns = t.read();
            log.info("compile completed in {d:.2}ms", .{
                @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(std.time.ns_per_ms)),
            });
        }

        return executable;
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

    pub fn execute(self: *Backend, exe: *LoadedExecutable, allocator: std.mem.Allocator, inputs: []const Buffer, options: ExecuteOptions) !ExecuteResult {
        const execute_context = try create_dispatch_execute_context(self, options);
        defer if (execute_context) |ctx| destroy_execute_context(self.api, ctx);
        return exe.execute_with_context(self.api, allocator, inputs, execute_context);
    }

    pub fn execute_into(self: *Backend, exe: *LoadedExecutable, input_ptrs: []const RawBuffer, output_ptrs: []RawBuffer, non_donatable: ?[]const i64, options: ExecuteOptions) !?Event {
        const execute_context = try create_dispatch_execute_context(self, options);
        defer if (execute_context) |ctx| destroy_execute_context(self.api, ctx);
        return exe.execute_into_opts_with_context(self.api, input_ptrs, output_ptrs, non_donatable, execute_context);
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

    pub fn has_typed_ffi(self: *Backend) bool {
        return self.api.ffi_extension() != null;
    }

    pub fn device_kind(self: *Backend, device: *const Device) ![]const u8 {
        return device.get_kind(self.api);
    }

    pub fn is_cuda(self: *Backend) bool {
        return self.platform == .cuda;
    }

    /// Enforce typed-FFI availability for kernelized custom_call execution.
    pub fn require_typed_ffi(self: *Backend) !void {
        if (!self.has_typed_ffi()) return error.TypedFfiUnavailable;
    }

    /// Register the temporary single-target typed-FFI dispatcher.
    ///
    /// Dispatch entries are resolved at execute time via `ExecuteOptions`.
    pub fn register_kernel_dispatcher(self: *Backend) !void {
        try self.require_typed_ffi();
        if (self.kernel_dispatch_registered) return;

        if (kernel_dispatch_target_registered) {
            self.kernel_dispatch_registered = true;
            return;
        }

        const ffi_ext = self.api.ffi_extension() orelse return error.TypedFfiUnavailable;

        if (!try register_ffi_for_platform(self.api, ffi_ext, self.platform))
            return error.TypedFfiRegistrationFailed;

        self.kernel_dispatch_registered = true;
    }

    pub fn device_memory_stats(self: *Backend, device: *const Device) !Device.MemoryStats {
        return device.get_memory_stats(self.api);
    }

    // ========================================================================
    // Type-erased interface (vtable impl)
    // ========================================================================

    const iface_vtable = BackendInterface.VTable{
        .compile = iface_compile,
        .buffer_from_host = iface_buffer_from_host,
        .buffer_to_host = iface_buffer_to_host,
        .execute = iface_execute,
        .execute_into = iface_execute_into,
        .await_event = iface_await_event,
        .serialize_executable = iface_serialize_executable,
        .load_serialized = iface_load_serialized,
        .deinit_buffer = iface_deinit_buffer,
        .deinit_event = iface_deinit_event,
        .deinit_executable = iface_deinit_executable,
        .get_devices = iface_get_devices,
    };

    fn promote(iface: *BackendInterface) *Backend {
        return @fieldParentPtr("interface", iface);
    }

    fn iface_compile(iface: *BackendInterface, device: BackendInterface.Device, mlir: []const u8, is_bytecode: bool, opts: BackendInterface.CompileOptions) BackendInterface.Error!BackendInterface.Executable {
        const self = promote(iface);
        const pjrt_device = unwrap_device(device);
        const exe = self.compile(&pjrt_device, mlir, is_bytecode, opts) catch return error.BackendError;
        return wrap_executable(self, exe);
    }

    fn iface_buffer_from_host(iface: *BackendInterface, device: BackendInterface.Device, data: []const u8, dtype: pr.DType, shape: []const i64) BackendInterface.Error!BackendInterface.Buffer {
        const self = promote(iface);
        const pjrt_device = unwrap_device(device);
        const buf = self.buffer_from_host(&pjrt_device, data, dtype, shape) catch return error.BackendError;
        return wrap_buffer(buf);
    }

    fn iface_buffer_to_host(iface: *BackendInterface, buf: BackendInterface.Buffer, dst: []u8) BackendInterface.Error!?BackendInterface.Event {
        const self = promote(iface);
        var pjrt_buf = unwrap_buffer(buf);

        // Fast path: host-resident buffer so direct memcpy is potentially viable, skipping PJRT API.
        // No-op when dst already points to the buffer's own memory.
        if (pjrt_buf.is_on_cpu(self.api) catch false) {
            const src_addr = pjrt_buf.unsafe_pointer(self.api) catch {
                // Fall through to normal PJRT path.
                const ev = self.buffer_to_host(&pjrt_buf, dst) catch return error.BackendError;
                return wrap_event(ev);
            };
            const src: [*]const u8 = @ptrFromInt(src_addr);
            if (src != dst.ptr) {
                @memcpy(dst, src[0..dst.len]);
            }
            return null;
        }

        const ev = self.buffer_to_host(&pjrt_buf, dst) catch return error.BackendError;
        return wrap_event(ev);
    }

    fn iface_execute(iface: *BackendInterface, exe: BackendInterface.Executable, allocator: std.mem.Allocator, inputs: []const BackendInterface.Buffer, opts: BackendInterface.ExecuteOptions) BackendInterface.Error!BackendInterface.ExecuteResult {
        const self = promote(iface);
        const pjrt_exe = unwrap_executable_ptr(exe);
        const pjrt_inputs = @as([*]const Buffer, @ptrCast(inputs.ptr))[0..inputs.len];
        const result = self.execute(pjrt_exe, allocator, pjrt_inputs, opts) catch return error.BackendError;
        const wrapped_outputs = @as([*]BackendInterface.Buffer, @ptrCast(result.outputs.ptr))[0..result.outputs.len];
        return .{
            .outputs = wrapped_outputs,
            .event = if (result.device_complete_event) |ev| wrap_event(ev) else null,
        };
    }

    /// Vtable shim for `execute_into`.
    fn iface_execute_into(iface: *BackendInterface, exe: BackendInterface.Executable, inputs: []const BackendInterface.Buffer, outputs: []BackendInterface.Buffer, non_donatable: ?[]const i64, opts: BackendInterface.ExecuteOptions) BackendInterface.Error!?BackendInterface.Event {
        // Both `BackendInterface.Buffer` (struct wrapping `*anyopaque`) and `RawBuffer` (`*c.PJRT_Buffer`)
        //  are single-pointer types with identical layout - slices can be reinterpreted directly via `@ptrCast`.
        const self = promote(iface);
        const pjrt_exe = unwrap_executable_ptr(exe);
        const pjrt_inputs = @as([*]const RawBuffer, @ptrCast(inputs.ptr))[0..inputs.len];
        const pjrt_outputs = @as([*]RawBuffer, @ptrCast(outputs.ptr))[0..outputs.len];
        const ev = self.execute_into(pjrt_exe, pjrt_inputs, pjrt_outputs, non_donatable, opts) catch return error.BackendError;
        return if (ev) |e| wrap_event(e) else null;
    }

    fn iface_await_event(iface: *BackendInterface, ev: BackendInterface.Event) BackendInterface.Error!void {
        const self = promote(iface);
        var pjrt_ev = unwrap_event(ev);
        self.await_event(&pjrt_ev) catch return error.BackendError;
    }

    fn iface_deinit_buffer(iface: *BackendInterface, buf: BackendInterface.Buffer) void {
        const self = promote(iface);
        var pjrt_buf = unwrap_buffer(buf);
        self.deinit_buffer(&pjrt_buf);
    }

    fn iface_deinit_event(iface: *BackendInterface, ev: BackendInterface.Event) void {
        const self = promote(iface);
        var pjrt_ev = unwrap_event(ev);
        self.deinit_event(&pjrt_ev);
    }

    fn iface_deinit_executable(iface: *BackendInterface, exe: BackendInterface.Executable) void {
        const self = promote(iface);
        const pjrt_exe = unwrap_executable_ptr(exe);
        pjrt_exe.deinit(self.api);
        self.allocator.destroy(pjrt_exe);
    }

    fn iface_serialize_executable(iface: *BackendInterface, exe: BackendInterface.Executable, allocator: std.mem.Allocator) BackendInterface.Error![]u8 {
        const self = promote(iface);
        const pjrt_exe = unwrap_executable_ptr(exe);
        return pjrt_exe.serialize(self.api, allocator) catch return error.BackendError;
    }

    fn iface_load_serialized(iface: *BackendInterface, data: []const u8) BackendInterface.Error!BackendInterface.Executable {
        const self = promote(iface);
        const exe = self.client.deserialize_and_load(data, null) catch return error.BackendError;
        return self.wrap_executable(exe);
    }

    fn iface_get_devices(iface: *BackendInterface, allocator: std.mem.Allocator) BackendInterface.Error![]BackendInterface.Device {
        const self = promote(iface);
        const pjrt_devices = self.get_devices(allocator) catch return error.BackendError;
        return @as([*]BackendInterface.Device, @ptrCast(pjrt_devices.ptr))[0..pjrt_devices.len];
    }

    // Handle wrapping/unwrapping helpers

    fn wrap_buffer(buf: Buffer) BackendInterface.Buffer {
        return .{ .handle = @ptrCast(buf.pjrt_buffer) };
    }

    fn unwrap_buffer(buf: BackendInterface.Buffer) Buffer {
        return .{ .pjrt_buffer = @ptrCast(@alignCast(buf.handle)) };
    }

    fn wrap_event(ev: Event) BackendInterface.Event {
        return .{ .handle = @ptrCast(ev.pjrt_event) };
    }

    fn unwrap_event(ev: BackendInterface.Event) Event {
        return .{ .pjrt_event = @ptrCast(@alignCast(ev.handle)) };
    }

    fn wrap_executable(self_backend: *Backend, exe: LoadedExecutable) BackendInterface.Error!BackendInterface.Executable {
        const heap = self_backend.allocator.create(LoadedExecutable) catch return error.OutOfMemory;
        heap.* = exe;
        return .{ .handle = @ptrCast(heap) };
    }

    fn unwrap_executable_ptr(exe: BackendInterface.Executable) *LoadedExecutable {
        return @ptrCast(@alignCast(exe.handle));
    }

    fn unwrap_device(device: BackendInterface.Device) Device {
        return .{ .pjrt_device = @ptrCast(@alignCast(device.handle)) };
    }
};

fn register_dispatch_target(api: *pjrt_api.Api, platform: Platform) !void {
    if (kernel_dispatch_target_registered) return;
    var registered_any = false;

    if (api.ffi_extension()) |ffi_ext| {
        const any_platform = try register_dispatch_handler_for_platform(api, ffi_ext, null);
        registered_any = registered_any or any_platform;
        registered_any = try register_ffi_for_platform(api, ffi_ext, platform) or registered_any;
    }

    if (api.gpu_custom_call_extension()) |gpu_ext| {
        const gpu_registered = try register_dispatch_handler_via_gpu_extension(api, gpu_ext);
        registered_any = registered_any or gpu_registered;
    }

    kernel_dispatch_target_registered = registered_any;
}

/// Register the dispatch handler for all platform name variants matching `platform`.
/// For `.unknown`, tries all known platforms.
fn register_ffi_for_platform(api: *pjrt_api.Api, ffi_ext: *c.PJRT_FFI, platform: Platform) !bool {
    const platform_names: []const struct { []const u8, []const u8 } = switch (platform) {
        .host => &.{.{ "Host", "host" }},
        .cuda => &.{.{ "CUDA", "cuda" }},
        .unknown => &.{ .{ "Host", "host" }, .{ "CUDA", "cuda" } },
    };
    var registered_any = false;
    for (platform_names) |names| {
        const a = try register_dispatch_handler_for_platform(api, ffi_ext, names[0]);
        const b = try register_dispatch_handler_for_platform(api, ffi_ext, names[1]);
        registered_any = registered_any or a or b;
    }
    return registered_any;
}

fn register_dispatch_handler_via_gpu_extension(api: *pjrt_api.Api, gpu_ext: *c.PJRT_Gpu_Custom_Call) !bool {
    const register = gpu_ext.custom_call orelse return false;

    var args: c.PJRT_Gpu_Register_Custom_Call_Args = std.mem.zeroes(c.PJRT_Gpu_Register_Custom_Call_Args);
    args.struct_size = pjrt_api.pjrt_struct_size(c.PJRT_Gpu_Register_Custom_Call_Args);
    args.function_name = dispatch_target_name.ptr;
    args.function_name_size = dispatch_target_name.len;
    args.api_version = 1;
    args.handler_instantiate = null;
    args.handler_prepare = null;
    args.handler_initialize = null;
    args.handler_execute = @ptrCast(@constCast(&kernel_dispatch_handler));

    const pjrt_err = register(&args);
    if (pjrt_err == null) {
        log.info("registered dispatcher via gpu custom-call extension target='{s}'", .{dispatch_target_name});
        return true;
    }

    var err = pjrt_api.PjrtError.from_handle(api, pjrt_err.?);
    const msg = err.get_message(std.heap.page_allocator) catch "<failed to read message>";
    defer if (msg.ptr != "<failed to read message>".ptr) std.heap.page_allocator.free(msg);
    const code = err.get_code() catch {
        err.deinit();
        return false;
    };
    err.deinit();

    log.debug(
        "gpu custom-call register failed target='{s}' code={d} msg={s}",
        .{ dispatch_target_name, code, msg },
    );

    return code == 6;
}

fn register_dispatch_handler_for_platform(api: *pjrt_api.Api, ffi_ext: *c.PJRT_FFI, platform_name: ?[]const u8) !bool {
    var args: c.PJRT_FFI_Register_Handler_Args = std.mem.zeroes(c.PJRT_FFI_Register_Handler_Args);
    args.struct_size = pjrt_api.pjrt_struct_size(c.PJRT_FFI_Register_Handler_Args);
    args.target_name = dispatch_target_name.ptr;
    args.target_name_size = dispatch_target_name.len;
    args.handler = @ptrCast(@constCast(&kernel_dispatch_handler));
    if (platform_name) |name| {
        args.platform_name = name.ptr;
        args.platform_name_size = name.len;
    } else {
        args.platform_name = null;
        args.platform_name_size = 0;
    }
    args.traits = 0;

    const register = ffi_ext.register_handler orelse return false;
    const pjrt_err = register(&args);
    if (pjrt_err == null) {
        log.info("registered typed-ffi dispatcher target='{s}' platform='{s}'", .{ dispatch_target_name, platform_name orelse "<any>" });
        return true;
    }

    var err = pjrt_api.PjrtError.from_handle(api, pjrt_err.?);
    const msg = err.get_message(std.heap.page_allocator) catch "<failed to read message>";
    defer if (msg.ptr != "<failed to read message>".ptr) std.heap.page_allocator.free(msg);
    const code = err.get_code() catch {
        err.deinit();
        return false;
    };
    err.deinit();

    log.warn(
        "typed-ffi register_dispatch_handler_for_platform failed target='{s}' platform='{s}' code={d} msg={s}",
        .{ dispatch_target_name, platform_name orelse "<any>", code, msg },
    );

    if (code == 6) return true;
    return false;
}

fn pjrt_error_to_zig(api: *pjrt_api.Api, pjrt_err: *c.PJRT_Error) !void {
    const err = pjrt_api.PjrtError.from_handle(api, pjrt_err);
    return err.to_zig_error();
}

fn ensure_dispatch_type_id(
    api: *pjrt_api.Api,
    ffi_ext: *c.PJRT_FFI,
    type_name: []const u8,
    reg: *DispatchTypeReg,
) !i64 {
    if (reg.registered) return reg.id;

    const type_register = ffi_ext.type_register orelse return error.TypedFfiUnavailable;
    var type_info: c.PJRT_FFI_Type_Info = .{
        .deleter = null,
        .serialize = null,
        .deserialize = null,
    };

    var args: c.PJRT_FFI_Type_Register_Args = std.mem.zeroes(c.PJRT_FFI_Type_Register_Args);
    args.struct_size = pjrt_api.pjrt_struct_size(c.PJRT_FFI_Type_Register_Args);
    args.extension_start = null;
    args.type_name = type_name.ptr;
    args.type_name_size = type_name.len;
    args.type_id = 0;
    args.type_info = &type_info;

    if (type_register(&args)) |pjrt_err| {
        try pjrt_error_to_zig(api, pjrt_err);
    }

    reg.id = args.type_id;
    reg.registered = true;
    return reg.id;
}

fn add_dispatch_user_data(
    api: *pjrt_api.Api,
    ffi_ext: *c.PJRT_FFI,
    context: *c.PJRT_ExecuteContext,
    type_id: i64,
    data_ptr: *const anyopaque,
) !void {
    const add_user_data = ffi_ext.user_data_add orelse return error.TypedFfiUnavailable;

    var add_args: c.PJRT_FFI_UserData_Add_Args = std.mem.zeroes(c.PJRT_FFI_UserData_Add_Args);
    add_args.struct_size = pjrt_api.pjrt_struct_size(c.PJRT_FFI_UserData_Add_Args);
    add_args.extension_start = null;
    add_args.context = context;
    add_args.user_data = .{
        .type_id = type_id,
        // HACK: errr, safe?..
        .data = @ptrCast(@constCast(data_ptr)),
    };

    if (add_user_data(&add_args)) |pjrt_err| {
        try pjrt_error_to_zig(api, pjrt_err);
    }
}

/// Build a PJRT execute context carrying kernel dispatch user data.
///
/// Returns `null` when no kernel dispatch is configured (both `store` and
///  `dispatch_registry` are null), signaling callers to skip the context.
/// Otherwise wires the store, registry, and platform into typed FFI user-data
///  slots on the context.
fn create_dispatch_execute_context(self: *Backend, options: ExecuteOptions) !?*c.PJRT_ExecuteContext {
    if (options.store == null and options.dispatch_registry == null) return null;

    const ffi_ext = self.api.ffi_extension() orelse return error.TypedFfiUnavailable;

    var create_args = pjrt_api.init_args(c.PJRT_ExecuteContext_Create_Args);
    create_args.context = null;
    try self.api.call("PJRT_ExecuteContext_Create", &create_args);

    const context = create_args.context orelse return error.PjrtReturnedNullExecuteContext;
    errdefer destroy_execute_context(self.api, context);

    if (options.store) |store| {
        const type_id = try ensure_dispatch_type_id(self.api, ffi_ext, dispatch_store_type_name, &dispatch_store_type);
        try add_dispatch_user_data(self.api, ffi_ext, context, type_id, @ptrCast(store));
    }

    if (options.dispatch_registry) |dreg| {
        const type_id = try ensure_dispatch_type_id(self.api, ffi_ext, dispatch_dispatch_registry_type_name, &dispatch_dispatch_registry_type);
        try add_dispatch_user_data(self.api, ffi_ext, context, type_id, @ptrCast(dreg));
    }

    // Thread platform into execute context so the dispatch handler doesn't rely on a process
    //  global for multi-backend coexistence.
    // Uses the backend's stored dispatch_platform (stable address, outlives context).
    {
        const type_id = try ensure_dispatch_type_id(self.api, ffi_ext, dispatch_platform_type_name, &dispatch_platform_type);
        try add_dispatch_user_data(self.api, ffi_ext, context, type_id, @ptrCast(&self.dispatch_platform));
    }

    return context;
}

fn destroy_execute_context(api: *pjrt_api.Api, context: *c.PJRT_ExecuteContext) void {
    var destroy_args = pjrt_api.init_args(c.PJRT_ExecuteContext_Destroy_Args);
    destroy_args.context = context;
    api.call("PJRT_ExecuteContext_Destroy", &destroy_args) catch {};
}

fn lookup_dispatch_user_data_from_context(frame: *c.XLA_FFI_CallFrame, type_id_value: i64) ?*const anyopaque {
    if (type_id_value == 0) return null;

    const ffi_api = frame.api orelse return null;
    const get_context_data = ffi_api.*.XLA_FFI_ExecutionContext_Get orelse return null;

    var type_id = c.XLA_FFI_TypeId{ .type_id = type_id_value };
    var args: c.XLA_FFI_ExecutionContext_Get_Args = std.mem.zeroes(c.XLA_FFI_ExecutionContext_Get_Args);
    args.struct_size = @sizeOf(c.XLA_FFI_ExecutionContext_Get_Args);
    args.extension_start = null;
    args.ctx = frame.ctx;
    args.type_id = &type_id;
    args.data = null;

    if (get_context_data(&args) != null) return null;

    const data_ptr = args.data orelse return null;
    return @ptrCast(@alignCast(data_ptr));
}

fn lookup_store_from_context(frame: *c.XLA_FFI_CallFrame) ?*const kernel.KernelStore {
    if (!dispatch_store_type.registered) return null;
    const data_ptr = lookup_dispatch_user_data_from_context(frame, dispatch_store_type.id) orelse return null;
    return @ptrCast(@alignCast(data_ptr));
}

fn lookup_dispatch_registry_new_from_context(frame: *c.XLA_FFI_CallFrame) ?*const kernel.DispatchRegistry {
    if (!dispatch_dispatch_registry_type.registered) return null;
    const data_ptr = lookup_dispatch_user_data_from_context(frame, dispatch_dispatch_registry_type.id) orelse return null;
    return @ptrCast(@alignCast(data_ptr));
}

fn lookup_platform_from_context(frame: *c.XLA_FFI_CallFrame) kernel.DispatchPlatform {
    if (!dispatch_platform_type.registered) return .host;
    const data_ptr = lookup_dispatch_user_data_from_context(frame, dispatch_platform_type.id) orelse return .host;
    const platform: *const kernel.DispatchPlatform = @ptrCast(@alignCast(data_ptr));
    return platform.*;
}

fn detect_platform(plugin_path: []const u8) Platform {
    if (std.mem.endsWith(u8, plugin_path, "pjrt_c_api_cpu_plugin.so")) return .host;
    if (std.mem.endsWith(u8, plugin_path, "pjrt_c_api_gpu_plugin.so")) return .cuda;
    if (std.mem.indexOf(u8, plugin_path, "cpu") != null) return .host;
    if (std.mem.indexOf(u8, plugin_path, "cuda") != null or std.mem.indexOf(u8, plugin_path, "gpu") != null) return .cuda;
    return .unknown;
}

/// Generic kernel dispatch handler invoked by the XLA FFI framework.
///
/// Extracts kernel_key from custom_call attributes, looks up the artifact
///  in the registry, builds a provider-agnostic DispatchContext from the
///  FFI frame, and delegates to `artifact.dispatch()`.
fn kernel_dispatch_handler(frame: *c.XLA_FFI_CallFrame) callconv(.c) ?*c.XLA_FFI_Error {
    if (handle_metadata_registration_hook(frame)) return null;

    if (frame.stage != c.XLA_FFI_ExecutionStage_EXECUTE) return null;

    const maybe_kernel_key = lookup_dispatch_attr(frame.attrs, "zigrad.kernel_key");
    const kernel_key = maybe_kernel_key orelse {
        return make_ffi_error(frame, "zigrad kernel dispatch: missing zigrad.kernel_key attribute", c.XLA_FFI_Error_Code_INVALID_ARGUMENT);
    };

    const store = lookup_store_from_context(frame) orelse {
        return make_ffi_error(frame, "zigrad kernel dispatch: no kernel store in execute context", c.XLA_FFI_Error_Code_FAILED_PRECONDITION);
    };

    const decision = store.get(kernel_key) orelse {
        return make_ffi_error(frame, "zigrad kernel dispatch: kernel key not found in store", c.XLA_FFI_Error_Code_NOT_FOUND);
    };

    switch (decision) {
        .profitable => |art| {
            const dreg = lookup_dispatch_registry_new_from_context(frame);
            return dispatch_from_store(frame, art, dreg, kernel_key);
        },
        .negative => {
            return make_ffi_error(frame, "zigrad kernel dispatch: store has negative decision for key", c.XLA_FFI_Error_Code_NOT_FOUND);
        },
    }
}

/// Dispatch from store-based path: resolve provider dispatch function from DispatchRegistry.
fn dispatch_from_store(
    frame: *c.XLA_FFI_CallFrame,
    art: kernel.StoredArtifact,
    dreg: ?*const kernel.DispatchRegistry,
    kernel_key: []const u8,
) ?*c.XLA_FFI_Error {
    const dispatch_entry = if (dreg) |reg| reg.get(art.provider_name) else null;
    if (dispatch_entry == null) {
        log.err("store dispatch: no dispatch entry for provider '{s}'", .{art.provider_name});
        return make_ffi_error(frame, "zigrad kernel dispatch: provider not in dispatch registry", c.XLA_FFI_Error_Code_FAILED_PRECONDITION);
    }
    const entry = dispatch_entry.?;

    return execute_dispatch(frame, kernel_key, art.workspace_bytes, entry.dispatch_fn, entry.dispatch_ctx, art.data);
}

/// Common dispatch execution: extract buffers, allocate workspace, call dispatch function.
fn execute_dispatch(
    frame: *c.XLA_FFI_CallFrame,
    dispatch_key: []const u8,
    workspace_bytes: usize,
    dispatch_fn: ?kernel.DispatchFn,
    dispatch_ctx: ?*anyopaque,
    artifact_data: []const u8,
) ?*c.XLA_FFI_Error {
    const dfn = dispatch_fn orelse {
        return make_ffi_error(frame, "zigrad kernel dispatch: no dispatch function", c.XLA_FFI_Error_Code_FAILED_PRECONDITION);
    };
    const dctx = dispatch_ctx orelse {
        return make_ffi_error(frame, "zigrad kernel dispatch: no dispatch context", c.XLA_FFI_Error_Code_FAILED_PRECONDITION);
    };

    // Extract all input and output buffers from the FFI frame.
    var input_descs: [16]kernel.BufferDesc = undefined;
    const num_inputs = extract_buffers(frame.args.size, frame.args.types, frame.args.args, &input_descs) orelse {
        return make_ffi_error(frame, "zigrad kernel dispatch: failed to extract input buffers", c.XLA_FFI_Error_Code_INVALID_ARGUMENT);
    };

    var output_descs: [16]kernel.BufferDesc = undefined;
    const num_outputs = extract_ret_buffers(frame.rets.size, frame.rets.types, frame.rets.rets, &output_descs) orelse {
        return make_ffi_error(frame, "zigrad kernel dispatch: failed to extract output buffers", c.XLA_FFI_Error_Code_INVALID_ARGUMENT);
    };

    const platform = lookup_platform_from_context(frame);

    // Allocate workspace from XLA's BFC pool when the kernel requires it.
    const workspace_alignment: usize = 128; // matches Mirage MemoryPlanner alignment
    var workspace_ptr: ?*anyopaque = null;
    if (workspace_bytes > 0) {
        workspace_ptr = alloc_device_memory(frame, workspace_bytes, workspace_alignment);
        if (workspace_ptr == null) {
            log.err("workspace allocation failed: {d} bytes for '{s}'", .{ workspace_bytes, dispatch_key });
            return make_ffi_error(frame, "zigrad kernel dispatch: workspace allocation failed", c.XLA_FFI_Error_Code_RESOURCE_EXHAUSTED);
        }
        log.debug("allocated {d} bytes workspace for '{s}'", .{ workspace_bytes, dispatch_key });
    }
    defer if (workspace_ptr) |ptr| {
        free_device_memory(frame, ptr, workspace_bytes);
        log.debug("freed {d} bytes workspace for '{s}'", .{ workspace_bytes, dispatch_key });
    };

    const ctx = kernel.DispatchContext{
        .inputs = input_descs[0..num_inputs],
        .outputs = output_descs[0..num_outputs],
        .device_ordinal = get_device_ordinal(frame),
        .platform = platform,
        .stream = get_stream(frame),
        .workspace = workspace_ptr,
        .workspace_bytes_required = workspace_bytes,
        .allocator = std.heap.c_allocator,
    };

    dfn(dctx, artifact_data, dispatch_key, ctx) catch |err| {
        log.err("kernel dispatch failed for '{s}': {s}", .{ dispatch_key, @errorName(err) });
        return dispatch_error_to_ffi(frame, err);
    };
    return null;
}

fn dispatch_error_to_ffi(frame: *c.XLA_FFI_CallFrame, err: kernel.DispatchError) ?*c.XLA_FFI_Error {
    return make_ffi_error(frame, "zigrad kernel dispatch failed", dispatch_error_code(err));
}

fn dispatch_error_code(err: kernel.DispatchError) c.XLA_FFI_Error_Code {
    return switch (err) {
        error.UnsupportedDType,
        error.ShapeMismatch,
        => c.XLA_FFI_Error_Code_INVALID_ARGUMENT,

        error.ProviderLoadFailed,
        error.WorkspaceUnavailable,
        => c.XLA_FFI_Error_Code_FAILED_PRECONDITION,

        error.DispatchFailed,
        error.OutOfMemory,
        => c.XLA_FFI_Error_Code_INTERNAL,
    };
}

/// Extract input buffers from an FFI args structure into BufferDesc array.
fn extract_buffers(
    size: i64,
    types: [*]const c.XLA_FFI_ArgType,
    args_ptr: [*]const ?*anyopaque,
    out: *[16]kernel.BufferDesc,
) ?usize {
    if (size < 0) return null;
    const count: usize = @intCast(size);
    if (count > 16) return null;
    for (0..count) |i| {
        if (types[i] != c.XLA_FFI_ArgType_BUFFER) return null;
        const ptr = args_ptr[i] orelse return null;
        const buf: *c.XLA_FFI_Buffer = @ptrCast(@alignCast(ptr));
        out[i] = ffi_buffer_to_desc(buf) orelse return null;
    }
    return count;
}

/// Extract output buffers from an FFI rets structure into BufferDesc array.
fn extract_ret_buffers(
    size: i64,
    types: [*]const c.XLA_FFI_RetType,
    rets_ptr: [*]const ?*anyopaque,
    out: *[16]kernel.BufferDesc,
) ?usize {
    if (size < 0) return null;
    const count: usize = @intCast(size);
    if (count > 16) return null;
    for (0..count) |i| {
        if (types[i] != c.XLA_FFI_RetType_BUFFER) return null;
        const ptr = rets_ptr[i] orelse return null;
        const buf: *c.XLA_FFI_Buffer = @ptrCast(@alignCast(ptr));
        out[i] = ffi_buffer_to_desc(buf) orelse return null;
    }
    return count;
}

/// Convert an XLA FFI buffer to a provider-agnostic BufferDesc.
fn ffi_buffer_to_desc(buf: *c.XLA_FFI_Buffer) ?kernel.BufferDesc {
    const dtype = ffi_dtype_to_kernel_dtype(buf.dtype) orelse return null;
    const rank: usize = @intCast(buf.rank);
    return .{
        .data = @ptrCast(buf.data orelse return null),
        .dtype = dtype,
        .dims = buf.dims[0..rank],
        .rank = rank,
    };
}

/// Map XLA FFI element types to kernel.DType.
fn ffi_dtype_to_kernel_dtype(xla_dtype: c.XLA_FFI_DataType) ?kernel.DType {
    if (xla_dtype == c.XLA_FFI_DataType_F16) return .f16;
    if (xla_dtype == c.XLA_FFI_DataType_BF16) return .bf16;
    if (xla_dtype == c.XLA_FFI_DataType_F32) return .f32;
    if (xla_dtype == c.XLA_FFI_DataType_F64) return .f64;
    if (xla_dtype == c.XLA_FFI_DataType_S8) return .i8;
    if (xla_dtype == c.XLA_FFI_DataType_U8) return .u8;
    if (xla_dtype == c.XLA_FFI_DataType_S32) return .i32;
    if (xla_dtype == c.XLA_FFI_DataType_S64) return .i64;
    if (xla_dtype == c.XLA_FFI_DataType_U32) return .u32;
    if (xla_dtype == c.XLA_FFI_DataType_U64) return .u64;
    return null;
}

/// Extract the GPU stream handle from an FFI call frame.
fn get_stream(frame: *c.XLA_FFI_CallFrame) ?*anyopaque {
    const ffi_api = frame.api orelse return null;
    const get_stream_fn = ffi_api.*.XLA_FFI_Stream_Get orelse return null;
    var args: c.XLA_FFI_Stream_Get_Args = std.mem.zeroes(c.XLA_FFI_Stream_Get_Args);
    args.struct_size = @sizeOf(c.XLA_FFI_Stream_Get_Args);
    args.ctx = frame.ctx;
    const ffi_err = get_stream_fn(&args);
    if (ffi_err != null) return null;
    return args.stream;
}

fn get_device_ordinal(frame: *c.XLA_FFI_CallFrame) i32 {
    const ffi_api = frame.api orelse return 0;
    const get_ordinal = ffi_api.*.XLA_FFI_DeviceOrdinal_Get orelse return 0;
    var args: c.XLA_FFI_DeviceOrdinal_Get_Args = std.mem.zeroes(c.XLA_FFI_DeviceOrdinal_Get_Args);
    args.struct_size = @sizeOf(c.XLA_FFI_DeviceOrdinal_Get_Args);
    args.ctx = frame.ctx;
    const err = get_ordinal(&args);
    if (err != null) return 0;
    return args.device_ordinal;
}

/// Allocate device memory from XLA's BFC pool via the FFI execution context.
///
/// Returns null if the API is unavailable or allocation fails. The returned
///  pointer is device memory owned by the BFC pool -- callers must pair each
///  successful allocation with `free_device_memory`.
fn alloc_device_memory(frame: *c.XLA_FFI_CallFrame, size: usize, alignment: usize) ?*anyopaque {
    const ffi_api = frame.api orelse return null;
    const alloc_fn = ffi_api.*.XLA_FFI_DeviceMemory_Allocate orelse return null;
    var args: c.XLA_FFI_DeviceMemory_Allocate_Args = std.mem.zeroes(c.XLA_FFI_DeviceMemory_Allocate_Args);
    args.struct_size = @sizeOf(c.XLA_FFI_DeviceMemory_Allocate_Args);
    args.ctx = frame.ctx;
    args.size = size;
    args.alignment = alignment;
    const ffi_err = alloc_fn(&args);
    if (ffi_err != null) {
        log.warn("XLA_FFI_DeviceMemory_Allocate failed for {d} bytes", .{size});
        return null;
    }
    return args.data;
}

/// Free device memory previously allocated via `alloc_device_memory`.
///
/// Returns memory to XLA's BFC pool (not `cudaFree`). Logs a warning on
///  failure but does not propagate errors -- workspace cleanup is best-effort.
fn free_device_memory(frame: *c.XLA_FFI_CallFrame, data: *anyopaque, size: usize) void {
    const ffi_api = frame.api orelse {
        log.warn("free_device_memory: FFI API unavailable", .{});
        return;
    };
    const free_fn = ffi_api.*.XLA_FFI_DeviceMemory_Free orelse {
        log.warn("free_device_memory: XLA_FFI_DeviceMemory_Free unavailable", .{});
        return;
    };
    var args: c.XLA_FFI_DeviceMemory_Free_Args = std.mem.zeroes(c.XLA_FFI_DeviceMemory_Free_Args);
    args.struct_size = @sizeOf(c.XLA_FFI_DeviceMemory_Free_Args);
    args.ctx = frame.ctx;
    args.size = size;
    args.data = data;
    const ffi_err = free_fn(&args);
    if (ffi_err != null) {
        log.warn("XLA_FFI_DeviceMemory_Free failed for {d} bytes", .{size});
    }
}

fn handle_metadata_registration_hook(frame: *c.XLA_FFI_CallFrame) bool {
    const ext = frame.extension_start orelse return false;
    const ext_ptr: *c.XLA_FFI_Extension_Base = @ptrCast(ext);
    if (ext_ptr.type != c.XLA_FFI_Extension_Metadata) return false;

    const metadata_ext: *c.XLA_FFI_Metadata_Extension = @fieldParentPtr("extension_base", ext_ptr);
    const metadata = metadata_ext.metadata orelse return false;
    const metadata_ptr: *c.XLA_FFI_Metadata = @ptrCast(metadata);
    metadata_ptr.api_version.major_version = c.XLA_FFI_API_MAJOR;
    metadata_ptr.api_version.minor_version = c.XLA_FFI_API_MINOR;
    return true;
}

fn lookup_dispatch_attr(attrs: c.XLA_FFI_Attrs, name: []const u8) ?[]const u8 {
    var i: usize = 0;
    while (i < @as(usize, @intCast(attrs.size))) : (i += 1) {
        const name_span = attrs.names[i][0];
        const key = name_span.ptr[0..name_span.len];
        if (!std.mem.eql(u8, key, name)) continue;

        if (attrs.types[i] != c.XLA_FFI_AttrType_STRING) return null;
        const attr_value = attrs.attrs[i] orelse return null;
        const span: *c.XLA_FFI_ByteSpan = @ptrCast(@alignCast(attr_value));
        return span.ptr[0..span.len];
    }
    return null;
}

fn lookup_dispatch_attr_u32(attrs: c.XLA_FFI_Attrs, name: []const u8) ?u32 {
    var i: usize = 0;
    while (i < @as(usize, @intCast(attrs.size))) : (i += 1) {
        const name_span = attrs.names[i][0];
        const key = name_span.ptr[0..name_span.len];
        if (!std.mem.eql(u8, key, name)) continue;

        if (attrs.types[i] != c.XLA_FFI_AttrType_SCALAR) return null;
        const attr_value = attrs.attrs[i] orelse return null;
        const scalar: *c.XLA_FFI_Scalar = @ptrCast(@alignCast(attr_value));
        const value_ptr = scalar.value orelse return null;

        return switch (scalar.dtype) {
            c.XLA_FFI_DataType_S32 => blk: {
                const value: i32 = @as(*const i32, @ptrCast(@alignCast(value_ptr))).*;
                if (value < 0) break :blk null;
                break :blk @intCast(value);
            },
            c.XLA_FFI_DataType_U32 => @as(*const u32, @ptrCast(@alignCast(value_ptr))).*,
            c.XLA_FFI_DataType_S64 => blk: {
                const value: i64 = @as(*const i64, @ptrCast(@alignCast(value_ptr))).*;
                if (value < 0 or value > std.math.maxInt(u32)) break :blk null;
                break :blk @intCast(value);
            },
            c.XLA_FFI_DataType_U64 => blk: {
                const value: u64 = @as(*const u64, @ptrCast(@alignCast(value_ptr))).*;
                if (value > std.math.maxInt(u32)) break :blk null;
                break :blk @intCast(value);
            },
            else => null,
        };
    }
    return null;
}

fn get_arg_buffer(args: c.XLA_FFI_Args, idx: usize) ?*c.XLA_FFI_Buffer {
    if (idx >= @as(usize, @intCast(args.size))) return null;
    if (args.types[idx] != c.XLA_FFI_ArgType_BUFFER) return null;
    const ptr = args.args[idx] orelse return null;
    return @ptrCast(@alignCast(ptr));
}

fn get_ret_buffer(rets: c.XLA_FFI_Rets, idx: usize) ?*c.XLA_FFI_Buffer {
    if (idx >= @as(usize, @intCast(rets.size))) return null;
    if (rets.types[idx] != c.XLA_FFI_RetType_BUFFER) return null;
    const ptr = rets.rets[idx] orelse return null;
    return @ptrCast(@alignCast(ptr));
}

fn make_ffi_error(frame: *c.XLA_FFI_CallFrame, message: [:0]const u8, code: c.XLA_FFI_Error_Code) ?*c.XLA_FFI_Error {
    const ffi_api = frame.api orelse return null;
    const create_error = ffi_api.*.XLA_FFI_Error_Create orelse return null;
    var args: c.XLA_FFI_Error_Create_Args = std.mem.zeroes(c.XLA_FFI_Error_Create_Args);
    args.struct_size = @sizeOf(c.XLA_FFI_Error_Create_Args);
    args.message = message.ptr;
    args.errc = code;
    return create_error(&args);
}

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
        .f16 => .f16,
        .bf16 => .bf16,
        .f32 => .f32,
        .f64 => .f64,
        .i8 => .i8,
        .u8 => .u8,
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

test "dispatch_error_to_ffi returns null when frame has no API" {
    var frame: c.XLA_FFI_CallFrame = std.mem.zeroes(c.XLA_FFI_CallFrame);
    try std.testing.expect(dispatch_error_to_ffi(&frame, error.ShapeMismatch) == null);
}

test "dispatch_error_code maps invalid-argument class" {
    const expected: c.XLA_FFI_Error_Code = @intCast(c.XLA_FFI_Error_Code_INVALID_ARGUMENT);
    try std.testing.expectEqual(expected, dispatch_error_code(error.ShapeMismatch));
}

test "dispatch_error_code maps failed-precondition class" {
    const expected: c.XLA_FFI_Error_Code = @intCast(c.XLA_FFI_Error_Code_FAILED_PRECONDITION);
    try std.testing.expectEqual(expected, dispatch_error_code(error.WorkspaceUnavailable));
}

test "dispatch_error_code maps internal class" {
    const expected: c.XLA_FFI_Error_Code = @intCast(c.XLA_FFI_Error_Code_INTERNAL);
    try std.testing.expectEqual(expected, dispatch_error_code(error.DispatchFailed));
}

test "lookup_dispatch_user_data_from_context returns payload" {
    const testing = std.testing;

    const FfiStub = struct {
        var expected_type_id: i64 = 0;
        var payload: ?*anyopaque = null;

        fn execution_context_get(raw_args: [*c]c.XLA_FFI_ExecutionContext_Get_Args) callconv(.c) ?*c.XLA_FFI_Error {
            const args: *c.XLA_FFI_ExecutionContext_Get_Args = @ptrCast(raw_args);
            if (args.type_id != null and args.type_id.*.type_id == expected_type_id) {
                args.data = payload;
            } else {
                args.data = null;
            }
            return null;
        }
    };

    var payload_byte: u8 = 42;
    FfiStub.expected_type_id = 123;
    FfiStub.payload = @ptrCast(&payload_byte);

    var api: c.XLA_FFI_Api = std.mem.zeroes(c.XLA_FFI_Api);
    api.struct_size = @sizeOf(c.XLA_FFI_Api);
    api.XLA_FFI_ExecutionContext_Get = FfiStub.execution_context_get;

    var frame: c.XLA_FFI_CallFrame = std.mem.zeroes(c.XLA_FFI_CallFrame);
    frame.struct_size = @sizeOf(c.XLA_FFI_CallFrame);
    frame.api = &api;
    frame.ctx = @ptrFromInt(1);

    const looked_up = lookup_dispatch_user_data_from_context(&frame, FfiStub.expected_type_id);
    try testing.expect(looked_up != null);
    try testing.expectEqual(@intFromPtr(&payload_byte), @intFromPtr(looked_up.?));
}

test "lookup_dispatch_user_data_from_context returns null without api" {
    const testing = std.testing;

    var frame: c.XLA_FFI_CallFrame = std.mem.zeroes(c.XLA_FFI_CallFrame);
    frame.struct_size = @sizeOf(c.XLA_FFI_CallFrame);
    frame.api = null;

    try testing.expect(lookup_dispatch_user_data_from_context(&frame, 1) == null);
}
