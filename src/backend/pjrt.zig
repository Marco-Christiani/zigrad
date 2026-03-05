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
const kernel = @import("../kernel.zig");
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
var kernel_dispatch_platform: Platform = .unknown;

const DispatchTypeReg = struct {
    id: i64 = 0,
    registered: bool = false,
};

var dispatch_package_type: DispatchTypeReg = .{};
var dispatch_registry_type: DispatchTypeReg = .{};

const dispatch_package_type_name = "zigrad.kernel.package.v1";
const dispatch_registry_type_name = "zigrad.kernel.registry.v1";

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
    /// Optional kernel package sidecar for id-based dispatch lookup.
    ///
    /// The runtime first attempts `zigrad.kernel_id` package lookup when this
    /// field is set, then falls back to registry key lookup.
    kernel_package: ?*const kernel.KernelPackage = null,

    /// Optional registry sidecar for executable-scoped fallback key lookup.
    ///
    /// When provided, runtime resolves `zigrad.kernel_key` from this
    /// executable-bound registry before process-global fallback.
    kernel_registry: ?*const kernel.KernelRegistry = null,
};

/// Unified PJRT Backend.
///
/// Owns the complete PJRT lifecycle: plugin, client, and provides both
/// compilation and execution capabilities.
pub const Backend = struct {
    api: *pjrt_api.Api,
    client: pjrt_types.Client,
    allocator: std.mem.Allocator,
    platform: Platform,
    kernel_dispatch_registered: bool,

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

        return .{
            .api = api_ptr,
            .client = client,
            .allocator = allocator,
            .platform = platform,
            .kernel_dispatch_registered = false,
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
        var executable = try self.client.compile(device, format, mlir_bytes, compile_opts_pb);
        executable.dispatch_sidecar = if (options.kernel_package) |package| @ptrCast(package) else null;
        executable.dispatch_registry_sidecar = if (options.kernel_registry) |registry| @ptrCast(registry) else null;
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

    pub fn execute(self: *Backend, exe: *LoadedExecutable, allocator: std.mem.Allocator, inputs: []const Buffer) !ExecuteResult {
        const execute_context = try create_dispatch_execute_context(self, exe);
        defer if (execute_context) |ctx| destroy_execute_context(self.api, ctx);
        return exe.execute_with_context(self.api, allocator, inputs, execute_context);
    }

    pub fn execute_into(self: *Backend, exe: *LoadedExecutable, input_ptrs: []const RawBuffer, output_ptrs: []?RawBuffer, non_donatable: ?[]const i64) !?Event {
        const execute_context = try create_dispatch_execute_context(self, exe);
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
    /// Runtime registry/package bindings are now passed per-executable via
    /// compile options sidecars and per-execute context user data.
    pub fn register_kernel_dispatcher(self: *Backend) !void {
        try self.require_typed_ffi();
        if (self.kernel_dispatch_registered) return;

        if (kernel_dispatch_target_registered) {
            kernel_dispatch_platform = self.platform;
            self.kernel_dispatch_registered = true;
            return;
        }

        const ffi_ext = self.api.ffi_extension() orelse return error.TypedFfiUnavailable;

        if (!try register_ffi_for_platform(self.api, ffi_ext, self.platform))
            return error.TypedFfiRegistrationFailed;

        kernel_dispatch_platform = self.platform;
        self.kernel_dispatch_registered = true;
    }

    pub fn device_memory_stats(self: *Backend, device: *const Device) !Device.MemoryStats {
        return device.get_memory_stats(self.api);
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

    log.debug(
        "typed-ffi register_handler failed target='{s}' platform='{s}' code={d} msg={s}",
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
        .data = @ptrCast(@constCast(data_ptr)),
    };

    if (add_user_data(&add_args)) |pjrt_err| {
        try pjrt_error_to_zig(api, pjrt_err);
    }
}

fn create_dispatch_execute_context(self: *Backend, executable: *LoadedExecutable) !?*c.PJRT_ExecuteContext {
    const package_ptr = executable.dispatch_sidecar;
    const registry_ptr = executable.dispatch_registry_sidecar;

    if (package_ptr == null and registry_ptr == null) return null;

    const ffi_ext = self.api.ffi_extension() orelse return error.TypedFfiUnavailable;

    var create_args = pjrt_api.init_args(c.PJRT_ExecuteContext_Create_Args);
    create_args.context = null;
    try self.api.call("PJRT_ExecuteContext_Create", &create_args);

    const context = create_args.context orelse return error.PjrtReturnedNullExecuteContext;
    errdefer destroy_execute_context(self.api, context);

    if (package_ptr) |ptr| {
        const type_id = try ensure_dispatch_type_id(self.api, ffi_ext, dispatch_package_type_name, &dispatch_package_type);
        try add_dispatch_user_data(self.api, ffi_ext, context, type_id, ptr);
    }

    if (registry_ptr) |ptr| {
        const type_id = try ensure_dispatch_type_id(self.api, ffi_ext, dispatch_registry_type_name, &dispatch_registry_type);
        try add_dispatch_user_data(self.api, ffi_ext, context, type_id, ptr);
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

fn lookup_dispatch_package_from_context(frame: *c.XLA_FFI_CallFrame) ?*const kernel.KernelPackage {
    if (!dispatch_package_type.registered) return null;
    const data_ptr = lookup_dispatch_user_data_from_context(frame, dispatch_package_type.id) orelse return null;
    return @ptrCast(@alignCast(data_ptr));
}

fn lookup_dispatch_registry_from_context(frame: *c.XLA_FFI_CallFrame) ?*const kernel.KernelRegistry {
    if (!dispatch_registry_type.registered) return null;
    const data_ptr = lookup_dispatch_user_data_from_context(frame, dispatch_registry_type.id) orelse return null;
    return @ptrCast(@alignCast(data_ptr));
}

fn detect_platform(plugin_path: []const u8) Platform {
    if (std.mem.endsWith(u8, plugin_path, "pjrt_c_api_cpu_plugin.so")) return .host;
    if (std.mem.endsWith(u8, plugin_path, "pjrt_c_api_gpu_plugin.so")) return .cuda;
    if (std.mem.indexOf(u8, plugin_path, "cpu") != null) return .host;
    if (std.mem.indexOf(u8, plugin_path, "cuda") != null or std.mem.indexOf(u8, plugin_path, "gpu") != null) return .cuda;
    return .unknown;
}

const DispatchLookupError = error{
    NoRegistryOrPackage,
    MissingKernelKey,
    MissingKernelKeyForFallback,
    KernelNotFound,
};

const ResolvedDispatch = struct {
    artifact: kernel.KernelArtifact,
    dispatch_key: []const u8,
};

fn resolve_dispatch_artifact(
    registry: ?*const kernel.KernelRegistry,
    package: ?*const kernel.KernelPackage,
    maybe_kernel_key: ?[]const u8,
    maybe_kernel_id: ?u32,
) DispatchLookupError!ResolvedDispatch {
    if (registry == null and package == null) return error.NoRegistryOrPackage;

    if (maybe_kernel_id) |kernel_id| {
        if (package) |pkg| {
            if (pkg.get(kernel_id)) |pkg_artifact| {
                return .{
                    .artifact = pkg_artifact,
                    .dispatch_key = maybe_kernel_key orelse pkg_artifact.target_name,
                };
            }

            if (registry) |reg| {
                const kernel_key = maybe_kernel_key orelse return error.MissingKernelKeyForFallback;
                const artifact = reg.get(kernel_key) orelse return error.KernelNotFound;
                return .{ .artifact = artifact, .dispatch_key = kernel_key };
            }

            return error.KernelNotFound;
        }

        if (registry) |reg| {
            const kernel_key = maybe_kernel_key orelse return error.MissingKernelKeyForFallback;
            const artifact = reg.get(kernel_key) orelse return error.KernelNotFound;
            return .{ .artifact = artifact, .dispatch_key = kernel_key };
        }

        return error.NoRegistryOrPackage;
    }

    const kernel_key = maybe_kernel_key orelse return error.MissingKernelKey;
    const reg = registry orelse return error.NoRegistryOrPackage;
    const artifact = reg.get(kernel_key) orelse return error.KernelNotFound;
    return .{ .artifact = artifact, .dispatch_key = kernel_key };
}

/// Generic kernel dispatch handler invoked by the XLA FFI framework.
///
/// Extracts kernel_key from custom_call attributes, looks up the artifact
/// in the registry, builds a provider-agnostic DispatchContext from the
/// FFI frame, and delegates to `artifact.dispatch()`.
fn kernel_dispatch_handler(frame: *c.XLA_FFI_CallFrame) callconv(.c) ?*c.XLA_FFI_Error {
    if (handle_metadata_registration_hook(frame)) return null;

    if (frame.stage != c.XLA_FFI_ExecutionStage_EXECUTE) return null;

    const registry = lookup_dispatch_registry_from_context(frame);

    const maybe_kernel_key = lookup_dispatch_attr(frame.attrs, "zigrad.kernel_key");
    const maybe_kernel_id = lookup_dispatch_attr_u32(frame.attrs, "zigrad.kernel_id");
    const provider = lookup_dispatch_attr(frame.attrs, "zigrad.provider") orelse "";
    const package = lookup_dispatch_package_from_context(frame);

    const resolved = resolve_dispatch_artifact(registry, package, maybe_kernel_key, maybe_kernel_id) catch |err| switch (err) {
        error.NoRegistryOrPackage => return make_ffi_error(frame, "zigrad kernel dispatch: no registry or package is configured", c.XLA_FFI_Error_Code_FAILED_PRECONDITION),
        error.MissingKernelKey => return make_ffi_error(frame, "zigrad kernel dispatch: missing zigrad.kernel_key attribute", c.XLA_FFI_Error_Code_INVALID_ARGUMENT),
        error.MissingKernelKeyForFallback => return make_ffi_error(frame, "zigrad kernel dispatch: kernel key required for fallback lookup", c.XLA_FFI_Error_Code_INVALID_ARGUMENT),
        error.KernelNotFound => return make_ffi_error(frame, "zigrad kernel dispatch: kernel id/key not found", c.XLA_FFI_Error_Code_NOT_FOUND),
    };

    const artifact = resolved.artifact;
    const dispatch_key = resolved.dispatch_key;

    if (provider.len != 0 and !std.mem.eql(u8, provider, artifact.provider_name)) {
        return make_ffi_error(frame, "zigrad kernel dispatch: provider mismatch for kernel key", c.XLA_FFI_Error_Code_FAILED_PRECONDITION);
    }

    // Extract all input and output buffers from the FFI frame.
    var input_descs: [16]kernel.BufferDesc = undefined;
    const num_inputs = extract_buffers(frame.args.size, frame.args.types, frame.args.args, &input_descs) orelse {
        return make_ffi_error(frame, "zigrad kernel dispatch: failed to extract input buffers", c.XLA_FFI_Error_Code_INVALID_ARGUMENT);
    };

    var output_descs: [16]kernel.BufferDesc = undefined;
    const num_outputs = extract_ret_buffers(frame.rets.size, frame.rets.types, frame.rets.rets, &output_descs) orelse {
        return make_ffi_error(frame, "zigrad kernel dispatch: failed to extract output buffers", c.XLA_FFI_Error_Code_INVALID_ARGUMENT);
    };

    const platform: kernel.DispatchPlatform = switch (kernel_dispatch_platform) {
        .cuda => .cuda,
        .host, .unknown => .host,
    };

    const ctx = kernel.DispatchContext{
        .inputs = input_descs[0..num_inputs],
        .outputs = output_descs[0..num_outputs],
        .device_ordinal = get_device_ordinal(frame),
        .platform = platform,
        .stream = get_stream(frame),
        .workspace = null,
        .workspace_bytes_required = artifact.workspace_bytes,
        .allocator = std.heap.c_allocator,
    };

    artifact.dispatch(dispatch_key, ctx) catch |err| {
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
        error.MirageInvalidArgument,
        => c.XLA_FFI_Error_Code_INVALID_ARGUMENT,

        error.MirageLoadFailed,
        error.MirageApiUnsupported,
        error.WorkspaceUnavailable,
        => c.XLA_FFI_Error_Code_FAILED_PRECONDITION,

        error.DispatchFailed,
        error.OutOfMemory,
        error.MirageInternalError,
        error.MirageContractError,
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

test "dispatch_error_to_ffi returns null when frame has no API" {
    var frame: c.XLA_FFI_CallFrame = std.mem.zeroes(c.XLA_FFI_CallFrame);
    try std.testing.expect(dispatch_error_to_ffi(&frame, error.MirageInvalidArgument) == null);
}

test "dispatch_error_code maps invalid-argument class" {
    const expected: c.XLA_FFI_Error_Code = @intCast(c.XLA_FFI_Error_Code_INVALID_ARGUMENT);
    try std.testing.expectEqual(expected, dispatch_error_code(error.MirageInvalidArgument));
}

test "dispatch_error_code maps failed-precondition class" {
    const expected: c.XLA_FFI_Error_Code = @intCast(c.XLA_FFI_Error_Code_FAILED_PRECONDITION);
    try std.testing.expectEqual(expected, dispatch_error_code(error.WorkspaceUnavailable));
}

test "dispatch_error_code maps internal class" {
    const expected: c.XLA_FFI_Error_Code = @intCast(c.XLA_FFI_Error_Code_INTERNAL);
    try std.testing.expectEqual(expected, dispatch_error_code(error.MirageInternalError));
}

test "resolve_dispatch_artifact prefers package entry for kernel id" {
    const testing = std.testing;

    var registry = kernel.KernelRegistry.init(testing.allocator);
    defer registry.deinit();
    var package = kernel.KernelPackage.init(testing.allocator);
    defer package.deinit();

    try registry.put("kernel_key", .{
        .provider_name = "mock",
        .data = try testing.allocator.dupe(u8, "registry_data"),
        .target_name = "kernel_key",
    });

    try package.put(7, .{
        .provider_name = "mock",
        .data = try testing.allocator.dupe(u8, "package_data"),
        .target_name = "kernel_key",
    });

    const resolved = try resolve_dispatch_artifact(&registry, &package, "kernel_key", 7);
    try testing.expectEqualStrings("package_data", resolved.artifact.data);
    try testing.expectEqualStrings("kernel_key", resolved.dispatch_key);
}

test "resolve_dispatch_artifact falls back to registry on package miss" {
    const testing = std.testing;

    var registry = kernel.KernelRegistry.init(testing.allocator);
    defer registry.deinit();
    var package = kernel.KernelPackage.init(testing.allocator);
    defer package.deinit();

    try registry.put("kernel_key", .{
        .provider_name = "mock",
        .data = try testing.allocator.dupe(u8, "registry_data"),
        .target_name = "kernel_key",
    });

    const resolved = try resolve_dispatch_artifact(&registry, &package, "kernel_key", 99);
    try testing.expectEqualStrings("registry_data", resolved.artifact.data);
    try testing.expectEqualStrings("kernel_key", resolved.dispatch_key);
}

test "resolve_dispatch_artifact allows package lookup without kernel key" {
    const testing = std.testing;

    var package = kernel.KernelPackage.init(testing.allocator);
    defer package.deinit();

    try package.put(42, .{
        .provider_name = "mock",
        .data = try testing.allocator.dupe(u8, "package_data"),
        .target_name = "pkg_kernel",
    });

    const resolved = try resolve_dispatch_artifact(null, &package, null, 42);
    try testing.expectEqualStrings("package_data", resolved.artifact.data);
    try testing.expectEqualStrings("pkg_kernel", resolved.dispatch_key);
}

test "resolve_dispatch_artifact requires key for registry fallback" {
    const testing = std.testing;

    var registry = kernel.KernelRegistry.init(testing.allocator);
    defer registry.deinit();

    try registry.put("kernel_key", .{
        .provider_name = "mock",
        .data = try testing.allocator.dupe(u8, "registry_data"),
        .target_name = "kernel_key",
    });

    try testing.expectError(error.MissingKernelKeyForFallback, resolve_dispatch_artifact(&registry, null, null, 5));
}

test "resolve_dispatch_artifact fails when no registry or package provided" {
    const testing = std.testing;
    try testing.expectError(error.NoRegistryOrPackage, resolve_dispatch_artifact(null, null, "kernel_key", null));
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
