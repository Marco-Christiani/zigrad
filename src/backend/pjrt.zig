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
const dlpack = @import("../c/dlpack.zig");
const tvm_api = @import("../c/tvm/api.zig");
const tvm_runtime = @import("../c/tvm/runtime.zig");
const tvm_c = @import("../c/tvm/c.zig");
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

var kernel_dispatch_registry: ?*const kernel.KernelRegistry = null;
var kernel_dispatch_target_registered: bool = false;
var kernel_dispatch_platform: Platform = .unknown;

const TvmDispatchEntry = struct {
    module: tvm_runtime.RuntimeModule,
    main_func: tvm_api.Value,
};

var tvm_dispatch_cache_mutex = std.Thread.Mutex{};
var tvm_dispatch_cache: ?std.StringHashMap(TvmDispatchEntry) = null;
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
    /// This is a process-global bridge used by the current checkpoint path.
    /// Caller must ensure `registry` outlives all executable invocations that
    /// may call into the dispatcher.
    pub fn register_kernel_dispatcher(self: *Backend, registry: *const kernel.KernelRegistry) !void {
        try self.require_typed_ffi();
        if (self.kernel_dispatch_registered) {
            kernel_dispatch_registry = registry;
            kernel_dispatch_platform = self.platform;
            return;
        }

        if (kernel_dispatch_target_registered) {
            kernel_dispatch_registry = registry;
            kernel_dispatch_platform = self.platform;
            self.kernel_dispatch_registered = true;
            return;
        }

        const ffi_ext = self.api.ffi_extension() orelse return error.TypedFfiUnavailable;

        var registered_any = false;
        switch (self.platform) {
            .host => {
                const host_title = try register_dispatch_handler_for_platform(self.api, ffi_ext, "Host");
                const host_lower = try register_dispatch_handler_for_platform(self.api, ffi_ext, "host");
                registered_any = host_title or host_lower;
            },
            .cuda => {
                const cuda_title = try register_dispatch_handler_for_platform(self.api, ffi_ext, "CUDA");
                const cuda_lower = try register_dispatch_handler_for_platform(self.api, ffi_ext, "cuda");
                registered_any = cuda_title or cuda_lower;
            },
            .unknown => {
                const host_ok = try register_dispatch_handler_for_platform(self.api, ffi_ext, "Host");
                const host_ok_lower = try register_dispatch_handler_for_platform(self.api, ffi_ext, "host");
                const cuda_ok = try register_dispatch_handler_for_platform(self.api, ffi_ext, "CUDA");
                const cuda_ok_lower = try register_dispatch_handler_for_platform(self.api, ffi_ext, "cuda");
                registered_any = host_ok or host_ok_lower or cuda_ok or cuda_ok_lower;
            },
        }

        if (!registered_any) return error.TypedFfiRegistrationFailed;

        kernel_dispatch_registry = registry;
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
        switch (platform) {
            .host => {
                const host_title = try register_dispatch_handler_for_platform(api, ffi_ext, "Host");
                const host_lower = try register_dispatch_handler_for_platform(api, ffi_ext, "host");
                registered_any = registered_any or host_title or host_lower;
            },
            .cuda => {
                const cuda_title = try register_dispatch_handler_for_platform(api, ffi_ext, "CUDA");
                const cuda_lower = try register_dispatch_handler_for_platform(api, ffi_ext, "cuda");
                registered_any = registered_any or cuda_title or cuda_lower;
            },
            .unknown => {
                const host_ok = try register_dispatch_handler_for_platform(api, ffi_ext, "Host");
                const host_ok_lower = try register_dispatch_handler_for_platform(api, ffi_ext, "host");
                const cuda_ok = try register_dispatch_handler_for_platform(api, ffi_ext, "CUDA");
                const cuda_ok_lower = try register_dispatch_handler_for_platform(api, ffi_ext, "cuda");
                registered_any = registered_any or host_ok or host_ok_lower or cuda_ok or cuda_ok_lower;
            },
        }
    }

    if (api.gpu_custom_call_extension()) |gpu_ext| {
        const gpu_registered = try register_dispatch_handler_via_gpu_extension(api, gpu_ext);
        registered_any = registered_any or gpu_registered;
    }

    kernel_dispatch_target_registered = registered_any;
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

fn detect_platform(plugin_path: []const u8) Platform {
    if (std.mem.endsWith(u8, plugin_path, "pjrt_c_api_cpu_plugin.so")) return .host;
    if (std.mem.endsWith(u8, plugin_path, "pjrt_c_api_gpu_plugin.so")) return .cuda;
    if (std.mem.indexOf(u8, plugin_path, "cpu") != null) return .host;
    if (std.mem.indexOf(u8, plugin_path, "cuda") != null or std.mem.indexOf(u8, plugin_path, "gpu") != null) return .cuda;
    return .unknown;
}

fn kernel_dispatch_handler(frame: *c.XLA_FFI_CallFrame) callconv(.c) ?*c.XLA_FFI_Error {
    if (handle_metadata_registration_hook(frame)) return null;

    if (frame.stage != c.XLA_FFI_ExecutionStage_EXECUTE) return null;

    const registry = kernel_dispatch_registry orelse {
        return make_ffi_error(frame, "zigrad kernel dispatch: kernel registry is not configured", c.XLA_FFI_Error_Code_FAILED_PRECONDITION);
    };

    const kernel_key = lookup_dispatch_attr(frame.attrs, "zigrad.kernel_key") orelse {
        return make_ffi_error(frame, "zigrad kernel dispatch: missing zigrad.kernel_key attribute", c.XLA_FFI_Error_Code_INVALID_ARGUMENT);
    };
    const provider = lookup_dispatch_attr(frame.attrs, "zigrad.provider") orelse "";

    const artifact = registry.get(kernel_key) orelse {
        return make_ffi_error(frame, "zigrad kernel dispatch: kernel key not found in registry", c.XLA_FFI_Error_Code_NOT_FOUND);
    };

    if (provider.len != 0 and !std.mem.eql(u8, provider, artifact.provider_name)) {
        return make_ffi_error(frame, "zigrad kernel dispatch: provider mismatch for kernel key", c.XLA_FFI_Error_Code_FAILED_PRECONDITION);
    }

    const a = get_arg_buffer(frame.args, 0) orelse {
        return make_ffi_error(frame, "zigrad kernel dispatch: missing arg0 buffer", c.XLA_FFI_Error_Code_INVALID_ARGUMENT);
    };
    const b = get_arg_buffer(frame.args, 1) orelse {
        return make_ffi_error(frame, "zigrad kernel dispatch: missing arg1 buffer", c.XLA_FFI_Error_Code_INVALID_ARGUMENT);
    };
    const out = get_ret_buffer(frame.rets, 0) orelse {
        return make_ffi_error(frame, "zigrad kernel dispatch: missing ret0 buffer", c.XLA_FFI_Error_Code_INVALID_ARGUMENT);
    };

    if (a.rank != 2 or b.rank != 2 or out.rank != 2) {
        return make_ffi_error(frame, "zigrad kernel dispatch: expected rank-2 buffers", c.XLA_FFI_Error_Code_INVALID_ARGUMENT);
    }
    if (a.dtype != c.XLA_FFI_DataType_F32 or b.dtype != c.XLA_FFI_DataType_F32 or out.dtype != c.XLA_FFI_DataType_F32) {
        return make_ffi_error(frame, "zigrad kernel dispatch: expected f32 buffers", c.XLA_FFI_Error_Code_INVALID_ARGUMENT);
    }

    const m: usize = @intCast(a.dims[0]);
    const k: usize = @intCast(a.dims[1]);
    const kb: usize = @intCast(b.dims[0]);
    const n: usize = @intCast(b.dims[1]);
    const out_m: usize = @intCast(out.dims[0]);
    const out_n: usize = @intCast(out.dims[1]);
    if (k != kb or m != out_m or n != out_n) {
        return make_ffi_error(frame, "zigrad kernel dispatch: matmul shape mismatch", c.XLA_FFI_Error_Code_INVALID_ARGUMENT);
    }

    if (std.mem.eql(u8, artifact.provider_name, "tvm")) {
        dispatch_tvm_kernel(frame, artifact, kernel_key, a, b, out) catch {
            return make_ffi_error(frame, "zigrad kernel dispatch: tvm execution failed", c.XLA_FFI_Error_Code_INTERNAL);
        };
        return null;
    }

    return make_ffi_error(frame, "zigrad kernel dispatch: unsupported provider", c.XLA_FFI_Error_Code_UNIMPLEMENTED);
}

fn dispatch_tvm_kernel(
    frame: *c.XLA_FFI_CallFrame,
    artifact: kernel.KernelArtifact,
    kernel_key: []const u8,
    a: *c.XLA_FFI_Buffer,
    b: *c.XLA_FFI_Buffer,
    out: *c.XLA_FFI_Buffer,
) !void {
    try tvm_api.ensure_loaded(std.heap.c_allocator);

    var entry: TvmDispatchEntry = undefined;
    {
        tvm_dispatch_cache_mutex.lock();
        defer tvm_dispatch_cache_mutex.unlock();

        var cache = get_tvm_dispatch_cache();
        if (cache.get(kernel_key)) |cached| {
            entry = cached;
        } else {
            const loaded = try load_tvm_dispatch_entry(kernel_key, artifact);
            const cache_key = try std.heap.c_allocator.dupe(u8, kernel_key);
            try cache.put(cache_key, loaded);
            entry = loaded;
        }
    }

    const m: usize = @intCast(a.dims[0]);
    const k: usize = @intCast(a.dims[1]);
    const n: usize = @intCast(b.dims[1]);

    const device_id = get_device_ordinal(frame);

    if (kernel_dispatch_platform == .cuda) {
        try configure_tvm_cuda_stream(frame, device_id);
    }

    const device_type: dlpack.DeviceType = if (kernel_dispatch_platform == .cuda) .cuda else .cpu;

    var shape_a = [_]i64{ @intCast(m), @intCast(k) };
    var shape_b = [_]i64{ @intCast(k), @intCast(n) };
    var shape_out = [_]i64{ @intCast(m), @intCast(n) };

    var a_tensor = try tensor_from_ffi_buffer(a, shape_a[0..], device_type, device_id);
    defer a_tensor.deinit();
    var b_tensor = try tensor_from_ffi_buffer(b, shape_b[0..], device_type, device_id);
    defer b_tensor.deinit();
    var out_tensor = try tensor_from_ffi_buffer(out, shape_out[0..], device_type, device_id);
    defer out_tensor.deinit();

    const func_handle = entry.main_func.as_object() orelse return error.TvmCallFailed;
    _ = try tvm_api.call_handle(std.heap.c_allocator, func_handle, &.{
        a_tensor.as_value(),
        b_tensor.as_value(),
        out_tensor.as_value(),
    });
}

fn tensor_from_ffi_buffer(buffer: *c.XLA_FFI_Buffer, shape: []i64, device_type: dlpack.DeviceType, device_id: i32) !tvm_runtime.Tensor {
    const dl_tensor: dlpack.Tensor = .{
        .data = @ptrCast(buffer.data),
        .device = .{ .device_type = device_type, .device_id = device_id },
        .ndim = @intCast(shape.len),
        .dtype = dlpack.DataType.f32_,
        .shape = shape.ptr, // heap_borrowing dupes this
        .strides = null,
        .byte_offset = 0,
    };
    const managed = try dlpack.ManagedTensor.heap_borrowing(std.heap.c_allocator, dl_tensor);
    return tvm_runtime.Tensor.from_dlpack(managed);
}

fn configure_tvm_cuda_stream(frame: *c.XLA_FFI_CallFrame, device_id: i32) !void {
    const ffi_api = frame.api orelse return error.TvmCallFailed;
    const get_stream = ffi_api.*.XLA_FFI_Stream_Get orelse return error.TvmCallFailed;

    var args: c.XLA_FFI_Stream_Get_Args = std.mem.zeroes(c.XLA_FFI_Stream_Get_Args);
    args.struct_size = @sizeOf(c.XLA_FFI_Stream_Get_Args);
    args.ctx = frame.ctx;
    const ffi_err = get_stream(&args);
    if (ffi_err != null) return error.TvmCallFailed;
    const stream_ptr = args.stream orelse return error.TvmCallFailed;

    _ = try tvm_api.call_global(std.heap.c_allocator, "runtime.TVMSetStream", &.{
        tvm_api.Value.int(2), // kDLCUDA
        tvm_api.Value.int(device_id),
        tvm_opaque_ptr_value(stream_ptr),
    });
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

fn tvm_opaque_ptr_value(ptr: *anyopaque) tvm_api.Value {
    var v = std.mem.zeroes(tvm_c.TVMFFIAny);
    v.type_index = tvm_c.kTVMFFIOpaquePtr;
    v.unnamed_1.v_int64 = @bitCast(@intFromPtr(ptr));
    return .{ .raw = v };
}

fn load_tvm_dispatch_entry(kernel_key: []const u8, artifact: kernel.KernelArtifact) !TvmDispatchEntry {
    const hash = std.hash.Wyhash.hash(0, kernel_key);
    const path = try std.fmt.allocPrintSentinel(std.heap.c_allocator, "/tmp/zigrad-kernel-{x}.so", .{hash}, 0);
    defer std.heap.c_allocator.free(path);

    const file = try std.fs.createFileAbsolute(path, .{ .truncate = true });
    defer file.close();
    try file.writeAll(artifact.data);

    var module = try tvm_runtime.RuntimeModule.load_from_file(std.heap.c_allocator, path);
    errdefer module.deinit();

    const main_func = try module.get_function(std.heap.c_allocator, "main", true);
    return .{
        .module = module,
        .main_func = main_func,
    };
}

fn get_tvm_dispatch_cache() *std.StringHashMap(TvmDispatchEntry) {
    if (tvm_dispatch_cache == null) {
        tvm_dispatch_cache = std.StringHashMap(TvmDispatchEntry).init(std.heap.c_allocator);
    }
    return &tvm_dispatch_cache.?;
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
