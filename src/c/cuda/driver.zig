//! CUDA driver ABI adapter for module management and kernel launch.
//!
//! Raw CUDA handles, result codes, and function signatures remain private.
const std = @import("std");

const log = std.log.scoped(.@"zg/cuda_driver");

const RawModule = *anyopaque;
const RawFunction = *anyopaque;
const RawStream = *anyopaque;
const RawResult = c_int;

const success: RawResult = 0;

const FnModuleLoadData = *const fn (*RawModule, *const anyopaque) callconv(.c) RawResult;
const FnInit = *const fn (c_uint) callconv(.c) RawResult;
const FnDeviceGet = *const fn (*c_int, c_int) callconv(.c) RawResult;
const FnDeviceGetAttribute = *const fn (*c_int, c_int, c_int) callconv(.c) RawResult;
const FnModuleUnload = *const fn (RawModule) callconv(.c) RawResult;
const FnModuleGetFunction = *const fn (*RawFunction, RawModule, [*:0]const u8) callconv(.c) RawResult;
const FnLaunchKernel = *const fn (
    RawFunction,
    c_uint, // gridDimX
    c_uint, // gridDimY
    c_uint, // gridDimZ
    c_uint, // blockDimX
    c_uint, // blockDimY
    c_uint, // blockDimZ
    c_uint, // sharedMemBytes
    ?RawStream, // hStream
    ?[*]?*anyopaque, // kernelParams
    ?[*]?*anyopaque, // extra
) callconv(.c) RawResult;
const FnFuncSetAttribute = *const fn (RawFunction, c_int, c_int) callconv(.c) RawResult;

const max_dynamic_shared_size_bytes: c_int = 8;

var fn_module_load_data: ?FnModuleLoadData = null;
var fn_init: ?FnInit = null;
var fn_device_get: ?FnDeviceGet = null;
var fn_device_get_attribute: ?FnDeviceGetAttribute = null;
var fn_module_unload: ?FnModuleUnload = null;
var fn_module_get_function: ?FnModuleGetFunction = null;
var fn_launch_kernel: ?FnLaunchKernel = null;
var fn_func_set_attribute: ?FnFuncSetAttribute = null;

var cuda_handle: ?*anyopaque = null;
var symbols_ready = false;

pub const LoadError = error{
    CudaDriverNotFound,
    CudaSymbolMissing,
};

pub const Error = LoadError || error{
    CudaInvalidImage,
    CudaInitializationFailed,
    CudaDeviceNotFound,
    CudaModuleLoadFailed,
    CudaModuleLookupFailed,
    CudaAttributeFailed,
    CudaLaunchFailed,
};

const RTLD_NOW: c_int = 0x2;

extern "c" fn dlopen(filename: [*:0]const u8, flags: c_int) ?*anyopaque;
extern "c" fn dlsym(handle: *anyopaque, symbol: [*:0]const u8) ?*anyopaque;
extern "c" fn dlerror() ?[*:0]const u8;

pub fn ensure_loaded() LoadError!void {
    if (symbols_ready) return;

    if (cuda_handle == null) {
        // The installed NVIDIA driver supplies the stable CUDA driver soname.
        cuda_handle = dlopen("libcuda.so.1", RTLD_NOW);
        if (cuda_handle == null) {
            if (dlerror()) |err| {
                log.err("failed to load CUDA driver: {s}", .{std.mem.span(err)});
            }
            return error.CudaDriverNotFound;
        }
    }

    const handle = cuda_handle.?;
    fn_init = try load_symbol(FnInit, handle, "cuInit");
    fn_device_get = try load_symbol(FnDeviceGet, handle, "cuDeviceGet");
    fn_device_get_attribute = try load_symbol(
        FnDeviceGetAttribute,
        handle,
        "cuDeviceGetAttribute",
    );
    fn_module_load_data = try load_symbol(FnModuleLoadData, handle, "cuModuleLoadData");
    fn_module_unload = try load_symbol(FnModuleUnload, handle, "cuModuleUnload");
    fn_module_get_function = try load_symbol(FnModuleGetFunction, handle, "cuModuleGetFunction");
    fn_launch_kernel = try load_symbol(FnLaunchKernel, handle, "cuLaunchKernel");
    fn_func_set_attribute = load_symbol_optional(FnFuncSetAttribute, handle, "cuFuncSetAttribute");
    symbols_ready = true;
}

/// CUDA compute capability for one driver device.
pub const ComputeCapability = struct {
    major: i32,
    minor: i32,
};

const compute_capability_major: c_int = 75;
const compute_capability_minor: c_int = 76;

/// Query the compute capability of one CUDA device ordinal.
pub fn device_compute_capability(ordinal: i32) Error!ComputeCapability {
    if (ordinal < 0) return error.CudaDeviceNotFound;
    try ensure_loaded();

    const init_result = fn_init.?(0);
    if (init_result != success) {
        log.err("cuInit failed: {d}", .{init_result});
        return error.CudaInitializationFailed;
    }

    var raw_device: c_int = undefined;
    const device_result = fn_device_get.?(&raw_device, ordinal);
    if (device_result != success) {
        log.err("cuDeviceGet({d}) failed: {d}", .{ ordinal, device_result });
        return error.CudaDeviceNotFound;
    }

    var capability: ComputeCapability = undefined;
    const major_result = fn_device_get_attribute.?(
        &capability.major,
        compute_capability_major,
        raw_device,
    );
    const minor_result = fn_device_get_attribute.?(
        &capability.minor,
        compute_capability_minor,
        raw_device,
    );
    if (major_result != success or minor_result != success) {
        log.err(
            "CUDA compute capability query failed: major={d}, minor={d}",
            .{ major_result, minor_result },
        );
        return error.CudaAttributeFailed;
    }
    return capability;
}

/// Opaque CUDA module handle released by `unload`.
pub const ModuleHandle = opaque {
    /// Loads PTX or a CUDA binary image.
    pub fn load(image: []const u8) Error!*ModuleHandle {
        if (image.len == 0) return error.CudaInvalidImage;
        try ensure_loaded();

        var raw: RawModule = undefined;
        const result = fn_module_load_data.?(&raw, image.ptr);
        if (result != success) {
            log.err("cuModuleLoadData failed: {d}", .{result});
            return error.CudaModuleLoadFailed;
        }
        return @ptrCast(raw);
    }

    pub fn unload(self: *ModuleHandle) void {
        const f = fn_module_unload orelse return;
        const result = f(@ptrCast(self));
        if (result != success) {
            log.warn("cuModuleUnload failed: {d}", .{result});
        }
    }

    /// Resolves a function that remains valid until this module is unloaded.
    pub fn function(self: *ModuleHandle, name: [:0]const u8) Error!*FunctionHandle {
        var raw: RawFunction = undefined;
        const f = fn_module_get_function orelse return error.CudaSymbolMissing;
        const result = f(&raw, @ptrCast(self), name.ptr);
        if (result != success) {
            log.err("cuModuleGetFunction('{s}') failed: {d}", .{ name, result });
            return error.CudaModuleLookupFailed;
        }
        return @ptrCast(raw);
    }
};

pub const LaunchOptions = struct {
    grid_dim: [3]u32,
    block_dim: [3]u32,
    shared_memory_bytes: u32 = 0,
    stream: ?*anyopaque = null,
    params: []?*anyopaque,
};

/// Opaque function handle that remains valid until its module is unloaded.
pub const FunctionHandle = opaque {
    pub fn set_max_dynamic_shared_memory(self: *FunctionHandle, bytes: u32) Error!void {
        const f = fn_func_set_attribute orelse return error.CudaSymbolMissing;
        const result = f(
            @ptrCast(self),
            max_dynamic_shared_size_bytes,
            @intCast(bytes),
        );
        if (result != success) {
            log.err("cuFuncSetAttribute failed: {d}", .{result});
            return error.CudaAttributeFailed;
        }
    }

    pub fn launch(self: *FunctionHandle, options: LaunchOptions) Error!void {
        const f = fn_launch_kernel orelse return error.CudaSymbolMissing;
        const stream: ?RawStream = if (options.stream) |value| @ptrCast(value) else null;
        const result = f(
            @ptrCast(self),
            options.grid_dim[0],
            options.grid_dim[1],
            options.grid_dim[2],
            options.block_dim[0],
            options.block_dim[1],
            options.block_dim[2],
            options.shared_memory_bytes,
            stream,
            if (options.params.len == 0) null else options.params.ptr,
            null,
        );
        if (result != success) {
            log.err("cuLaunchKernel failed: {d}", .{result});
            return error.CudaLaunchFailed;
        }
    }
};

fn load_symbol(comptime T: type, handle: *anyopaque, comptime symbol: [:0]const u8) LoadError!T {
    _ = dlerror();
    const raw = dlsym(handle, symbol.ptr) orelse {
        if (dlerror()) |err| {
            log.err("missing CUDA symbol {s}: {s}", .{ symbol, std.mem.span(err) });
        } else {
            log.err("missing CUDA symbol {s}", .{symbol});
        }
        return error.CudaSymbolMissing;
    };
    return @ptrCast(raw);
}

fn load_symbol_optional(comptime T: type, handle: *anyopaque, comptime symbol: [:0]const u8) ?T {
    _ = dlerror();
    const raw = dlsym(handle, symbol.ptr) orelse {
        _ = dlerror();
        return null;
    };
    return @ptrCast(raw);
}
