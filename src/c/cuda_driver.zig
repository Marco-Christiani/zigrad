//! CUDA Driver API bindings for kernel module management and launch.
//!
//! Provides cuModuleLoadData, cuModuleGetFunction, cuLaunchKernel, etc.
//! Functions are loaded at runtime via dlsym since libcuda.so is a driver
//! library that may or may not be present.
const std = @import("std");

const log = std.log.scoped(.@"zg/cuda_driver");

/// Opaque CUDA types (from the driver API).
pub const CUmodule = *anyopaque;
pub const CUfunction = *anyopaque;
pub const CUstream = *anyopaque;
pub const CUresult = c_int;

pub const CUDA_SUCCESS: CUresult = 0;

// ---------------------------------------------------------------------------
// Function pointer types
// ---------------------------------------------------------------------------

const FnModuleLoadData = *const fn (*CUmodule, *const anyopaque) callconv(.c) CUresult;
const FnModuleUnload = *const fn (CUmodule) callconv(.c) CUresult;
const FnModuleGetFunction = *const fn (*CUfunction, CUmodule, [*:0]const u8) callconv(.c) CUresult;
const FnLaunchKernel = *const fn (
    CUfunction,
    c_uint, // gridDimX
    c_uint, // gridDimY
    c_uint, // gridDimZ
    c_uint, // blockDimX
    c_uint, // blockDimY
    c_uint, // blockDimZ
    c_uint, // sharedMemBytes
    ?CUstream, // hStream
    ?[*]?*anyopaque, // kernelParams
    ?[*]?*anyopaque, // extra
) callconv(.c) CUresult;
const FnFuncSetAttribute = *const fn (CUfunction, c_int, c_int) callconv(.c) CUresult;

/// CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
pub const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;

// ---------------------------------------------------------------------------
// Runtime-loaded symbols
// ---------------------------------------------------------------------------

var fn_module_load_data: ?FnModuleLoadData = null;
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

const RTLD_NOW: c_int = 0x2;

extern "c" fn dlopen(filename: [*:0]const u8, flags: c_int) ?*anyopaque;
extern "c" fn dlsym(handle: *anyopaque, symbol: [*:0]const u8) ?*anyopaque;
extern "c" fn dlerror() ?[*:0]const u8;

pub fn ensure_loaded() LoadError!void {
    if (symbols_ready) return;

    if (cuda_handle == null) {
        // libcuda.so is the driver library — always present when a GPU is.
        // Try unversioned first, then common versioned names.
        for ([_][*:0]const u8{ "libcuda.so", "libcuda.so.1" }) |name| {
            cuda_handle = dlopen(name, RTLD_NOW);
            if (cuda_handle != null) break;
        }
        if (cuda_handle == null) {
            if (dlerror()) |err| {
                log.err("failed to load CUDA driver: {s}", .{std.mem.span(err)});
            }
            return error.CudaDriverNotFound;
        }
    }

    const handle = cuda_handle.?;
    fn_module_load_data = try load_symbol(FnModuleLoadData, handle, "cuModuleLoadData");
    fn_module_unload = try load_symbol(FnModuleUnload, handle, "cuModuleUnload");
    fn_module_get_function = try load_symbol(FnModuleGetFunction, handle, "cuModuleGetFunction");
    fn_launch_kernel = try load_symbol(FnLaunchKernel, handle, "cuLaunchKernel");
    fn_func_set_attribute = load_symbol_optional(FnFuncSetAttribute, handle, "cuFuncSetAttribute");
    symbols_ready = true;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn cuModuleLoadData(module: *CUmodule, image: *const anyopaque) CUresult {
    const f = fn_module_load_data orelse return 1;
    return f(module, image);
}

pub fn cuModuleUnload(module: CUmodule) CUresult {
    const f = fn_module_unload orelse return 1;
    return f(module);
}

pub fn cuModuleGetFunction(func: *CUfunction, module: CUmodule, name: [*:0]const u8) CUresult {
    const f = fn_module_get_function orelse return 1;
    return f(func, module, name);
}

pub fn cuLaunchKernel(
    func: CUfunction,
    grid_dim_x: c_uint,
    grid_dim_y: c_uint,
    grid_dim_z: c_uint,
    block_dim_x: c_uint,
    block_dim_y: c_uint,
    block_dim_z: c_uint,
    shared_mem_bytes: c_uint,
    stream: ?CUstream,
    kernel_params: ?[*]?*anyopaque,
    extra: ?[*]?*anyopaque,
) CUresult {
    const f = fn_launch_kernel orelse return 1;
    return f(func, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z, shared_mem_bytes, stream, kernel_params, extra);
}

pub fn cuFuncSetAttribute(func: CUfunction, attrib: c_int, value: c_int) CUresult {
    const f = fn_func_set_attribute orelse return 1;
    return f(func, attrib, value);
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

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
