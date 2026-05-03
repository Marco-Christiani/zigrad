//! NVRTC C API runtime bindings.

const std = @import("std");

const log = std.log.scoped(.@"zg/nvrtc_cffi");

const Impl = @import("c-nvrtc");

pub const nvrtcProgram = Impl.nvrtcProgram;
pub const nvrtcResult = Impl.nvrtcResult;
pub const NVRTC_SUCCESS = Impl.NVRTC_SUCCESS;

const FnCreateProgram = *const fn (*nvrtcProgram, ?[*:0]const u8, ?[*:0]const u8, c_int, ?[*]const [*:0]const u8, ?[*]const [*:0]const u8) callconv(.c) nvrtcResult;
const FnDestroyProgram = *const fn (*nvrtcProgram) callconv(.c) nvrtcResult;
const FnCompileProgram = *const fn (nvrtcProgram, c_int, ?[*]const [*:0]const u8) callconv(.c) nvrtcResult;
const FnGetPTXSize = *const fn (nvrtcProgram, *usize) callconv(.c) nvrtcResult;
const FnGetPTX = *const fn (nvrtcProgram, [*]u8) callconv(.c) nvrtcResult;
const FnGetProgramLogSize = *const fn (nvrtcProgram, *usize) callconv(.c) nvrtcResult;
const FnGetProgramLog = *const fn (nvrtcProgram, [*]u8) callconv(.c) nvrtcResult;

var fn_create_program: ?FnCreateProgram = null;
var fn_destroy_program: ?FnDestroyProgram = null;
var fn_compile_program: ?FnCompileProgram = null;
var fn_get_ptx_size: ?FnGetPTXSize = null;
var fn_get_ptx: ?FnGetPTX = null;
var fn_get_program_log_size: ?FnGetProgramLogSize = null;
var fn_get_program_log: ?FnGetProgramLog = null;

var nvrtc_handle: ?*anyopaque = null;
var symbols_ready = false;

pub const LoadError = error{
    NvrtcLibraryNotFound,
    NvrtcSymbolMissing,
};

const RTLD_NOW: c_int = 0x2;
const RTLD_GLOBAL: c_int = 0x100;

extern "c" fn dlopen(filename: [*:0]const u8, flags: c_int) ?*anyopaque;
extern "c" fn dlsym(handle: *anyopaque, symbol: [*:0]const u8) ?*anyopaque;
extern "c" fn dlerror() ?[*:0]const u8;

pub fn ensure_loaded() LoadError!void {
    if (symbols_ready) return;

    if (nvrtc_handle == null) {
        nvrtc_handle = dlopen("libnvrtc.so", RTLD_NOW | RTLD_GLOBAL);
        if (nvrtc_handle == null) {
            nvrtc_handle = dlopen("libnvrtc.so.12", RTLD_NOW | RTLD_GLOBAL);
        }
        if (nvrtc_handle == null) {
            nvrtc_handle = dlopen("libnvrtc.so.11", RTLD_NOW | RTLD_GLOBAL);
        }
        if (nvrtc_handle == null) {
            if (dlerror()) |err| {
                log.err("failed to load NVRTC: {s}", .{std.mem.span(err)});
            }
            return error.NvrtcLibraryNotFound;
        }
    }

    const handle = nvrtc_handle.?;
    fn_create_program = try load_symbol(FnCreateProgram, handle, "nvrtcCreateProgram");
    fn_destroy_program = try load_symbol(FnDestroyProgram, handle, "nvrtcDestroyProgram");
    fn_compile_program = try load_symbol(FnCompileProgram, handle, "nvrtcCompileProgram");
    fn_get_ptx_size = try load_symbol(FnGetPTXSize, handle, "nvrtcGetPTXSize");
    fn_get_ptx = try load_symbol(FnGetPTX, handle, "nvrtcGetPTX");
    fn_get_program_log_size = try load_symbol(FnGetProgramLogSize, handle, "nvrtcGetProgramLogSize");
    fn_get_program_log = try load_symbol(FnGetProgramLog, handle, "nvrtcGetProgramLog");
    symbols_ready = true;
}

pub fn nvrtcCreateProgram(
    prog: *nvrtcProgram,
    src: ?[*:0]const u8,
    name: ?[*:0]const u8,
    num_headers: c_int,
    headers: ?[*]const [*:0]const u8,
    include_names: ?[*]const [*:0]const u8,
) nvrtcResult {
    const f = fn_create_program orelse return 1;
    return f(prog, src, name, num_headers, headers, include_names);
}

pub fn nvrtcDestroyProgram(prog: *nvrtcProgram) nvrtcResult {
    const f = fn_destroy_program orelse return 1;
    return f(prog);
}

pub fn nvrtcCompileProgram(prog: nvrtcProgram, num_options: c_int, options: ?[*]const [*:0]const u8) nvrtcResult {
    const f = fn_compile_program orelse return 1;
    return f(prog, num_options, options);
}

pub fn nvrtcGetPTXSize(prog: nvrtcProgram, ptx_size_ret: *usize) nvrtcResult {
    const f = fn_get_ptx_size orelse return 1;
    return f(prog, ptx_size_ret);
}

pub fn nvrtcGetPTX(prog: nvrtcProgram, ptx: [*]u8) nvrtcResult {
    const f = fn_get_ptx orelse return 1;
    return f(prog, ptx);
}

pub fn nvrtcGetProgramLogSize(prog: nvrtcProgram, log_size_ret: *usize) nvrtcResult {
    const f = fn_get_program_log_size orelse return 1;
    return f(prog, log_size_ret);
}

pub fn nvrtcGetProgramLog(prog: nvrtcProgram, output_log: [*]u8) nvrtcResult {
    const f = fn_get_program_log orelse return 1;
    return f(prog, output_log);
}

fn load_symbol(comptime T: type, handle: *anyopaque, comptime symbol: [:0]const u8) LoadError!T {
    const raw = dlsym(handle, symbol.ptr) orelse {
        if (dlerror()) |err| {
            log.err("missing NVRTC symbol {s}: {s}", .{ symbol, std.mem.span(err) });
        } else {
            log.err("missing NVRTC symbol {s}", .{symbol});
        }
        return error.NvrtcSymbolMissing;
    };
    return @ptrCast(raw);
}
