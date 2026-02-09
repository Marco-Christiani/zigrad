//! NVRTC compilation callback for TVM
//!
//! Registers tvm_callback_cuda_compile to provide custom NVRTC compilation with:
//! - Nix-aware include path logic (CUDA, glibc, gcc builtins)
//! - Filtering of problematic includes (#include <cuda.h>, #include <cstdint>)
//!    that TVM generates but NVRTC can't compile in JIT mode

const std = @import("std");
const c = @import("../ffi/tvm.zig");
const ffi = @import("../ffi/tvm.zig");
const nvrtc = @import("../ffi/nvrtc.zig");

const log = std.log.scoped(.@"zg/nvrtc_callback");

/// Register the NVRTC compilation callback with TVM.
/// This must be called during initialization, before any CUDA compilation.
pub fn register(allocator: std.mem.Allocator) !void {
    _ = allocator;

    // create a tvm function that wraps our callback
    var func: c.TVMFFIObjectHandle = undefined; // NOTE: audit use of undefined

    // cast to the c fn pointer type tvm expects
    // TVMFFISafeCallType signature: int (*)(void*, const TVMFFIAny*, int32_t, TVMFFIAny*)
    const callback_fn: *const fn (?*anyopaque, [*c]const c.TVMFFIAny, i32, [*c]c.TVMFFIAny) callconv(.c) c_int = &nvrtcCompileCallback;

    const ret = c.TVMFFIFunctionCreate(
        null, // no closure context needed (self)
        @ptrCast(callback_fn), // the actual function pointer (safe_call)
        null, // no destructor
        &func, // output handle
    );

    if (ret != 0) {
        log.err("Failed to create NVRTC callback function", .{});
        return error.TvmFfiError;
    }

    // register it globally as "tvm_callback_cuda_compile"
    const name = "tvm_callback_cuda_compile";
    var name_arr: c.TVMFFIByteArray = .{ .data = name.ptr, .size = name.len };
    const ret2 = c.TVMFFIFunctionSetGlobal(&name_arr, func, 0);
    if (ret2 != 0) {
        log.err("Failed to register tvm_callback_cuda_compile", .{});
        return error.TvmFfiError;
    }

    log.info("Registered tvm_callback_cuda_compile callback", .{});

    // verify we can retrieve the callback
    const test_name = "tvm_callback_cuda_compile";
    var test_name_arr: c.TVMFFIByteArray = .{ .data = test_name.ptr, .size = test_name.len };
    var test_func: c.TVMFFIObjectHandle = null;
    const ret3 = c.TVMFFIFunctionGetGlobal(&test_name_arr, &test_func);
    if (ret3 != 0 or test_func == null) {
        log.err("Failed to verify callback registration", .{});
        return error.TvmFfiError;
    }
    log.info("Verified: callback is retrievable", .{});
}

/// Callback function called by TVM when compiling CUDA code.
///
/// Arguments: [code: string, target: Target] // NOTE: this doesnt match the zig fn signature is it referring to something else?
/// Returns: string (compiled PTX) NOTE: this returns a c_int not a string
fn nvrtcCompileCallback(
    handle: ?*anyopaque,
    args: [*c]const c.TVMFFIAny,
    num_args: i32,
    ret: [*c]c.TVMFFIAny,
) callconv(.c) c_int {
    _ = handle;
    log.info("nvrtcCompileCallback CALLED with {d} args", .{num_args});

    if (num_args != 2) {
        log.err("nvrtcCompileCallback: expected 2 args, got {d}", .{num_args});
        return -1;
    }

    // extract args
    const code_arg = args[0];
    const target_arg = args[1];

    // get cuda source code
    const code_ptr = code_arg.unnamed_1.v_c_str;
    if (code_ptr == null) {
        log.err("nvrtcCompileCallback: code string is null", .{});
        return -1;
    }
    const original_code = std.mem.span(code_ptr.?);

    const preview_len = @min(original_code.len, 500);
    log.debug("CUDA code preview ({d} bytes total):\n{s}...", .{ original_code.len, original_code[0..preview_len] });

    // Get target and extract arch from it
    // TODO: For now, use a default architecture, but we need to actually extract arch from target
    _ = target_arg;
    const arch = "sm_86";

    // thread-local arena for this compilation
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // strip problematic includes nvrtc cant handle
    // tvm generates #include <cuda.h> and #include <cstdint>, but nvrtc doesnt need them
    // (cuda.h brings in stdlib.h which causes issues, cstdint is C++ STL)
    var filtered_code = std.ArrayList(u8){};
    defer filtered_code.deinit(allocator);

    var lines = std.mem.splitSequence(u8, original_code, "\n");
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        // skip lines that include cuda.h or cstdint
        if (std.mem.eql(u8, trimmed, "#include <cuda.h>") or
            std.mem.eql(u8, trimmed, "#include <cstdint>")) {
            log.debug("Stripped: {s}", .{trimmed});
            continue;
        }
        filtered_code.appendSlice(allocator, line) catch @panic("OOM");
        filtered_code.append(allocator, '\n') catch @panic("OOM");
    }
    const patched_code = filtered_code.items;

    const patched_preview_len = @min(patched_code.len, 300);
    log.debug("Filtered code preview ({d} bytes total):\n{s}...", .{ patched_code.len, patched_code[0..patched_preview_len] });

    // compile
    const ptx = compileWithNvrtc(allocator, patched_code, arch) catch |err| {
        log.err("NVRTC compilation failed: {s}", .{@errorName(err)});
        return -1;
    };

    // return ptx as a tvm String object
    // NOTE: using kTVMFFIRawStr would return a borrowed pointer that tvm doesnt own, which can
    //  cause heap corruption when tvm tries to manage the memory.
    var ptx_bytes: c.TVMFFIByteArray = .{ .data = ptx.ptr, .size = ptx.len };
    var str_obj: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    if (c.TVMFFIStringFromByteArray(&ptx_bytes, &str_obj) != 0) {
        log.err("Failed to create TVM string from PTX", .{});
        return -1;
    }
    ret.* = str_obj;
    return 0;
}

/// Compile CUDA source code to PTX using NVRTC injecting the right include paths.
fn compileWithNvrtc(
    allocator: std.mem.Allocator,
    code: []const u8,
    arch: []const u8,
) ![]const u8 {
    const cuda_path = std.posix.getenv("CUDA_HOME") orelse std.posix.getenv("CUDA_PATH") orelse {
        log.err("CUDA_HOME or CUDA_PATH must be set", .{});
        return error.CudaPathNotSet;
    };

    // build nvrtc compile options
    var options: std.ArrayList([]const u8) = .{};
    defer options.deinit(allocator);

    // cuda cpp stdlib headers
    const cuda_libcxx_path = try std.fmt.allocPrint(
        allocator,
        "--include-path={s}/include/cuda/std/detail/libcxx/include",
        .{cuda_path},
    );
    try options.append(allocator, cuda_libcxx_path);

    // cuda main include dir
    const cuda_include_path = try std.fmt.allocPrint(
        allocator,
        "--include-path={s}/include",
        .{cuda_path},
    );
    try options.append(allocator, cuda_include_path);

    // glibc C headers - get from NIX_GLIBC_INCLUDE or use a common path NOTE: missing common path suggested by comment?
    if (std.posix.getenv("NIX_GLIBC_INCLUDE")) |glibc_include| {
        const glibc_path = try std.fmt.allocPrint(
            allocator,
            "--include-path={s}",
            .{glibc_include},
        );
        try options.append(allocator, glibc_path);
    }

    // GCC builtin headers - get from NIX_GCC_INCLUDE or use a common path NOTE: missing common path suggested by comment?
    if (std.posix.getenv("NIX_GCC_INCLUDE")) |gcc_include| {
        const gcc_path = try std.fmt.allocPrint(
            allocator,
            "--include-path={s}",
            .{gcc_include},
        );
        try options.append(allocator, gcc_path);
    }

    // arch and defines
    const arch_flag = try std.fmt.allocPrint(allocator, "--gpu-architecture={s}", .{arch});
    try options.append(allocator, arch_flag);
    try options.append(allocator, "--std=c++17"); // NOTE: can we assume this?
    try options.append(allocator, "-D__x86_64__"); // NOTE: can we assume this?
    try options.append(allocator, "-default-device");  // required for jit mode NOTE: still needed? documentation required

    // convert to c strings
    const c_options = try allocator.alloc([*c]const u8, options.items.len);
    for (options.items, 0..) |opt, i| {
        c_options[i] = (try allocator.dupeZ(u8, opt)).ptr;
    }

    log.debug("Compiling CUDA code with NVRTC ({d} options)", .{c_options.len});
    for (c_options) |opt| {
        log.debug("  {s}", .{opt});
    }

    // create nvrtc program
    var prog: nvrtc.nvrtcProgram = undefined; // NOTE: audit use of undefined
    const code_z = try allocator.dupeZ(u8, code);
    const create_result = nvrtc.nvrtcCreateProgram(
        &prog,
        code_z.ptr,
        "default_program",
        0,
        null,
        null,
    );

    if (create_result != nvrtc.NVRTC_SUCCESS) {
        log.err("nvrtcCreateProgram failed: {d}", .{create_result});
        return error.NvrtcCreateProgramFailed;
    }
    defer _ = nvrtc.nvrtcDestroyProgram(&prog);

    // compile
    const compile_result = nvrtc.nvrtcCompileProgram(
        prog,
        @intCast(c_options.len),
        @ptrCast(c_options.ptr),
    );

    // check compilation log
    var log_size: usize = 0;
    _ = nvrtc.nvrtcGetProgramLogSize(prog, &log_size);
    if (log_size > 1) {
        const compile_log = try allocator.alloc(u8, log_size);
        _ = nvrtc.nvrtcGetProgramLog(prog, compile_log.ptr);
        log.debug("NVRTC log:\n{s}", .{compile_log});
    }

    if (compile_result != nvrtc.NVRTC_SUCCESS) {
        log.err("nvrtcCompileProgram failed: {d}", .{compile_result});
        return error.NvrtcCompileFailed;
    }

    // get PTX
    var ptx_size: usize = 0;
    var ptx_result = nvrtc.nvrtcGetPTXSize(prog, &ptx_size);
    if (ptx_result != nvrtc.NVRTC_SUCCESS) {
        log.err("nvrtcGetPTXSize failed: {d}", .{ptx_result});
        return error.NvrtcGetPtxFailed;
    }

    const ptx = try allocator.alloc(u8, ptx_size);
    ptx_result = nvrtc.nvrtcGetPTX(prog, ptx.ptr);
    if (ptx_result != nvrtc.NVRTC_SUCCESS) {
        log.err("nvrtcGetPTX failed: {d}", .{ptx_result});
        return error.NvrtcGetPtxFailed;
    }

    log.info("NVRTC compilation successful ({d} bytes PTX)", .{ptx_size});
    return ptx[0 .. ptx_size - 1]; // remove null terminator NOTE: do we need to dereference to move to stack mem? wont arena free this?
}
