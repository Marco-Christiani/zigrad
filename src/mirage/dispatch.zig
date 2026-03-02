const std = @import("std");
const kernel = @import("../kernel.zig");
const mirage_api = @import("../c/mirage/api.zig");
const mirage_c = @import("../c/mirage/c.zig");
const nvrtc = @import("../c/nvrtc.zig");
const cuda = @import("../c/cuda_driver.zig");

const log = std.log.scoped(.@"zg/mirage_dispatch");

/// A compiled kernel module: PTX loaded into a CUmodule with cached
/// CUfunction handles for each custom kernel.
const CompiledModule = struct {
    cu_module: cuda.CUmodule,
    kernels: []CachedKernel,
    allocator: std.mem.Allocator,

    const CachedKernel = struct {
        func: cuda.CUfunction,
        grid_dim: [3]u32,
        block_dim: [3]u32,
        smem_bytes: u32,
        func_name: []const u8,
    };

    fn deinit(self: *CompiledModule) void {
        _ = cuda.cuModuleUnload(self.cu_module);
        self.allocator.free(self.kernels);
    }
};

pub const MirageDispatchState = struct {
    allocator: std.mem.Allocator,

    /// Cache of compiled modules keyed by a hash of the source code.
    /// Avoids recompiling the same source on repeated dispatch.
    cache: std.StringHashMap(CompiledModule),
    cache_mutex: std.Thread.Mutex = .{},

    pub fn init(allocator: std.mem.Allocator) mirage_api.MirageError!MirageDispatchState {
        return .{
            .allocator = allocator,
            .cache = std.StringHashMap(CompiledModule).init(allocator),
        };
    }

    pub fn deinit(self: *MirageDispatchState) void {
        var it = self.cache.valueIterator();
        while (it.next()) |mod| {
            mod.deinit();
        }
        self.cache.deinit();
    }

    pub fn dispatch(
        provider_ctx: *anyopaque,
        artifact_data: []const u8,
        kernel_key: []const u8,
        ctx: kernel.DispatchContext,
    ) kernel.DispatchError!void {
        const self: *MirageDispatchState = @ptrCast(@alignCast(provider_ctx));
        self.dispatch_impl(artifact_data, kernel_key, ctx) catch |err| {
            log.err("mirage dispatch failed for '{s}': {s}", .{ kernel_key, @errorName(err) });
            return err;
        };
    }

    fn dispatch_impl(
        self: *MirageDispatchState,
        artifact_data: []const u8,
        kernel_key: []const u8,
        ctx: kernel.DispatchContext,
    ) kernel.DispatchError!void {
        _ = kernel_key;

        // The artifact data is CUDA source code from mirage_transpile().
        // We need to: compile to PTX → load module → launch kernels.
        //
        // For now, this is a stub until the NVRTC pipeline is wired up
        // with the correct kernel argument layout from Layer 2/3 metadata.
        //
        // The missing piece is mapping DispatchContext (input/output buffers
        // + workspace) to the kernel argument pointers that _execute_mugraph
        // would compute (output_ptrs..., input_ptrs..., with buf offsets
        // for intermediates).
        _ = self;
        _ = artifact_data;
        _ = ctx;

        log.err("mirage NVRTC dispatch not yet implemented: kernel argument layout mapping pending", .{});
        return error.MirageInternalError;
    }
};

/// Compile CUDA source to PTX using NVRTC.
/// The source should be the filtered device-only code from mirage_transpile().
pub fn compile_to_ptx(
    allocator: std.mem.Allocator,
    source: []const u8,
    arch: []const u8,
) ![]const u8 {
    try nvrtc.ensure_loaded();

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const tmp = arena.allocator();

    // Build NVRTC options
    var options = std.ArrayList([*:0]const u8){};

    const sdk_root = std.posix.getenv("ZG_EXTERNAL_SDK_ROOT") orelse "./result";
    const cuda_home = std.posix.getenv("CUDA_HOME") orelse std.posix.getenv("CUDA_PATH") orelse "/usr/local/cuda";

    // Include paths: SDK (mirage + cutlass + cute), CUDA, runtime headers
    for ([_][]const u8{
        try std.fmt.allocPrint(tmp, "--include-path={s}/include", .{sdk_root}),
        try std.fmt.allocPrint(tmp, "--include-path={s}/include/mirage/transpiler/runtime", .{sdk_root}),
        try std.fmt.allocPrint(tmp, "--include-path={s}/include", .{cuda_home}),
    }) |opt| {
        try options.append(tmp, (try tmp.dupeZ(u8, opt)).ptr);
    }

    // Nix-specific include paths
    if (std.posix.getenv("NIX_GLIBC_INCLUDE")) |g| {
        try options.append(tmp, (try tmp.dupeZ(u8, try std.fmt.allocPrint(tmp, "--include-path={s}", .{g}))).ptr);
    }
    if (std.posix.getenv("NIX_GCC_INCLUDE")) |g| {
        try options.append(tmp, (try tmp.dupeZ(u8, try std.fmt.allocPrint(tmp, "--include-path={s}", .{g}))).ptr);
    }

    try options.append(tmp, (try tmp.dupeZ(u8, try std.fmt.allocPrint(tmp, "--gpu-architecture={s}", .{arch}))).ptr);
    try options.append(tmp, "--std=c++17");
    try options.append(tmp, "-default-device");
    try options.append(tmp, "-DMIRAGE_BACKEND_USE_CUDA");

    // Create and compile program
    var prog: nvrtc.nvrtcProgram = std.mem.zeroes(nvrtc.nvrtcProgram);
    const source_z = try tmp.dupeZ(u8, source);
    const create_rc = nvrtc.nvrtcCreateProgram(&prog, source_z.ptr, "mirage_kernel.cu", 0, null, null);
    if (create_rc != nvrtc.NVRTC_SUCCESS) {
        log.err("nvrtcCreateProgram failed: {d}", .{create_rc});
        return error.NvrtcCompileFailed;
    }
    defer _ = nvrtc.nvrtcDestroyProgram(&prog);

    const compile_rc = nvrtc.nvrtcCompileProgram(
        prog,
        @intCast(options.items.len),
        if (options.items.len == 0) null else @ptrCast(options.items.ptr),
    );

    // Get compilation log
    var log_size: usize = 0;
    _ = nvrtc.nvrtcGetProgramLogSize(prog, &log_size);
    if (log_size > 1) {
        const compile_log = try tmp.alloc(u8, log_size);
        _ = nvrtc.nvrtcGetProgramLog(prog, compile_log.ptr);
        const log_str = std.mem.trimRight(u8, compile_log[0 .. log_size - 1], "\x00");
        if (log_str.len > 0) {
            log.debug("NVRTC log ({d} bytes): {s}", .{ log_str.len, log_str[0..@min(log_str.len, 500)] });
        }
    }

    if (compile_rc != nvrtc.NVRTC_SUCCESS) {
        log.err("NVRTC compilation failed: {d}", .{compile_rc});
        return error.NvrtcCompileFailed;
    }

    // Extract PTX
    var ptx_size: usize = 0;
    if (nvrtc.nvrtcGetPTXSize(prog, &ptx_size) != nvrtc.NVRTC_SUCCESS) {
        return error.NvrtcCompileFailed;
    }

    const ptx = try allocator.alloc(u8, ptx_size);
    if (nvrtc.nvrtcGetPTX(prog, ptx.ptr) != nvrtc.NVRTC_SUCCESS) {
        allocator.free(ptx);
        return error.NvrtcCompileFailed;
    }

    log.info("NVRTC compilation successful ({d} bytes PTX)", .{ptx_size});
    return ptx;
}

/// Filter Mirage transpiler output for NVRTC compilation.
/// Strips host-only code (_init, _execute_mugraph, execute_mugraph wrappers)
/// and replaces #include "runtime.h" with individual device-compatible includes.
pub fn filter_source_for_nvrtc(allocator: std.mem.Allocator, source: []const u8) ![]const u8 {
    var out = std.ArrayList(u8){};

    // Transpiler-defined config macros (normally emitted per-graph).
    try out.appendSlice(allocator, "#define USE_NVSHMEM 0\n#define NUM_GPUS 1\n\n");

    var in_host_function = false;
    var brace_depth: i32 = 0;

    var lines = std.mem.splitSequence(u8, source, "\n");
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");

        // Skip host-only includes
        if (std.mem.startsWith(u8, trimmed, "#include <vector>") or
            std.mem.startsWith(u8, trimmed, "#include <cuda_runtime") or
            std.mem.startsWith(u8, trimmed, "#include <cublas"))
        {
            continue;
        }

        // Replace runtime.h with individual device-compatible includes
        if (std.mem.eql(u8, trimmed, "#include \"runtime.h\"")) {
            try out.appendSlice(allocator, "#include <cute/layout.hpp>\n");
            try out.appendSlice(allocator, "#include <cute/tensor.hpp>\n");
            try out.appendSlice(allocator, "#include <cutlass/cutlass.h>\n");
            try out.appendSlice(allocator, "#include \"config.h\"\n");
            try out.appendSlice(allocator, "#include \"kernel/element_unary.h\"\n");
            try out.appendSlice(allocator, "#include \"kernel/element_binary.h\"\n");
            try out.appendSlice(allocator, "#include \"kernel/reduction.h\"\n");
            try out.appendSlice(allocator, "#include \"threadblock/threadblock.h\"\n");
            try out.appendSlice(allocator, "#include \"utils.h\"\n");
            continue;
        }

        // Detect start of host functions and skip them
        if (!in_host_function) {
            if (std.mem.startsWith(u8, trimmed, "static void _init()") or
                std.mem.startsWith(u8, trimmed, "static void _execute_mugraph(") or
                std.mem.startsWith(u8, trimmed, "extern \"C\" void execute_mugraph"))
            {
                in_host_function = true;
                brace_depth = 0;
                for (line) |ch| {
                    if (ch == '{') brace_depth += 1;
                    if (ch == '}') brace_depth -= 1;
                }
                if (brace_depth <= 0 and std.mem.indexOfScalar(u8, line, '{') != null) {
                    in_host_function = false;
                }
                continue;
            }
        }

        if (in_host_function) {
            for (line) |ch| {
                if (ch == '{') brace_depth += 1;
                if (ch == '}') brace_depth -= 1;
            }
            if (brace_depth <= 0) {
                in_host_function = false;
            }
            continue;
        }

        try out.appendSlice(allocator, line);
        try out.append(allocator, '\n');
    }

    return out.items;
}

fn map_mirage_api_error(err: mirage_api.MirageError) kernel.DispatchError {
    return switch (err) {
        error.MirageUnavailable => error.MirageLoadFailed,
        error.MirageInvalidArgument => error.MirageInvalidArgument,
        error.MirageInternalError => error.MirageInternalError,
        error.MirageApiUnsupported => error.MirageApiUnsupported,
        error.MirageNotFound => error.MirageInternalError,
        error.OutOfMemory => error.OutOfMemory,
    };
}

fn map_mirage_status(status: mirage_c.MirageStatus) kernel.DispatchError {
    if (status == mirage_c.status_invalid_argument) return error.MirageInvalidArgument;
    if (status == mirage_c.status_internal_error) return error.MirageInternalError;
    if (status == mirage_c.status_unsupported) return error.MirageApiUnsupported;
    if (status == mirage_c.status_not_found) return error.MirageInternalError;
    return error.MirageInternalError;
}

test "map_mirage_status preserves runtime detail" {
    try std.testing.expectEqual(error.MirageInvalidArgument, map_mirage_status(mirage_c.status_invalid_argument));
    try std.testing.expectEqual(error.MirageInternalError, map_mirage_status(mirage_c.status_internal_error));
    try std.testing.expectEqual(error.MirageApiUnsupported, map_mirage_status(mirage_c.status_unsupported));
}
