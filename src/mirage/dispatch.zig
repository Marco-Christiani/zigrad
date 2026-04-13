const std = @import("std");
const kernel = @import("../kernel.zig");
const mirage = @import("../c/mirage/api.zig");
const nvrtc = @import("../c/nvrtc.zig");
const cuda = @import("../c/cuda_driver.zig");
const artifact_mod = @import("artifact.zig");
const TypedPtr = @import("../utils/rtti.zig").TypedPtr;

const log = std.log.scoped(.@"zg/mirage_dispatch");

/// A compiled kernel module: PTX loaded into a CUmodule with cached
/// CUfunction handles for each custom kernel.
const CompiledModule = struct {
    cu_module: cuda.CUmodule,
    /// One entry per kernel in the artifact, same order.
    funcs: []cuda.CUfunction,

    fn deinit(self: *CompiledModule, allocator: std.mem.Allocator) void {
        _ = cuda.cuModuleUnload(self.cu_module);
        allocator.free(self.funcs);
    }
};

pub const MirageDispatchState = struct {
    allocator: std.mem.Allocator,

    /// Cache of compiled modules keyed by a FNV hash of the artifact data.
    cache: std.AutoHashMap(u64, CompiledModule),
    cache_mutex: std.Thread.Mutex = .{},

    pub fn init(allocator: std.mem.Allocator) mirage.MirageError!MirageDispatchState {
        return .{
            .allocator = allocator,
            .cache = std.AutoHashMap(u64, CompiledModule).init(allocator),
        };
    }

    pub fn deinit(self: *MirageDispatchState) void {
        var it = self.cache.valueIterator();
        while (it.next()) |mod| {
            mod.deinit(self.allocator);
        }
        self.cache.deinit();
    }

    pub fn dispatch(
        provider_ctx: TypedPtr,
        artifact_data: []const u8,
        kernel_key: []const u8,
        ctx: kernel.DispatchContext,
    ) kernel.DispatchError!void {
        const self = provider_ctx.cast(MirageDispatchState);
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
        // Decode the artifact.
        const art = artifact_mod.decode(self.allocator, artifact_data) catch {
            log.err("failed to decode mirage artifact for '{s}'", .{kernel_key});
            return error.DispatchFailed;
        };
        defer self.allocator.free(art.kernels);
        defer {
            for (art.kernels) |k| self.allocator.free(k.args);
        }

        if (art.kernels.len == 0) {
            log.err("mirage artifact for '{s}' has no kernels", .{kernel_key});
            return error.DispatchFailed;
        }

        // Get or compile the module.
        const hash = std.hash.Fnv1a_64.hash(artifact_data);
        const compiled = try self.get_or_compile(hash, art);

        // Launch each kernel.
        for (art.kernels, 0..) |kd, ki| {
            try self.launch_kernel(compiled.funcs[ki], kd, ctx);
        }
    }

    fn get_or_compile(self: *MirageDispatchState, hash: u64, art: artifact_mod.Artifact) kernel.DispatchError!CompiledModule {
        self.cache_mutex.lock();
        defer self.cache_mutex.unlock();

        if (self.cache.get(hash)) |mod| {
            return mod;
        }

        // Compile: source -> PTX -> CUmodule.
        const ptx = compile_to_ptx(self.allocator, art.source, "sm_86") catch {
            return error.DispatchFailed;
        };
        defer self.allocator.free(ptx);

        cuda.ensure_loaded() catch {
            log.err("CUDA driver not available", .{});
            return error.DispatchFailed;
        };

        var cu_module: cuda.CUmodule = undefined;
        var rc = cuda.cuModuleLoadData(&cu_module, ptx.ptr);
        if (rc != cuda.CUDA_SUCCESS) {
            log.err("cuModuleLoadData failed: {d}", .{rc});
            return error.DispatchFailed;
        }

        // Extract function handles for each kernel.
        const funcs = self.allocator.alloc(cuda.CUfunction, art.kernels.len) catch {
            _ = cuda.cuModuleUnload(cu_module);
            return error.OutOfMemory;
        };

        for (art.kernels, 0..) |kd, ki| {
            const name_z = self.allocator.dupeZ(u8, kd.func_name) catch {
                _ = cuda.cuModuleUnload(cu_module);
                self.allocator.free(funcs);
                return error.OutOfMemory;
            };
            defer self.allocator.free(name_z);

            rc = cuda.cuModuleGetFunction(&funcs[ki], cu_module, name_z.ptr);
            if (rc != cuda.CUDA_SUCCESS) {
                log.err("cuModuleGetFunction('{s}') failed: {d}", .{ kd.func_name, rc });
                _ = cuda.cuModuleUnload(cu_module);
                self.allocator.free(funcs);
                return error.DispatchFailed;
            }

            // Set max dynamic shared memory if needed.
            if (kd.smem_bytes > 48 * 1024) {
                _ = cuda.cuFuncSetAttribute(
                    funcs[ki],
                    cuda.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    @intCast(kd.smem_bytes),
                );
            }
        }

        const compiled = CompiledModule{ .cu_module = cu_module, .funcs = funcs };
        self.cache.put(hash, compiled) catch {
            // Cache failure is non-fatal -- just won't cache.
        };
        return compiled;
    }

    fn launch_kernel(
        self: *MirageDispatchState,
        func: cuda.CUfunction,
        kd: artifact_mod.KernelDesc,
        ctx: kernel.DispatchContext,
    ) kernel.DispatchError!void {
        // Build kernel argument pointers from the arg mapping.
        // cuLaunchKernel takes void** kernel_params where each element
        // is a pointer to the argument value (which is itself a device pointer).
        const arg_ptrs = self.allocator.alloc(?*anyopaque, kd.args.len) catch
            return error.OutOfMemory;
        defer self.allocator.free(arg_ptrs);

        // We need stable storage for the pointer values themselves.
        const ptr_values = self.allocator.alloc(*anyopaque, kd.args.len) catch
            return error.OutOfMemory;
        defer self.allocator.free(ptr_values);

        for (kd.args, 0..) |arg, i| {
            switch (arg.source) {
                .input => {
                    if (arg.index_or_offset >= ctx.inputs.len) {
                        log.err("kernel arg input index {d} out of range (have {d} inputs)", .{ arg.index_or_offset, ctx.inputs.len });
                        return error.DispatchFailed;
                    }
                    ptr_values[i] = ctx.inputs[arg.index_or_offset].data;
                },
                .output => {
                    if (arg.index_or_offset >= ctx.outputs.len) {
                        log.err("kernel arg output index {d} out of range (have {d} outputs)", .{ arg.index_or_offset, ctx.outputs.len });
                        return error.DispatchFailed;
                    }
                    ptr_values[i] = ctx.outputs[arg.index_or_offset].data;
                },
                .buf => {
                    if (ctx.workspace == null) {
                        log.err("kernel arg requires workspace buffer but none provided", .{});
                        return error.WorkspaceUnavailable;
                    }
                    const base: [*]u8 = @ptrCast(ctx.workspace.?);
                    ptr_values[i] = @ptrCast(base + arg.index_or_offset);
                },
            }
            // cuLaunchKernel wants &ptr_values[i] for each argument.
            arg_ptrs[i] = @ptrCast(&ptr_values[i]);
        }

        const stream: ?cuda.CUstream = if (ctx.stream) |s| @ptrCast(s) else null;

        const rc = cuda.cuLaunchKernel(
            func,
            kd.grid_dim[0],
            kd.grid_dim[1],
            kd.grid_dim[2],
            kd.block_dim[0],
            kd.block_dim[1],
            kd.block_dim[2],
            kd.smem_bytes,
            stream,
            arg_ptrs.ptr,
            null,
        );
        if (rc != cuda.CUDA_SUCCESS) {
            log.err("cuLaunchKernel failed: {d}", .{rc});
            return error.DispatchFailed;
        }
    }
};

/// Compile CUDA source to PTX using NVRTC.
pub fn compile_to_ptx(
    allocator: std.mem.Allocator,
    source: []const u8,
    arch: []const u8,
) ![]const u8 {
    try nvrtc.ensure_loaded();

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const tmp = arena.allocator();

    var options = std.ArrayList([*:0]const u8){};

    const sdk_root = std.posix.getenv("ZG_EXTERNAL_SDK_ROOT") orelse "./result";
    const cuda_home = std.posix.getenv("CUDA_HOME") orelse std.posix.getenv("CUDA_PATH") orelse "/usr/local/cuda";

    for ([_][]const u8{
        try std.fmt.allocPrint(tmp, "--include-path={s}/include", .{sdk_root}),
        try std.fmt.allocPrint(tmp, "--include-path={s}/include/mirage/transpiler/runtime", .{sdk_root}),
        try std.fmt.allocPrint(tmp, "--include-path={s}/include", .{cuda_home}),
    }) |opt| {
        try options.append(tmp, (try tmp.dupeZ(u8, opt)).ptr);
    }

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

    // TODO: proper init or .empty-style pattern
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
/// Strips host-only code and replaces #include "runtime.h" with
/// individual device-compatible includes.
pub fn filter_source_for_nvrtc(allocator: std.mem.Allocator, source: []const u8) ![]const u8 {
    var out = std.ArrayList(u8){};

    try out.appendSlice(allocator, "#define USE_NVSHMEM 0\n#define NUM_GPUS 1\n\n");

    var in_host_function = false;
    var brace_depth: i32 = 0;

    var lines = std.mem.splitSequence(u8, source, "\n");
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");

        if (std.mem.startsWith(u8, trimmed, "#include <vector>") or
            std.mem.startsWith(u8, trimmed, "#include <cuda_runtime") or
            std.mem.startsWith(u8, trimmed, "#include <cublas"))
        {
            continue;
        }

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
