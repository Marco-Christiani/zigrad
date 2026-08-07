//! NVRTC compilation callback for TVM.
//!
//! Registers `tvm_callback_cuda_compile` and adapts TVM-generated CUDA source
//!  to the shared Zigrad NVRTC compiler.

const std = @import("std");
const builtin = @import("builtin");
const api = @import("../c/tvm/api.zig");
const c = @import("../c/tvm/c.zig");
const cuda_nvrtc = @import("../cuda/nvrtc.zig");

const log = std.log.scoped(.@"zg/tvm_nvrtc");

pub const RegisterError = std.mem.Allocator.Error || error{
    NvrtcLoadFailed,
    TvmFfiError,
};

const CallbackState = struct {
    library_path: []u8,
    toolkit_root: []u8,
    glibc_include_dir: ?[]u8,
    gcc_include_dir: ?[]u8,
    gpu_arch: []u8,

    fn create(input: cuda_nvrtc.Config, gpu_arch: []const u8) std.mem.Allocator.Error!*CallbackState {
        const allocator = std.heap.c_allocator;
        const self = try allocator.create(CallbackState);
        errdefer allocator.destroy(self);

        self.library_path = try allocator.dupe(u8, input.library_path);
        errdefer allocator.free(self.library_path);

        self.toolkit_root = try allocator.dupe(u8, input.toolkit_root);
        errdefer allocator.free(self.toolkit_root);

        self.glibc_include_dir = null;
        self.gcc_include_dir = null;
        self.gpu_arch = undefined;

        if (input.glibc_include_dir) |path| {
            self.glibc_include_dir = try allocator.dupe(u8, path);
        }
        errdefer if (self.glibc_include_dir) |path| allocator.free(path);

        if (input.gcc_include_dir) |path| {
            self.gcc_include_dir = try allocator.dupe(u8, path);
        }
        errdefer if (self.gcc_include_dir) |path| allocator.free(path);

        self.gpu_arch = try allocator.dupe(u8, gpu_arch);
        return self;
    }

    const Snapshot = struct {
        config: cuda_nvrtc.Config,
        gpu_arch: []const u8,
    };

    fn snapshot(self: *const CallbackState) Snapshot {
        return .{
            .config = .{
                .library_path = self.library_path,
                .toolkit_root = self.toolkit_root,
                .glibc_include_dir = self.glibc_include_dir,
                .gcc_include_dir = self.gcc_include_dir,
            },
            .gpu_arch = self.gpu_arch,
        };
    }

    fn destroy(self: *CallbackState) void {
        const allocator = std.heap.c_allocator;
        allocator.free(self.library_path);
        allocator.free(self.toolkit_root);
        if (self.glibc_include_dir) |path| allocator.free(path);
        if (self.gcc_include_dir) |path| allocator.free(path);
        allocator.free(self.gpu_arch);
        allocator.destroy(self);
    }
};

/// Register the NVRTC compilation callback with TVM.
///
/// Registration must precede TVM CUDA compilation.
pub fn register(
    allocator: std.mem.Allocator,
    config: cuda_nvrtc.Config,
    gpu_arch: []const u8,
) RegisterError!void {
    cuda_nvrtc.ensure_available(config) catch {
        log.err("failed to load NVRTC runtime", .{});
        return error.NvrtcLoadFailed;
    };

    const state = try CallbackState.create(config, gpu_arch);
    var state_owned = true;
    errdefer if (state_owned) state.destroy();

    const func_val = api.create_packed_func(
        state,
        &nvrtc_compile_callback,
        &destroy_callback_state,
    ) catch {
        log.err("failed to create NVRTC callback function", .{});
        return error.TvmFfiError;
    };
    state_owned = false;
    defer func_val.decref();
    const func_handle = func_val.as_object() orelse {
        log.err("NVRTC callback function has no object handle", .{});
        return error.TvmFfiError;
    };

    api.set_global("tvm_callback_cuda_compile", func_handle, true) catch {
        log.err("failed to register tvm_callback_cuda_compile", .{});
        return error.TvmFfiError;
    };
    log.info("registered tvm_callback_cuda_compile callback", .{});

    const verify_handle = api.get_global(allocator, "tvm_callback_cuda_compile") catch {
        log.err("failed to verify callback registration", .{});
        return error.TvmFfiError;
    };
    _ = c.TVMFFIObjectDecRef(verify_handle);
    log.info("verified: callback is retrievable", .{});
}

fn destroy_callback_state(handle: ?*anyopaque) callconv(.c) void {
    const state: *CallbackState = @ptrCast(@alignCast(handle orelse return));
    state.destroy();
}

/// Callback function called by TVM when compiling CUDA code.
///
/// TVM supplies the CUDA source and target through its packed-call convention.
/// The callback writes PTX into `ret` and returns zero on success.
fn nvrtc_compile_callback(
    handle: ?*anyopaque,
    args: [*c]const c.TVMFFIAny,
    num_args: i32,
    ret: [*c]c.TVMFFIAny,
) callconv(.c) c_int {
    const state: *const CallbackState = @ptrCast(@alignCast(handle orelse {
        log.err("NVRTC callback has no configuration", .{});
        return -1;
    }));

    if (num_args != 2) {
        log.err("nvrtc_compile_callback: expected 2 args, got {d}", .{num_args});
        return -1;
    }

    const code_arg = args[0];
    const target_arg = args[1];

    const code_ptr = code_arg.unnamed_1.v_c_str;
    if (code_ptr == null) {
        log.err("nvrtc_compile_callback: code string is null", .{});
        return -1;
    }
    const original_code = std.mem.span(code_ptr.?);

    const preview_len = @min(original_code.len, 500);
    log.debug("CUDA code preview ({d} bytes total):\n{s}...", .{ original_code.len, original_code[0..preview_len] });

    // Target selection is resolved before callback registration.
    _ = target_arg;

    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Remove host includes that this NVRTC source path cannot compile.
    //
    // `cuda.h` reaches the host C library, and `cstdint` requires C++ standard
    //  library headers.
    var filtered_code: std.ArrayList(u8) = .empty;
    defer filtered_code.deinit(allocator);

    var lines = std.mem.splitSequence(u8, original_code, "\n");
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (std.mem.eql(u8, trimmed, "#include <cuda.h>") or
            std.mem.eql(u8, trimmed, "#include <cstdint>"))
        {
            log.debug("Stripped: {s}", .{trimmed});
            continue;
        }
        filtered_code.appendSlice(allocator, line) catch {
            log.err("failed to allocate filtered CUDA source", .{});
            return -1;
        };
        filtered_code.append(allocator, '\n') catch {
            log.err("failed to allocate filtered CUDA source", .{});
            return -1;
        };
    }
    const patched_code = filtered_code.items;

    const patched_preview_len = @min(patched_code.len, 300);
    log.debug("Filtered code preview ({d} bytes total):\n{s}...", .{ patched_code.len, patched_code[0..patched_preview_len] });

    const ptx = compile_with_nvrtc(
        allocator,
        patched_code,
        state.snapshot(),
    ) catch |err| {
        log.err("NVRTC compilation failed: {s}", .{@errorName(err)});
        return -1;
    };

    // The TVM string takes responsibility for the returned PTX bytes.
    const str_val = api.make_tvm_string(ptx) catch {
        log.err("failed to create TVM string from PTX", .{});
        return -1;
    };
    ret.* = str_val.raw;
    return 0;
}

/// Compile CUDA source to PTX with the configured include paths.
///
/// The caller frees the returned string.
fn compile_with_nvrtc(
    allocator: std.mem.Allocator,
    code: []const u8,
    snapshot: CallbackState.Snapshot,
) cuda_nvrtc.CompileError![]const u8 {
    const defines: []const []const u8 = switch (builtin.cpu.arch) {
        .x86_64 => &.{"__x86_64__"},
        else => &.{},
    };
    const ptx = try cuda_nvrtc.compile(allocator, code, snapshot.config, .{
        .gpu_arch = snapshot.gpu_arch,
        .program_name = "tvm_kernel.cu",
        .defines = defines,
    });
    if (ptx.len == 0) return error.NvrtcGetPtxFailed;
    return ptx[0 .. ptx.len - 1];
}
