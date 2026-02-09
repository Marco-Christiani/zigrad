const std = @import("std");
const build_options = @import("build_options");
const c = @import("ffi/tvm.zig");

// TVM subsystem modules
const tvm_common = @import("tvm/common.zig");
const tvm_cuda = @import("tvm/cuda.zig");
const tvm_builder = @import("tvm/builder.zig");
const nvrtc_callback = @import("tvm/nvrtc_callback.zig");

/// Handle to the dynamically loaded libtvm.so (full compiler with TE/codegen).
/// This is loaded on-demand since the Zig linker drops it (no direct symbol refs).
/// Also, this is so we can follow a plugin model and leave TVM opt-in so (1) it is not
/// required for the build and (2) do not need to rebuild zigrad to use tvm (this is a
/// possible stretch goal that is not true right now though)
/// TE functions (te.Placeholder, etc.) register themselves via static initializers
/// when the library loads.
/// We use a raw dlopen handle with RTLD_GLOBAL so symbols are available globally
/// (needed for loading compiled TVM modules that depend on TVM runtime symbols).
var tvm_compiler_lib_handle: ?*anyopaque = null;

/// Handle to libtvm_ffi.so loaded with RTLD_GLOBAL.
/// While the Zig linker loads this library, it uses RTLD_LOCAL which doesn't expose
/// symbols globally. We need to reload it with RTLD_GLOBAL so compiled TVM modules
/// can find TVM runtime symbols.
var tvm_ffi_lib_handle: ?*anyopaque = null;

/// Find the directory containing libtvm_ffi.so by reading /proc/self/maps.
/// Returns the path to libtvm.so in the same directory, or null if not found.
fn findTvmLibPath(allocator: std.mem.Allocator) !?[]const u8 {
    const maps_file = std.fs.openFileAbsolute("/proc/self/maps", .{}) catch return null;
    defer maps_file.close();

    var read_buf: [8192]u8 = undefined;  // NOTE: audit use of undefined
    var file_reader = maps_file.reader(&read_buf);
    const reader = &file_reader.interface;

    while (true) {
        const line = reader.takeDelimiter('\n') catch break;
        if (line == null) break;
        const l = line.?;

        // Look for libtvm_ffi.so in the path column
        if (std.mem.indexOf(u8, l, "libtvm_ffi.so")) |_| {
            // Find the path (starts with '/')
            if (std.mem.indexOf(u8, l, "/")) |path_start| {
                const path = l[path_start..];
                // Extract directory from the path
                if (std.mem.lastIndexOf(u8, path, "/")) |slash| {
                    const dir = path[0..slash];
                    const tvm_path = try std.fmt.allocPrint(allocator, "{s}/libtvm.so", .{dir});
                    return tvm_path;
                }
            }
        }
    }
    return null;
}

// dlopen flags for loading TVM libraries with global symbol visibility
const RTLD_NOW: c_int = 0x2;
const RTLD_GLOBAL: c_int = 0x100; // Symbols available globally (can cause conflicts)
extern "c" fn dlopen(filename: [*:0]const u8, flags: c_int) ?*anyopaque;
extern "c" fn dlerror() ?[*:0]const u8;

/// Ensure TVM FFI library is loaded with RTLD_GLOBAL.
/// The Zig linker loads libtvm_ffi.so with RTLD_LOCAL, so we need to reload it
/// with RTLD_GLOBAL for compiled TVM modules to find runtime symbols.
fn ensureTvmFfiLoaded(allocator: std.mem.Allocator) !void {
    if (tvm_ffi_lib_handle != null) return;

    const log = std.log.scoped(.@"zg/tvm_init");
    _ = allocator;

    // First try libtvm_ffi.so by name (should be in LD_LIBRARY_PATH/linker path)
    tvm_ffi_lib_handle = dlopen("libtvm_ffi.so", RTLD_NOW | RTLD_GLOBAL);
    if (tvm_ffi_lib_handle != null) {
        log.info("loaded libtvm_ffi.so with RTLD_GLOBAL", .{});
        return;
    }

    // If that fails, try libtvm_runtime.so (older TVM versions)
    tvm_ffi_lib_handle = dlopen("libtvm_runtime.so", RTLD_NOW | RTLD_GLOBAL);
    if (tvm_ffi_lib_handle != null) {
        log.info("loaded libtvm_runtime.so with RTLD_GLOBAL", .{});
        return;
    }

    // Log failure but don't error - the Zig linker should have loaded it already
    if (dlerror()) |err| {
        log.warn("could not reload TVM FFI with RTLD_GLOBAL: {s}", .{std.mem.span(err)});
    }
}

/// Ensure the full TVM compiler library is loaded. Required for TE API access.
/// Safe to call multiple times - only loads once.
/// Uses RTLD_GLOBAL so TVM symbols are available to compiled modules loaded later.
fn ensureTvmCompilerLoaded(allocator: std.mem.Allocator) !void {
    // First, ensure libtvm_ffi.so is loaded with RTLD_GLOBAL
    // This is needed so compiled TVM modules can find FFI runtime symbols
    try ensureTvmFfiLoaded(allocator);

    if (tvm_compiler_lib_handle != null) return;

    const log = std.log.scoped(.@"zg/tvm_init");

    // First, try to find libtvm.so in the same directory as libtvm_ffi.so
    const lib_path = try findTvmLibPath(allocator);
    defer if (lib_path) |p| allocator.free(p);

    if (lib_path) |path| {
        log.info("found TVM compiler at: {s}", .{path});

        // Allocate null-terminated path for dlopen
        const path_z = try allocator.allocSentinel(u8, path.len, 0);
        defer allocator.free(path_z);
        @memcpy(path_z, path);

        // Load with RTLD_GLOBAL so symbols are available to modules we compile
        tvm_compiler_lib_handle = dlopen(path_z, RTLD_NOW | RTLD_GLOBAL);
        if (tvm_compiler_lib_handle == null) {
            if (dlerror()) |err| {
                log.err("dlopen failed: {s}", .{std.mem.span(err)});
            }
            return error.TvmRuntimeError;
        }
        log.info("loaded libtvm.so (TVM compiler with TE/codegen)", .{});

        // Register NVRTC compilation callback for nix-compatible CUDA compilation
        nvrtc_callback.register(allocator) catch |err| {
            log.warn("Failed to register NVRTC callback: {s}", .{@errorName(err)});
        };

        return;
    }

    // Fallback: try just "libtvm.so" in case LD_LIBRARY_PATH is set
    tvm_compiler_lib_handle = dlopen("libtvm.so", RTLD_NOW | RTLD_GLOBAL);
    if (tvm_compiler_lib_handle == null) {
        if (dlerror()) |err| {
            log.err("dlopen failed: {s}", .{std.mem.span(err)});
        }
        log.err("TE functions (te.Placeholder etc.) will not be available", .{});
        return error.TvmRuntimeError;
    }

    log.info("loaded libtvm.so (TVM compiler with TE/codegen)", .{});
}

/// Track whether CUDA intrinsics have been loaded.
var cuda_intrinsics_loaded: bool = false;

/// Load and register CUDA tensor intrinsics from pre-serialized JSON files.
///
/// This function loads 81 CUDA tensor intrinsics (WMMA, MMA, etc.) that are required
/// for CUDA MetaSchedule auto-tuning. The intrinsics are pre-generated by
/// scripts/generate_cuda_intrinsics.py and stored as JSON files.
///
/// Safe to call multiple times - only loads once.
/// Must be called after ensureTvmCompilerLoaded().
fn loadCudaIntrinsics(allocator: std.mem.Allocator) !void {
    if (cuda_intrinsics_loaded) return;

    const log = std.log.scoped(.@"zg/cuda_intrinsics");

    const intrinsics_dir = "artifacts/cuda_intrinsics";

    // Open the intrinsics directory
    var dir = std.fs.cwd().openDir(intrinsics_dir, .{ .iterate = true }) catch |err| {
        log.err("failed to open {s}: {s}", .{ intrinsics_dir, @errorName(err) });
        log.err("Run: python3 scripts/generate_cuda_intrinsics.py", .{});
        return err;
    };
    defer dir.close();

    log.info("Loading CUDA intrinsics from {s}/", .{intrinsics_dir});

    var loaded_count: usize = 0;
    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".json")) continue;

        // Read JSON file
        const json_data = dir.readFileAlloc(allocator, entry.name, 1_000_000) catch |err| {
            log.warn("failed to read {s}: {s}", .{ entry.name, @errorName(err) });
            continue;
        };
        defer allocator.free(json_data);

        // Parse JSON to extract name, desc, impl
        const parsed = std.json.parseFromSlice(
            struct {
                name: []const u8,
                desc: []const u8,
                impl: []const u8,
            },
            allocator,
            json_data,
            .{},
        ) catch |err| {
            log.warn("failed to parse {s}: {s}", .{ entry.name, @errorName(err) });
            continue;
        };
        defer parsed.deinit();

        const intrinsic_data = parsed.value;

        // Convert strings to C strings
        const name_cstr = try cstr_alloc(allocator, intrinsic_data.name);
        defer allocator.free(name_cstr);

        const desc_cstr = try cstr_alloc(allocator, intrinsic_data.desc);
        defer allocator.free(desc_cstr);

        const impl_cstr = try cstr_alloc(allocator, intrinsic_data.impl);
        defer allocator.free(impl_cstr);

        // Load PrimFuncs from JSON via FFI: node.LoadJSON
        var desc_primfunc: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        try ffi_call_global(allocator, "node.LoadJSON", &.{any_raw_str(cstr_ptr(desc_cstr))}, &desc_primfunc);

        var impl_primfunc: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        try ffi_call_global(allocator, "node.LoadJSON", &.{any_raw_str(cstr_ptr(impl_cstr))}, &impl_primfunc);

        // Create TensorIntrin object via FFI: tir.TensorIntrin(desc, impl)
        var tensor_intrin: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        try ffi_call_global(allocator, "tir.TensorIntrin", &.{ desc_primfunc, impl_primfunc }, &tensor_intrin);

        // Register via FFI: tir.TensorIntrinRegister(name, intrin, override=false)
        var dummy_out: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        try ffi_call_global(allocator, "tir.TensorIntrinRegister", &.{
            any_raw_str(cstr_ptr(name_cstr)),
            tensor_intrin,
            any_bool(false),
        }, &dummy_out);

        loaded_count += 1;
    }

    log.info("Loaded {d} CUDA tensor intrinsics", .{loaded_count});
    cuda_intrinsics_loaded = true;
}

fn ffi_fail(comptime what: []const u8) !noreturn {
    _ = what;
    return error.TvmRuntimeError;
}

pub fn any_none() c.TVMFFIAny {
    var v = std.mem.zeroes(c.TVMFFIAny);
    v.type_index = c.kTVMFFINone;
    return v;
}

pub fn any_int(value: i64) c.TVMFFIAny {
    var v = std.mem.zeroes(c.TVMFFIAny);
    v.type_index = c.kTVMFFIInt;
    v.unnamed_1.v_int64 = value;
    return v;
}

pub fn any_bool(value: bool) c.TVMFFIAny {
    var v = std.mem.zeroes(c.TVMFFIAny);
    v.type_index = c.kTVMFFIBool;
    v.unnamed_1.v_int64 = if (value) 1 else 0;
    return v;
}

pub fn any_raw_str(cstr: [*:0]const u8) c.TVMFFIAny {
    var v = std.mem.zeroes(c.TVMFFIAny);
    v.type_index = c.kTVMFFIRawStr;
    v.unnamed_1.v_c_str = cstr;
    return v;
}

pub fn any_obj(handle: c.TVMFFIObjectHandle, type_index: i32) c.TVMFFIAny {
    var v = std.mem.zeroes(c.TVMFFIAny);
    v.type_index = type_index;
    v.unnamed_1.v_obj = @ptrCast(@alignCast(handle));
    return v;
}

fn any_device(device_type: i32, device_id: i32) c.TVMFFIAny {
    var v = std.mem.zeroes(c.TVMFFIAny);
    v.type_index = c.kTVMFFIDevice;
    v.unnamed_1.v_device = .{ .device_type = @intCast(device_type), .device_id = device_id };
    return v;
}

fn any_dtype_f32() c.TVMFFIAny {
    var v = std.mem.zeroes(c.TVMFFIAny);
    v.type_index = c.kTVMFFIDataType;
    v.unnamed_1.v_dtype = .{ .code = c.kDLFloat, .bits = 32, .lanes = 1 };
    return v;
}

fn any_ptr(ptr: *anyopaque) c.TVMFFIAny {
    var v = std.mem.zeroes(c.TVMFFIAny);
    v.type_index = c.kTVMFFIOpaquePtr;
    // Store raw pointer as int64 (kTVMFFIOpaquePtr uses the int field for void*)
    v.unnamed_1.v_int64 = @bitCast(@intFromPtr(ptr));
    return v;
}

/// Allocate a TVM tensor on the specified device, filled with the provided f32 data.
///
/// For CPU (device_type = kDLCPU): uses DLPack zero-copy wrapping.
/// For GPU (device_type = kDLCUDA): allocates via `runtime.TVMTensorAllocWithScope`,
/// then copies from host via `runtime.TVMTensorCopyFromBytes`.
///
/// Returns a TVM Tensor object handle (caller must DecRef).
fn allocate_tensor(
    allocator: std.mem.Allocator,
    data: []f32,
    shape: []i64,
    device_type: i32,
) !c.TVMFFIObjectHandle {
    if (device_type == c.kDLCPU) {
        // CPU: zero-copy DLPack path
        var dl = c.DLManagedTensor{
            .dl_tensor = make_dl_tensor_f32(data, shape),
            .manager_ctx = null,
            .deleter = dlpack_noop_deleter,
        };
        return tensor_from_dlpack(allocator, &dl);
    }

    // GPU: allocate on device via TVM FFI
    // 1. Create Shape object
    var shape_args: [4]c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    for (shape, 0..) |dim, i| {
        shape_args[i] = any_int(dim);
    }
    var shape_obj: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    try ffi_call_global(allocator, "ffi.Shape", shape_args[0..shape.len], &shape_obj);
    defer if (shape_obj.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    // 2. Allocate empty tensor on device
    var tensor: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    try ffi_call_global(allocator, "runtime.TVMTensorAllocWithScope", &.{
        shape_obj,
        any_dtype_f32(),
        any_device(device_type, 0),
        any_none(), // mem_scope
    }, &tensor);

    // 3. Copy host data to device tensor
    const nbytes = data.len * @sizeOf(f32);
    var copy_out: c.TVMFFIAny = std.mem.zeroes(c.TVMFFIAny);
    try ffi_call_global(allocator, "runtime.TVMTensorCopyFromBytes", &.{
        tensor,
        any_ptr(@ptrCast(@constCast(data.ptr))),
        any_int(@intCast(nbytes)),
    }, &copy_out);

    return @ptrCast(tensor.unnamed_1.v_obj);
}

fn get_last_error_message(allocator: std.mem.Allocator) ![]u8 {
    var err_obj: c.TVMFFIObjectHandle = null;
    c.TVMFFIErrorMoveFromRaised(&err_obj);
    if (err_obj == null) return try allocator.dupe(u8, "(unknown TVM error)");
    defer _ = c.TVMFFIObjectDecRef(err_obj);

    const hdr_size = @sizeOf(c.TVMFFIObject);
    const cell_ptr: *const c.TVMFFIErrorCell = @ptrCast(@alignCast(@as([*]const u8, @ptrCast(err_obj)) + hdr_size));
    const msg = cell_ptr.message;
    if (msg.data == null or msg.size == 0) return try allocator.dupe(u8, "(TVM error without message)");
    return try allocator.dupe(u8, msg.data[0..msg.size]);
}

pub fn any_to_string(allocator: std.mem.Allocator, v: *c.TVMFFIAny) ![]u8 {
    // small string is stored inline // NOTE: explain
    if (v.type_index == c.kTVMFFISmallStr) {
        const n: usize = @intCast(v.unnamed_0.small_str_len);
        return try allocator.dupe(u8, v.unnamed_1.v_bytes[0..n]);
    }
    // String object: { TVMFFIObject header, TVMFFIByteArray cell, ... }
    if (v.type_index == c.kTVMFFIStr) {
        const obj: c.TVMFFIObjectHandle = @ptrCast(v.unnamed_1.v_obj);
        defer _ = c.TVMFFIObjectDecRef(obj);

        const hdr_size = @sizeOf(c.TVMFFIObject);
        const ba_ptr: *const c.TVMFFIByteArray = @ptrCast(@alignCast(@as([*]const u8, @ptrCast(obj)) + hdr_size));
        if (ba_ptr.data == null or ba_ptr.size == 0) return try allocator.dupe(u8, "");
        return try allocator.dupe(u8, ba_ptr.data[0..ba_ptr.size]);
    }
    return error.UnexpectedTvmType;
}

fn ffi_call(allocator: std.mem.Allocator, func: c.TVMFFIObjectHandle, args: []const c.TVMFFIAny, out: *c.TVMFFIAny) !void {
    out.* = any_none();
    const arg_ptr = if (args.len == 0) null else @constCast(args.ptr);
    if (c.TVMFFIFunctionCall(func, arg_ptr, @intCast(args.len), out) != 0) {
        const msg = try get_last_error_message(allocator);
        defer allocator.free(msg);
        std.log.err("TVMFFIFunctionCall failed: {s}", .{msg});
        return error.TvmRuntimeError;
    }
}

fn ffi_call0(allocator: std.mem.Allocator, func: c.TVMFFIObjectHandle, out: *c.TVMFFIAny) !void {
    try ffi_call(allocator, func, &.{}, out);
}

fn ffi_call1_i64(allocator: std.mem.Allocator, func: c.TVMFFIObjectHandle, x: i64, out: *c.TVMFFIAny) !void {
    const arg = any_int(x);
    const args = [_]c.TVMFFIAny{arg};
    try ffi_call(allocator, func, args[0..], out);
}

// NOTE: allocator is only used for getting error msg, could just use a buffer or an fba. does passing an allocator obscure ownership here? could be viable to keep the allocator but document lifetime, or have user pass buffer? same comment goes for other locations with the same pattern.
fn ffi_get_global(allocator: std.mem.Allocator, name: []const u8) !c.TVMFFIObjectHandle {
    var name_arr: c.TVMFFIByteArray = .{ .data = name.ptr, .size = name.len };
    var out: c.TVMFFIObjectHandle = null;
    if (c.TVMFFIFunctionGetGlobal(&name_arr, &out) != 0 or out == null) {
        const msg = try get_last_error_message(allocator);
        defer allocator.free(msg);
        std.log.err("TVMFFIFunctionGetGlobal({s}) failed: {s}", .{ name, msg });
        return error.TvmRuntimeError;
    }
    return out;
}

pub fn ffi_call_global(allocator: std.mem.Allocator, name: []const u8, args: []const c.TVMFFIAny, out: *c.TVMFFIAny) !void {
    const func = try ffi_get_global(allocator, name);
    defer _ = c.TVMFFIObjectDecRef(func);
    try ffi_call(allocator, func, args, out);
}

pub fn cstr_alloc(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    const buf = try allocator.alloc(u8, s.len + 1);
    std.mem.copyForwards(u8, buf[0..s.len], s);
    buf[s.len] = 0;
    return buf;
}

pub fn cstr_ptr(buf: []u8) [*:0]const u8 {
    return @ptrCast(buf.ptr);
}

/// Log number of functions in an IRModule.
fn logModuleFuncCount(allocator: std.mem.Allocator, mod: c.TVMFFIAny, label: []const u8, comptime logger: anytype) void {
    // get global vars array w/ ir.Module_GetGlobalVars (method on IRModule)
    var gvars: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    ffi_call_global(allocator, "ir.Module_GetGlobalVars", &.{mod}, &gvars) catch {
        logger.debug("{s}: could not get global vars", .{label});
        return;
    };
    // get array size
    var size: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    ffi_call_global(allocator, "ffi.ArraySize", &.{gvars}, &size) catch {
        logger.debug("{s}: could not get array size", .{label});
        return;
    };
    logger.debug("{s}: {d} functions in module", .{ label, size.unnamed_1.v_int64 });
}

/// Load a TVM module from file. Format is auto-detected from extension.
fn module_load_from_file(allocator: std.mem.Allocator, path: []const u8) !c.TVMFFIObjectHandle {
    const path_buf = try cstr_alloc(allocator, path);
    defer allocator.free(path_buf);
    const path_z = cstr_ptr(path_buf);

    var out: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    var args = [_]c.TVMFFIAny{any_raw_str(path_z)};
    try ffi_call_global(allocator, "ffi.ModuleLoadFromFile", &args, &out);
    if (out.type_index != c.kTVMFFIModule or out.unnamed_1.v_obj == null) {
        std.log.err("unexpected ModuleLoadFromFile return type: {d}", .{out.type_index});
        return error.UnexpectedTvmType;
    }
    return @ptrCast(out.unnamed_1.v_obj);
}

/// Write a TVM module to file.
///
/// Supports formats: "o" (object), "ll" (LLVM IR), "bc" (bitcode), "s" (asm).
fn module_write_to_file(allocator: std.mem.Allocator, module: c.TVMFFIAny, path: []const u8, format: []const u8) !void {
    const log = std.log.scoped(.@"zg/tvm_module");

    const path_buf = try cstr_alloc(allocator, path);
    defer allocator.free(path_buf);
    const fmt_buf = try cstr_alloc(allocator, format);
    defer allocator.free(fmt_buf);
    const path_z = cstr_ptr(path_buf);
    const fmt_z = cstr_ptr(fmt_buf);

    var out: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    var args = [_]c.TVMFFIAny{ module, any_raw_str(path_z), any_raw_str(fmt_z) };
    try ffi_call_global(allocator, "ffi.ModuleWriteToFile", &args, &out);
    log.debug("Wrote module to {s} (format={s})", .{ path, format });
}

/// Link one or more object files into a shared library using the system linker with `zig cc`.
fn link_objects_to_shared(allocator: std.mem.Allocator, obj_paths: []const []const u8, so_path: []const u8) !void {
    const log = std.log.scoped(.@"zg/tvm_linker");

    // build argv: zig cc -shared -fPIC -o <so_path> <obj1> [obj2] ...
    var argv_list = std.ArrayList([]const u8).empty;
    defer argv_list.deinit(allocator);
    try argv_list.appendSlice(allocator, &.{ "zig", "cc", "-shared", "-fPIC", "-o", so_path });
    try argv_list.appendSlice(allocator, obj_paths);

    var child = std.process.Child.init(argv_list.items, allocator);
    const term = try child.spawnAndWait();

    switch (term) {
        .Exited => |code| {
            if (code != 0) {
                log.err("Linker exited with code {d}", .{code});
                return error.LinkerFailed;
            }
        },
        else => {
            log.err("Linker terminated abnormally", .{});
            return error.LinkerFailed;
        },
    }
    for (obj_paths) |p| log.debug("  linked: {s}", .{p});
    log.debug("-> {s}", .{so_path});
}

fn module_get_function(
    allocator: std.mem.Allocator,
    module: c.TVMFFIObjectHandle,
    name: []const u8,
    query_imports: bool,
) !c.TVMFFIObjectHandle {
    // NOTE: seems like "name" is always comptime known, so we dont need heap. Also, we could just change the param to be the required type and it will
    //  coerce or be otherwise cast by the caller.
    const name_buf = try cstr_alloc(allocator, name);
    defer allocator.free(name_buf);
    const name_z = cstr_ptr(name_buf);

    var out: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    var args = [_]c.TVMFFIAny{
        any_obj(module, c.kTVMFFIModule),
        any_raw_str(name_z),
        any_bool(query_imports),
    };
    try ffi_call_global(allocator, "ffi.ModuleGetFunction", &args, &out);
    if (out.type_index != c.kTVMFFIFunction or out.unnamed_1.v_obj == null) {
        std.log.err("unexpected ModuleGetFunction return type: {d}", .{out.type_index});
        return error.UnexpectedTvmType;
    }
    return @ptrCast(out.unnamed_1.v_obj);
}

fn make_dl_tensor_f32(data: []f32, shape: []i64) c.DLTensor {
    return .{
        .data = @ptrCast(data.ptr),
        .device = .{ .device_type = c.kDLCPU, .device_id = 0 },
        .ndim = @intCast(shape.len),
        .dtype = .{ .code = c.kDLFloat, .bits = 32, .lanes = 1 },
        .shape = shape.ptr,
        .strides = null,
        .byte_offset = 0,
    };
}

fn dlpack_noop_deleter(tensor: ?*c.DLManagedTensor) callconv(.c) void {
    _ = tensor;
}

fn tensor_from_dlpack(allocator: std.mem.Allocator, managed: *c.DLManagedTensor) !c.TVMFFIObjectHandle {
    var out: c.TVMFFIObjectHandle = null;
    if (c.TVMFFITensorFromDLPack(managed, 0, 0, &out) != 0 or out == null) {
        const msg = try get_last_error_message(allocator);
        defer allocator.free(msg);
        std.log.err("TVMFFITensorFromDLPack failed: {s}", .{msg});
        return error.TvmRuntimeError;
    }
    return out;
}

pub fn print_global_functions(allocator: std.mem.Allocator) !void {
    if (!build_options.enable_tvm) {
        std.log.err("TVM runtime support disabled (SDK missing TVM or build with -Dtvm=true)", .{});
        return error.TvmDisabled;
    }

    const name = "ffi.FunctionListGlobalNamesFunctor";
    const factory = try ffi_get_global(allocator, name);
    defer _ = c.TVMFFIObjectDecRef(factory);

    var res0: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    try ffi_call0(allocator, factory, &res0);
    if (res0.type_index != c.kTVMFFIFunction or res0.unnamed_1.v_obj == null) {
        std.log.err("unexpected return type from {s}(): type_index={d}", .{ name, res0.type_index });
        return error.UnexpectedTvmType;
    }
    const functor: c.TVMFFIObjectHandle = @ptrCast(res0.unnamed_1.v_obj);
    defer _ = c.TVMFFIObjectDecRef(functor);

    var res_len: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    try ffi_call1_i64(allocator, functor, -1, &res_len);
    if (res_len.type_index != c.kTVMFFIInt) {
        std.log.err("unexpected len return type: type_index={d}", .{res_len.type_index});
        return error.UnexpectedTvmType;
    }
    const count: usize = @intCast(res_len.unnamed_1.v_int64);

    var stdout_buffer: [8192]u8 = undefined; // NOTE: audit use of undefined
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const out = &stdout_writer.interface;
    defer out.flush() catch {};

    try out.print("TVM FFI global functions: {d}\n", .{count});
    for (0..count) |i| {
        var res_name: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        try ffi_call1_i64(allocator, functor, @intCast(i), &res_name);
        const s = try any_to_string(allocator, &res_name);
        defer allocator.free(s);
        try out.print("  {s}\n", .{s});
    }
}

// Re-export TargetKind from common for convenience
pub const TargetKind = tvm_common.TargetKind;

/// Build a TVM vec_add module in-memory and execute it.
///
/// Uses TVM's TE (Tensor Expression) API to define the computation, then
/// compiles and runs it. This requires the full TVM compiler (libtvm.so).
pub fn build_and_run_vec_add(allocator: std.mem.Allocator, n: usize, target_kind: TargetKind) !void {
    if (!build_options.enable_tvm) {
        std.log.err("TVM runtime support disabled", .{});
        return error.TvmDisabled;
    }
    const log = std.log.scoped(.@"zg/tvm_build");

    // Load the full TVM compiler (registers TE functions via static initializers)
    ensureTvmCompilerLoaded(allocator) catch |err| {
        log.err("cannot proceed without TVM compiler: {s}", .{@errorName(err)});
        return err;
    };

    // 1. Create shape as Array<ir.PrimExpr>
    const n_i64: i64 = @intCast(n);

    // te.Placeholder expects shape as Array<ir.PrimExpr>
    // integers can be used directly as PrimExpr in TVM
    var shape_array: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        // ffi.Array takes variadic elements - integers are auto-converted to IntImm
        var args = [_]c.TVMFFIAny{any_int(n_i64)};
        ffi_call_global(allocator, "ffi.Array", &args, &shape_array) catch |err| {
            log.err("ffi.Array failed: {s}", .{@errorName(err)});
            return err;
        };
    }
    log.info("Created shape array [{d}]", .{n});
    defer if (shape_array.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    // 2. Create placeholder tensors A and B
    // te.Placeholder(shape: Array<PrimExpr>, dtype: DataType, name: string)
    const name_a_buf = try cstr_alloc(allocator, "A");
    defer allocator.free(name_a_buf);
    const name_b_buf = try cstr_alloc(allocator, "B");
    defer allocator.free(name_b_buf);
    // Use float32 for now (standard C support, CUDA with fp32 also works)
    // TODO: switch to float16 for CUDA when we get the include paths sorted out
    const dtype_buf = try cstr_alloc(allocator, "float32");
    defer allocator.free(dtype_buf);

    var tensor_a: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{
            shape_array,
            any_raw_str(cstr_ptr(dtype_buf)),
            any_raw_str(cstr_ptr(name_a_buf)),
        };
        ffi_call_global(allocator, "te.Placeholder", &args, &tensor_a) catch |err| {
            log.err("te.Placeholder(A) failed: {s}", .{@errorName(err)});
            return err;
        };
    }
    log.info("Created placeholder A: type_index={d}", .{tensor_a.type_index});
    defer if (tensor_a.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    var tensor_b: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{
            shape_array,
            any_raw_str(cstr_ptr(dtype_buf)),
            any_raw_str(cstr_ptr(name_b_buf)),
        };
        ffi_call_global(allocator, "te.Placeholder", &args, &tensor_b) catch |err| {
            log.err("te.Placeholder(B) failed: {s}", .{@errorName(err)});
            return err;
        };
    }
    log.info("Created placeholder B: type_index={d}", .{tensor_b.type_index});
    defer if (tensor_b.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    // 3. Create iteration variable for the compute
    const name_i_buf = try cstr_alloc(allocator, "i");
    defer allocator.free(name_i_buf);
    const int32_buf = try cstr_alloc(allocator, "int32");
    defer allocator.free(int32_buf);

    // create a tir.Var for the loop index
    // tir.Var(name: str, dtype: AnyView, span: ir.Span)
    var iter_var_i: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{
            any_raw_str(cstr_ptr(name_i_buf)),
            any_raw_str(cstr_ptr(int32_buf)),
            any_none(), // span
        };
        ffi_call_global(allocator, "tir.Var", &args, &iter_var_i) catch |err| {
            log.err("tir.Var failed: {s}", .{@errorName(err)});
            return err;
        };
    }
    log.info("Created iter var i: type_index={d}", .{iter_var_i.type_index});
    defer if (iter_var_i.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    // 4. Create load expressions A[i] and B[i]
    // tir.ProducerLoad(producer, indices)
    var indices_array: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{iter_var_i};
        ffi_call_global(allocator, "ffi.Array", &args, &indices_array) catch |err| {
            log.err("ffi.Array(indices) failed: {s}", .{@errorName(err)});
            return err;
        };
    }
    defer if (indices_array.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    var load_a: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{ tensor_a, indices_array, any_none() };
        ffi_call_global(allocator, "tir.ProducerLoad", &args, &load_a) catch |err| {
            log.err("tir.ProducerLoad(A) failed: {s}", .{@errorName(err)});
            return err;
        };
    }
    log.info("Created ProducerLoad(A, [i]): type_index={d}", .{load_a.type_index});
    defer if (load_a.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    var load_b: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{ tensor_b, indices_array, any_none() };
        ffi_call_global(allocator, "tir.ProducerLoad", &args, &load_b) catch |err| {
            log.err("tir.ProducerLoad(B) failed: {s}", .{@errorName(err)});
            return err;
        };
    }
    log.info("Created ProducerLoad(B, [i]): type_index={d}", .{load_b.type_index});
    defer if (load_b.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    // 5. Create Add expression: A[i] + B[i]
    var add_expr: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{ load_a, load_b, any_none() };
        ffi_call_global(allocator, "tir.Add", &args, &add_expr) catch |err| {
            log.err("tir.Add failed: {s}", .{@errorName(err)});
            return err;
        };
    }
    log.info("Created Add(A[i], B[i]): type_index={d}", .{add_expr.type_index});
    defer if (add_expr.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    // 6. Create IterVar for the compute axis
    // tir.IterVar(dom, var, iter_type, thread_tag)
    var iter_dom: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        // ir.Range(begin, end, span) - span can be None
        var args = [_]c.TVMFFIAny{ any_int(0), any_int(n_i64), any_none() };
        ffi_call_global(allocator, "ir.Range", &args, &iter_dom) catch |err| {
            log.err("ir.Range failed: {s}", .{@errorName(err)});
            return err;
        };
    }
    log.info("Created Range [0, {d})", .{n});
    defer if (iter_dom.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    var axis_iter_var: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        // IterVar(dom, var, iter_type=0 (DataPar), thread_tag="", span)
        const empty_buf = try cstr_alloc(allocator, "");
        defer allocator.free(empty_buf);
        var args = [_]c.TVMFFIAny{
            iter_dom,
            iter_var_i,
            any_int(0), // kDataPar
            any_raw_str(cstr_ptr(empty_buf)),
            any_none(), // span
        };
        ffi_call_global(allocator, "tir.IterVar", &args, &axis_iter_var) catch |err| {
            log.err("tir.IterVar failed: {s}", .{@errorName(err)});
            return err;
        };
    }
    log.info("Created IterVar: type_index={d}", .{axis_iter_var.type_index});
    defer if (axis_iter_var.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    // 7. Create ComputeOp
    // te.ComputeOp(name, tag, attrs, axis, body)
    const name_c_buf = try cstr_alloc(allocator, "C");
    defer allocator.free(name_c_buf);
    const tag_buf = try cstr_alloc(allocator, "");
    defer allocator.free(tag_buf);

    var axis_array: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{axis_iter_var};
        ffi_call_global(allocator, "ffi.Array", &args, &axis_array) catch |err| {
            log.err("ffi.Array(axis) failed: {s}", .{@errorName(err)});
            return err;
        };
    }
    defer if (axis_array.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    var body_array: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{add_expr};
        ffi_call_global(allocator, "ffi.Array", &args, &body_array) catch |err| {
            log.err("ffi.Array(body) failed: {s}", .{@errorName(err)});
            return err;
        };
    }
    defer if (body_array.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    var compute_op: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        // te.ComputeOp(name: str, tag: str, attrs: dict, axis: Array[IterVar], body: Array[Expr])
        var args = [_]c.TVMFFIAny{
            any_raw_str(cstr_ptr(name_c_buf)),
            any_raw_str(cstr_ptr(tag_buf)),
            any_none(), // attrs
            axis_array,
            body_array,
        };
        ffi_call_global(allocator, "te.ComputeOp", &args, &compute_op) catch |err| {
            log.err("te.ComputeOp failed: {s}", .{@errorName(err)});
            return err;
        };
    }
    log.info("Created ComputeOp C = A + B: type_index={d}", .{compute_op.type_index});
    defer if (compute_op.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    // 8. Get output tensor C from ComputeOp
    var tensor_c: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{ compute_op, any_int(0) };
        ffi_call_global(allocator, "te.OpGetOutput", &args, &tensor_c) catch |err| {
            log.err("te.OpGetOutput failed: {s}", .{@errorName(err)});
            return err;
        };
    }
    log.info("Created output tensor C: type_index={d}", .{tensor_c.type_index});
    defer if (tensor_c.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    // 9. Create PrimFunc from tensors
    // te.CreatePrimFunc(tensors: Array[Tensor]) -> tir.PrimFunc
    var tensors_array: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{ tensor_a, tensor_b, tensor_c };
        ffi_call_global(allocator, "ffi.Array", &args, &tensors_array) catch |err| {
            log.err("ffi.Array(tensors) failed: {s}", .{@errorName(err)});
            return err;
        };
    }
    defer if (tensors_array.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    var prim_func: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        // te.CreatePrimFunc(tensors: Array<ObjectRef>, index_dtype_override: Optional<DataType>)
        var args = [_]c.TVMFFIAny{ tensors_array, any_none() };
        ffi_call_global(allocator, "te.CreatePrimFunc", &args, &prim_func) catch |err| {
            log.err("te.CreatePrimFunc failed: {s}", .{@errorName(err)});
            return err;
        };
    }
    log.info("Created PrimFunc: type_index={d}", .{prim_func.type_index});

    // Add global_symbol attribute - required for MakePackedAPI to process this function
    {
        const global_symbol_buf = try cstr_alloc(allocator, "global_symbol");
        defer allocator.free(global_symbol_buf);
        const main_name_buf = try cstr_alloc(allocator, "main");
        defer allocator.free(main_name_buf);

        var prim_func_with_attr: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        var args = [_]c.TVMFFIAny{
            prim_func,
            any_raw_str(cstr_ptr(global_symbol_buf)),
            any_raw_str(cstr_ptr(main_name_buf)),
        };
        ffi_call_global(allocator, "ir.BaseFuncWithAttr", &args, &prim_func_with_attr) catch |err| {
            log.err("ir.BaseFuncWithAttr failed: {s}", .{@errorName(err)});
            return err;
        };
        // Replace prim_func with the attributed version
        if (prim_func.unnamed_1.v_obj) |obj| {
            _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
        }
        prim_func = prim_func_with_attr;
        log.info("Added global_symbol attribute to PrimFunc", .{});
    }

    defer if (prim_func.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    // 10. Create target with host
    // For CPU: use 'llvm' target for JIT compilation (now works with LLVM 21).
    // For CUDA: device is 'cuda', host is 'llvm'.
    const host_str = try cstr_alloc(allocator, "llvm");
    defer allocator.free(host_str);

    // Device target depends on target_kind
    const device_str = switch (target_kind) {
        .cpu => try cstr_alloc(allocator, "llvm"), // NOTE: audit static string being allocated, also casting indirection when it could be initialized to the correct type -- this happens in MANY places
        .cuda => try cstr_alloc(allocator, "cuda"), // NOTE: audit static string being allocated, also casting indirection when it could be initialized to the correct type -- this happens in MANY places
    };
    defer allocator.free(device_str);

    var target: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        // Create device target
        var device_target: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        {
            var args = [_]c.TVMFFIAny{any_raw_str(cstr_ptr(device_str))};
            ffi_call_global(allocator, "target.Target", &args, &device_target) catch |err| {
                log.err("target.Target({s}) failed: {s}", .{ device_str, @errorName(err) });
                return err;
            };
        }
        // Create host target
        var host_target: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        {
            var args = [_]c.TVMFFIAny{any_raw_str(cstr_ptr(host_str))};
            ffi_call_global(allocator, "target.Target", &args, &host_target) catch |err| {
                if (device_target.unnamed_1.v_obj) |obj| {
                    _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
                }
                log.err("target.Target(host) failed: {s}", .{@errorName(err)});
                return err;
            };
        }
        defer if (host_target.unnamed_1.v_obj) |obj| {
            _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
        };

        // Set host on target: target.WithHost(target, host)
        {
            var args = [_]c.TVMFFIAny{ device_target, host_target };
            ffi_call_global(allocator, "target.WithHost", &args, &target) catch |err| {
                if (device_target.unnamed_1.v_obj) |obj| {
                    _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
                }
                log.err("target.WithHost failed: {s}", .{@errorName(err)});
                return err;
            };
        }
        // device_target is consumed by WithHost
        if (device_target.unnamed_1.v_obj) |obj| {
            _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
        }
    }
    log.info("Created target '{s}' with host: type_index={d}", .{ device_str, target.type_index });
    defer if (target.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    // 11. Build module
    // target.Build(IRModule, Target) -> runtime.Module
    // First wrap prim_func in an IRModule
    var ir_mod: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        // Create GlobalVar for the function name
        const main_buf = try cstr_alloc(allocator, "main");
        defer allocator.free(main_buf);

        var global_var: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        {
            var args = [_]c.TVMFFIAny{any_raw_str(cstr_ptr(main_buf))};
            ffi_call_global(allocator, "ir.GlobalVar", &args, &global_var) catch |err| {
                log.err("ir.GlobalVar failed: {s}", .{@errorName(err)});
                return err;
            };
        }
        defer if (global_var.unnamed_1.v_obj) |obj| {
            _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
        };

        // Create Map<GlobalVar, BaseFunc> with the function
        var func_map: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        {
            var map_args = [_]c.TVMFFIAny{ global_var, prim_func };
            ffi_call_global(allocator, "ffi.Map", &map_args, &func_map) catch |err| {
                log.err("ffi.Map(funcs) failed: {s}", .{@errorName(err)});
                return err;
            };
        }
        defer if (func_map.unnamed_1.v_obj) |obj| {
            _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
        };

        // Empty map for global_infos
        var empty_map: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        {
            ffi_call_global(allocator, "ffi.Map", &.{}, &empty_map) catch |err| {
                log.err("ffi.Map(empty) failed: {s}", .{@errorName(err)});
                return err;
            };
        }
        defer if (empty_map.unnamed_1.v_obj) |obj| {
            _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
        };

        // ir.IRModule(Map<GlobalVar, BaseFunc>, ObjectRef attrs, Map<String, Array<GlobalInfo>>)
        var mod_args = [_]c.TVMFFIAny{ func_map, any_none(), empty_map };
        ffi_call_global(allocator, "ir.IRModule", &mod_args, &ir_mod) catch |err| {
            log.err("ir.IRModule failed: {s}", .{@errorName(err)});
            return err;
        };
    }
    log.info("Created IRModule: type_index={d}", .{ir_mod.type_index});
    defer if (ir_mod.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    // 11b. Apply TIR lowering passes
    // The codegen requires buffer_map to be empty. We must run lowering passes first.
    // Minimal pipeline: BindTarget → FlattenBuffer → MakePackedAPI → finalization
    var lowered_mod = ir_mod;

    // Create and apply sequential passes
    {
        // 1. BindTarget - binds target to all functions
        var bind_pass: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        {
            var args = [_]c.TVMFFIAny{target};
            try ffi_call_global(allocator, "tir.transform.BindTarget", &args, &bind_pass);
        }
        defer if (bind_pass.unnamed_1.v_obj) |obj| {
            _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
        };

        var pass_result: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        {
            var args = [_]c.TVMFFIAny{ bind_pass, lowered_mod };
            try ffi_call_global(allocator, "transform.RunPass", &args, &pass_result);
        }
        if (lowered_mod.unnamed_1.v_obj != pass_result.unnamed_1.v_obj) {
            if (lowered_mod.unnamed_1.v_obj) |obj| {
                _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
            }
        }
        lowered_mod = pass_result;
        log.info("Applied BindTarget", .{});
    }

    // Helper to apply a nullary transform pass // NOTE: explain
    const apply_pass = struct {
        fn f(alloc: std.mem.Allocator, pass_name: []const u8, mod: *c.TVMFFIAny, logger: anytype) !void {
            var pass: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
            ffi_call_global(alloc, pass_name, &.{}, &pass) catch |err| {
                logger.err("{s}() failed: {s}", .{ pass_name, @errorName(err) });
                return err;
            };
            defer if (pass.unnamed_1.v_obj) |obj| {
                _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
            };

            var result: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
            var run_args = [_]c.TVMFFIAny{ pass, mod.* };
            ffi_call_global(alloc, "transform.RunPass", &run_args, &result) catch |err| {
                logger.err("RunPass({s}) failed: {s}", .{ pass_name, @errorName(err) });
                return err;
            };

            if (mod.unnamed_1.v_obj != result.unnamed_1.v_obj) {
                if (mod.unnamed_1.v_obj) |obj| {
                    _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
                }
            }
            mod.* = result;
            logger.info("Applied {s}", .{pass_name});
        }
    }.f;

    // Apply the lowering pipeline
    try apply_pass(allocator, "tir.transform.PlanAndUpdateBufferAllocationLocation", &lowered_mod, log);
    try apply_pass(allocator, "tir.transform.ConvertBlocksToOpaque", &lowered_mod, log);
    try apply_pass(allocator, "tir.transform.LowerOpaqueBlock", &lowered_mod, log);
    try apply_pass(allocator, "tir.transform.FlattenBuffer", &lowered_mod, log);
    try apply_pass(allocator, "tir.transform.Simplify", &lowered_mod, log);
    try apply_pass(allocator, "tir.transform.AnnotateEntryFunc", &lowered_mod, log);
    try apply_pass(allocator, "tir.transform.SplitHostDevice", &lowered_mod, log);
    try apply_pass(allocator, "tir.transform.MakePackedAPI", &lowered_mod, log);

    // Finalization passes
    try apply_pass(allocator, "tir.transform.LowerTVMBuiltin", &lowered_mod, log);
    try apply_pass(allocator, "tir.transform.LowerIntrin", &lowered_mod, log);

    defer if (lowered_mod.unnamed_1.v_obj != ir_mod.unnamed_1.v_obj) {
        if (lowered_mod.unnamed_1.v_obj) |obj| {
            _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
        }
    };

    // 12. Build runtime module(s)
    var built_mod: c.TVMFFIAny = undefined; // NOTE: audit use of undefined

    if (target_kind == .cpu) {
        // CPU: use LLVM JIT (now works with LLVM 21 matching SDK's LLVM 22)
        var args = [_]c.TVMFFIAny{ lowered_mod, target };
        ffi_call_global(allocator, "target.build.llvm", &args, &built_mod) catch |err| {
            log.err("target.build.llvm failed: {s}", .{@errorName(err)});
            return err;
        };
        log.info("Built CPU module (LLVM JIT): type_index={d}", .{built_mod.type_index});
    } else {
        // CUDA: need to filter and build host (llvm) and device (cuda) separately
        // After SplitHostDevice, functions are annotated with:
        //   - kIsHostFunc attribute (true = host)
        //   - target attribute (llvm = host, cuda = device)
        //
        // We use tir.transform.Filter with a predicate function to separate modules.

        // Create a predicate function for is_host_func using TVMFFIFunctionCreate
        // The predicate checks the function's target attribute
        //
        // After SplitHostDevice, functions have target attribute:
        //   - Host functions: target = "llvm -jit=mcjit" (or similar)
        //   - Device functions: target = "cuda"
        //
        // We check if the target kind name is "llvm" or "c" for host functions.

        const FilterCtx = struct {
            want_host: bool, // true = keep host functions, false = keep device functions
        };

        const filter_cb = struct { // NOTE: do we still need/use this? dont we have the new filter impl? also lots of unresolved comments and dialogue here that needs to be fact checked and audited if this is kept.
            fn f(
                handle: ?*anyopaque,
                args: [*c]const c.TVMFFIAny,
                num_args: i32,
                result: [*c]c.TVMFFIAny,
            ) callconv(.c) c_int {
                _ = num_args;
                const ctx: *const FilterCtx = @ptrCast(@alignCast(handle));

                // args[0] is the PrimFunc
                const func = args[0];

                // Check if this function has a target attribute set to LLVM/C (host) or CUDA (device)
                // We'll use the type_index and target attribute to determine this.
                //
                // For now, use a simple heuristic: check if the function has a body
                // that contains CUDA-specific nodes, or check the target attribute.
                //
                // Since accessing attributes from C callback is complex, let's use
                // a simpler approach: check the function's "is_host_func" attribute
                // which SplitHostDevice should set.

                // For functions after SplitHostDevice:
                // - Host functions have kIsHostFunc = true
                // - Device kernels don't have this attribute
                //
                // We need to call TVM's HasNonzeroAttr or similar, but that's complex.
                // Let's try a workaround: inspect the function's internal structure.

                // For now, as a workaround, we'll check the type_index.
                // This is not robust, but might work for basic cases.
                _ = func;

                // Simple approach: we know that after SplitHostDevice with CUDA target:
                // - There should be 2 functions: host wrapper (main) and device kernel (main_kernel)
                // - The host wrapper calls the device kernel
                //
                // Since both are PrimFuncs with same type_index, we need attribute checking.
                // As a temporary workaround, return true for both and let the build handle it.
                // This won't work, but let's see what error we get.

                // Actually, let's always keep the function for now
                // and fix the filtering logic later
                result.* = any_bool(ctx.want_host);
                return 0;
            }
        }.f;

        // Create targets
        var host_only_target: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        {
            var args = [_]c.TVMFFIAny{any_raw_str(cstr_ptr(host_str))};
            ffi_call_global(allocator, "target.Target", &args, &host_only_target) catch |err| {
                log.err("target.Target(host_only) failed: {s}", .{@errorName(err)});
                return err;
            };
        }
        defer if (host_only_target.unnamed_1.v_obj) |obj| {
            _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
        };

        var device_only_target: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        {
            var args = [_]c.TVMFFIAny{any_raw_str(cstr_ptr(device_str))};
            ffi_call_global(allocator, "target.Target", &args, &device_only_target) catch |err| {
                log.err("target.Target(device_only) failed: {s}", .{@errorName(err)});
                return err;
            };
        }
        defer if (device_only_target.unnamed_1.v_obj) |obj| {
            _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
        };

        // use tir.transform.Filter to create host and device modules
        // we need to register a callback function with tvm
        var host_filter_ctx = FilterCtx{ .want_host = true };
        var host_filter_func: c.TVMFFIObjectHandle = null;
        if (c.TVMFFIFunctionCreate(&host_filter_ctx, filter_cb, null, &host_filter_func) != 0) {
            log.err("TVMFFIFunctionCreate(host_filter) failed", .{});
            return error.TvmRuntimeError;
        }
        defer _ = c.TVMFFIObjectDecRef(host_filter_func);

        var device_filter_ctx = FilterCtx{ .want_host = false };
        var device_filter_func: c.TVMFFIObjectHandle = null;
        if (c.TVMFFIFunctionCreate(&device_filter_ctx, filter_cb, null, &device_filter_func) != 0) {
            log.err("TVMFFIFunctionCreate(device_filter) failed", .{});
            return error.TvmRuntimeError;
        }
        defer _ = c.TVMFFIObjectDecRef(device_filter_func);

        // create Filter passes
        var host_filter_pass: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        {
            var args = [_]c.TVMFFIAny{any_obj(host_filter_func, c.kTVMFFIFunction)};
            ffi_call_global(allocator, "tir.transform.Filter", &args, &host_filter_pass) catch |err| {
                log.err("tir.transform.Filter(host) failed: {s}", .{@errorName(err)});
                return err;
            };
        }
        defer if (host_filter_pass.unnamed_1.v_obj) |obj| {
            _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
        };

        var device_filter_pass: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        {
            var args = [_]c.TVMFFIAny{any_obj(device_filter_func, c.kTVMFFIFunction)};
            ffi_call_global(allocator, "tir.transform.Filter", &args, &device_filter_pass) catch |err| {
                log.err("tir.transform.Filter(device) failed: {s}", .{@errorName(err)});
                return err;
            };
        }
        defer if (device_filter_pass.unnamed_1.v_obj) |obj| {
            _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
        };

        // apply host filter to get host-only module
        var host_ir_mod: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        {
            var args = [_]c.TVMFFIAny{ host_filter_pass, lowered_mod };
            ffi_call_global(allocator, "transform.RunPass", &args, &host_ir_mod) catch |err| {
                log.err("RunPass(host_filter) failed: {s}", .{@errorName(err)});
                return err;
            };
        }
        defer if (host_ir_mod.unnamed_1.v_obj) |obj| {
            _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
        };
        log.info("Created host-filtered IRModule", .{});

        // apply device filter to get device-only module
        var device_ir_mod: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        {
            var args = [_]c.TVMFFIAny{ device_filter_pass, lowered_mod };
            ffi_call_global(allocator, "transform.RunPass", &args, &device_ir_mod) catch |err| {
                log.err("RunPass(device_filter) failed: {s}", .{@errorName(err)});
                return err;
            };
        }
        defer if (device_ir_mod.unnamed_1.v_obj) |obj| {
            _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
        };
        log.info("Created device-filtered IRModule", .{});

        // Build host module with C backend (avoids LLVM JIT conflicts) // NOTE: still relevant? we switched back to llvm after fixing build issues, this needs analyzes, should we support both?
        var host_mod: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        {
            var args = [_]c.TVMFFIAny{ host_ir_mod, host_only_target };
            ffi_call_global(allocator, "target.build.c", &args, &host_mod) catch |err| {
                log.err("target.build.c(host) failed: {s}", .{@errorName(err)});
                return err;
            };
        }
        log.info("Built host module (C target): type_index={d}", .{host_mod.type_index});

        // build device module w/ CUDA
        var device_mod: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        {
            var args = [_]c.TVMFFIAny{ device_ir_mod, device_only_target };
            ffi_call_global(allocator, "target.build.cuda", &args, &device_mod) catch |err| {
                log.err("target.build.cuda failed: {s}", .{@errorName(err)});
                if (host_mod.unnamed_1.v_obj) |obj| {
                    _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
                }
                return err;
            };
        }
        log.info("Built CUDA device module: type_index={d}", .{device_mod.type_index});
        defer if (device_mod.unnamed_1.v_obj) |obj| {
            _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
        };

        // import device module into host module
        {
            var args = [_]c.TVMFFIAny{ host_mod, device_mod };
            var result: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
            ffi_call_global(allocator, "ffi.ModuleImportModule", &args, &result) catch |err| {
                log.err("ffi.ModuleImportModule failed: {s}", .{@errorName(err)});
                if (host_mod.unnamed_1.v_obj) |obj| {
                    _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
                }
                return err;
            };
        }
        log.info("Imported CUDA module into host", .{});

        built_mod = host_mod;
    }
    defer if (built_mod.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    // 13. Debug module info
    // LLVM JIT produces an LLVMModule thats directly executable via ORC JIT. NOTE: did we get ORC JIT to work? we switched to MC JIT at one point, also we are using C above so is this still an accurate comment?
    {
        var kind: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        var args = [_]c.TVMFFIAny{built_mod};
        ffi_call_global(allocator, "ffi.ModuleGetKind", &args, &kind) catch {};
        if (kind.type_index != c.kTVMFFINone) {
            if (any_to_string(allocator, &kind)) |kind_str| {
                defer allocator.free(kind_str);
                log.info("Module kind: {s}", .{kind_str});
            } else |_| {}
        }
    }

    // 14. Get compiled function
    var compiled_func: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        const main_buf2 = try cstr_alloc(allocator, "main");
        defer allocator.free(main_buf2);
        var args = [_]c.TVMFFIAny{
            built_mod,
            any_raw_str(cstr_ptr(main_buf2)),
            any_bool(true), // query_imports
        };
        ffi_call_global(allocator, "ffi.ModuleGetFunction", &args, &compiled_func) catch |err| {
            log.err("ffi.ModuleGetFunction failed: {s}", .{@errorName(err)});
            return err;
        };
    }
    log.info("Got compiled function 'main': type_index={d}", .{compiled_func.type_index});
    defer if (compiled_func.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    // 15. Execute w/ test data
    // use f32 for CPU (standard C support) // NOTE: hard coded f32?
    const a_data = try allocator.alloc(f32, n);
    defer allocator.free(a_data);
    const b_data = try allocator.alloc(f32, n);
    defer allocator.free(b_data);
    const c_data = try allocator.alloc(f32, n);
    defer allocator.free(c_data);

    for (a_data, 0..) |*v, i| v.* = @floatFromInt(i);
    for (b_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 2.0;
    @memset(c_data, 0);

    const nbytes = n * @sizeOf(f32);
    var shape_arr = [_]i64{n_i64};

    // Device type for tensors
    const device_type: i32 = switch (target_kind) {
        .cpu => c.kDLCPU,
        .cuda => c.kDLCUDA,
    };
    const device_id: i32 = 0;

    // CUDA: build device tensors
    var t_a: c.TVMFFIObjectHandle = undefined; // NOTE: audit use of undefined
    var t_b: c.TVMFFIObjectHandle = undefined; // NOTE: audit use of undefined
    var t_c: c.TVMFFIObjectHandle = undefined; // NOTE: audit use of undefined

    if (target_kind == .cpu) {
        // CPU: use DLPack with host memory directly (f32)
        var dl_a = c.DLManagedTensor{
            .dl_tensor = make_dl_tensor_f32(a_data, &shape_arr),
            .manager_ctx = null,
            .deleter = dlpack_noop_deleter,
        };
        var dl_b = c.DLManagedTensor{
            .dl_tensor = make_dl_tensor_f32(b_data, &shape_arr),
            .manager_ctx = null,
            .deleter = dlpack_noop_deleter,
        };
        var dl_c = c.DLManagedTensor{
            .dl_tensor = make_dl_tensor_f32(c_data, &shape_arr),
            .manager_ctx = null,
            .deleter = dlpack_noop_deleter,
        };
        t_a = try tensor_from_dlpack(allocator, &dl_a);
        t_b = try tensor_from_dlpack(allocator, &dl_b);
        t_c = try tensor_from_dlpack(allocator, &dl_c);
    } else {
        // CUDA: allocate device tensors and copy input data
        const float32_buf = try cstr_alloc(allocator, "float32"); // NOTE: static string alloc and conversion indirection again, check this
        defer allocator.free(float32_buf);

        // create TVM Shape
        var shape_any: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        {
            var args = [_]c.TVMFFIAny{any_int(n_i64)};
            try ffi_call_global(allocator, "ffi.Shape", &args, &shape_any);
        }
        defer if (shape_any.unnamed_1.v_obj) |obj| {
            _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
        };

        // allocate device tensors
        // runtime.TVMTensorAllocWithScope(shape, dtype, device, mem_scope)
        const alloc_tensor = struct {
            fn f(alloc: std.mem.Allocator, shape: c.TVMFFIAny, dtype_cstr: [*:0]const u8, dev_type: i32, dev_id: i32) !c.TVMFFIObjectHandle {
                var result: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
                const dev_arr = [_]i64{ dev_type, dev_id };
                var dev: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
                {
                    var args = [_]c.TVMFFIAny{ any_int(dev_arr[0]), any_int(dev_arr[1]) };
                    try ffi_call_global(alloc, "ffi.Device", &args, &dev);
                }
                defer if (dev.unnamed_1.v_obj) |obj| {
                    _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
                };

                var args = [_]c.TVMFFIAny{ shape, any_raw_str(dtype_cstr), dev, any_none() };
                try ffi_call_global(alloc, "runtime.TVMTensorAllocWithScope", &args, &result);
                return @ptrCast(result.unnamed_1.v_obj);
            }
        }.f;

        t_a = try alloc_tensor(allocator, shape_any, cstr_ptr(float32_buf), device_type, device_id);
        t_b = try alloc_tensor(allocator, shape_any, cstr_ptr(float32_buf), device_type, device_id);
        t_c = try alloc_tensor(allocator, shape_any, cstr_ptr(float32_buf), device_type, device_id);

        // host->device
        // runtime.TVMTensorCopyFromBytes(tensor, data_ptr, nbytes)
        const copy_to_device = struct {
            fn f(alloc: std.mem.Allocator, tensor: c.TVMFFIObjectHandle, data: [*]const f32, bytes: usize) !void {
                var result: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
                var args = [_]c.TVMFFIAny{
                    any_obj(tensor, c.kTVMFFITensor),
                    .{ .type_index = c.kTVMFFIOpaquePtr, .unnamed_0 = .{ .small_str_len = 0 }, .unnamed_1 = .{ .v_ptr = @ptrCast(@constCast(data)) } },
                    any_int(@intCast(bytes)),
                };
                try ffi_call_global(alloc, "runtime.TVMTensorCopyFromBytes", &args, &result);
            }
        }.f;

        try copy_to_device(allocator, t_a, a_data.ptr, nbytes);
        try copy_to_device(allocator, t_b, b_data.ptr, nbytes);
        log.info("Copied input data to GPU", .{});
    }

    defer _ = c.TVMFFIObjectDecRef(t_a);
    defer _ = c.TVMFFIObjectDecRef(t_b);
    defer _ = c.TVMFFIObjectDecRef(t_c);

    // call compiled function
    var exec_args = [_]c.TVMFFIAny{
        any_obj(t_a, c.kTVMFFITensor),
        any_obj(t_b, c.kTVMFFITensor),
        any_obj(t_c, c.kTVMFFITensor),
    };
    const func_handle: c.TVMFFIObjectHandle = @ptrCast(compiled_func.unnamed_1.v_obj);
    var exec_result: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    try ffi_call(allocator, func_handle, &exec_args, &exec_result);

    log.info("Executed vec_add.", .{});

    // CUDA: transfer result device -> host
    if (target_kind == .cuda) {
        var result: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        var args = [_]c.TVMFFIAny{
            any_obj(t_c, c.kTVMFFITensor),
            .{ .type_index = c.kTVMFFIOpaquePtr, .unnamed_0 = .{ .small_str_len = 0 }, .unnamed_1 = .{ .v_ptr = @ptrCast(c_data.ptr) } },
            any_int(@intCast(nbytes)),
        };
        try ffi_call_global(allocator, "runtime.TVMTensorCopyToBytes", &args, &result);
        log.info("Copied result from GPU", .{});
    }

    // print results
    var stdout_buffer: [8192]u8 = undefined; // NOTE: audit use of undefined
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const out = &stdout_writer.interface;
    defer out.flush() catch {};

    const target_name: []const u8 = switch (target_kind) {
        .cpu => "CPU",
        .cuda => "CUDA",
    };
    try out.print("TVM in-memory vec_add ({s}) result (first 8):\n", .{target_name});
    const limit = @min(n, 8);
    for (c_data[0..limit], 0..) |v, i| {
        try out.print("  C[{d}] = {d} (expected {d})\n", .{ i, v, @as(f32, @floatFromInt(i)) * 3.0 });
    }
}

pub fn run_vec_add(allocator: std.mem.Allocator, module_path: []const u8, n: usize) !void {
    if (!build_options.enable_tvm) {
        std.log.err("TVM runtime support disabled (SDK missing TVM or build with -Dtvm=true)", .{});
        return error.TvmDisabled;
    }

    const module = try module_load_from_file(allocator, module_path);
    defer _ = c.TVMFFIObjectDecRef(module);

    const func = try module_get_function(allocator, module, "vec_add", true);
    defer _ = c.TVMFFIObjectDecRef(func);

    const a = try allocator.alloc(f32, n);
    const b = try allocator.alloc(f32, n);
    const c_out = try allocator.alloc(f32, n);
    defer allocator.free(a);
    defer allocator.free(b);
    defer allocator.free(c_out);

    for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i));
    for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 2.0;
    @memset(c_out, 0);

    var shape = [_]i64{@intCast(n)};

    var dl_a = c.DLManagedTensor{
        .dl_tensor = make_dl_tensor_f32(a, &shape),
        .manager_ctx = null,
        .deleter = dlpack_noop_deleter,
    };
    var dl_b = c.DLManagedTensor{
        .dl_tensor = make_dl_tensor_f32(b, &shape),
        .manager_ctx = null,
        .deleter = dlpack_noop_deleter,
    };
    var dl_c = c.DLManagedTensor{
        .dl_tensor = make_dl_tensor_f32(c_out, &shape),
        .manager_ctx = null,
        .deleter = dlpack_noop_deleter,
    };

    const t_a = try tensor_from_dlpack(allocator, &dl_a);
    const t_b = try tensor_from_dlpack(allocator, &dl_b);
    const t_c = try tensor_from_dlpack(allocator, &dl_c);
    defer _ = c.TVMFFIObjectDecRef(t_a);
    defer _ = c.TVMFFIObjectDecRef(t_b);
    defer _ = c.TVMFFIObjectDecRef(t_c);

    var args = [_]c.TVMFFIAny{
        any_obj(t_a, c.kTVMFFITensor),
        any_obj(t_b, c.kTVMFFITensor),
        any_obj(t_c, c.kTVMFFITensor),
    };
    var res: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    try ffi_call(allocator, func, &args, &res);

    var stdout_buffer: [8192]u8 = undefined; // NOTE: audit use of undefined
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const out = &stdout_writer.interface;
    defer out.flush() catch {};

    try out.writeAll("TVM vec_add result (first 8):\n");
    const limit = @min(n, 8);
    for (c_out[0..limit], 0..) |v, i| {
        try out.print("  [{d}] = {d}\n", .{ i, v });
    }
}

// ============================================================================
// MetaSchedule Autotuning Infrastructure
// ============================================================================

/// Options for TVM autotuning via MetaSchedule.
pub const TuneOpts = struct {
    /// Directory to store tuning database and artifacts.
    work_dir: []const u8 = "artifacts/tvm_cache",
    /// Maximum number of tuning trials.
    max_trials: u32 = 64,
    /// Number of trials per iteration (batch size for parallel builds).
    trials_per_iter: u32 = 16,
};

/// Matmul shape for tuning.
pub const MatmulShape = struct {
    M: usize,
    N: usize,
    K: usize,
};

/// Context passed to Zig builder/runner callbacks during autotuning.
/// Stored in a global to bridge the C callback interface.
const TuneContext = struct {
    allocator: std.mem.Allocator,
    target: c.TVMFFIAny,
    target_kind: TargetKind,
    work_dir: []const u8,
    /// Matmul dimensions for tensor allocation in runner.
    shape: MatmulShape,
    /// Counter for generating unique build IDs (used for .so filenames).
    build_counter: u32 = 0,

    fn init(allocator: std.mem.Allocator, target: c.TVMFFIAny, target_kind: TargetKind, work_dir: []const u8, shape: MatmulShape) TuneContext {
        return .{
            .allocator = allocator,
            .target = target,
            .target_kind = target_kind,
            .work_dir = work_dir,
            .shape = shape,
            .build_counter = 0,
        };
    }

    fn deinit(self: *TuneContext) void { // NOTE: is this used anywhere?
        // No cleanup needed - .so files are left in work_dir for potential reuse
        _ = self;
    }
};

/// Global tuning context - set during tune() and accessed by callbacks.
///
/// ## Notes
/// - This is necessary bc TVM's C callback interface doesnt pass user data.
var g_tune_ctx: ?*TuneContext = null;

/// Helper to get a field from a TVM object by name. // NOTE: should this be in ffi module?
///
/// Uses TVM's reflection system via TVMFFIGetTypeInfo. The field access works by:
/// 1. Getting the object's type_index from the TVMFFIAny
/// 2. Looking up the TVMFFITypeInfo for that type
/// 3. Finding the field by name in the type's field list
/// 4. Computing field_ptr = object_ptr + field.offset
/// 5. Calling the field's getter function
fn ffi_get_attr(_: std.mem.Allocator, obj: c.TVMFFIAny, attr_name: []const u8) !c.TVMFFIAny {
    const log = std.log.scoped(.@"zg/tvm_attr");

    // get type index - for objects, its in `type_index`
    const type_index = obj.type_index;
    if (type_index < c.kTVMFFIStaticObjectBegin) {
        log.warn("Cannot get field from non-object type (type_index={d})", .{type_index});
        return error.TvmRuntimeError;
    }

    // get object pointer
    const obj_ptr = obj.unnamed_1.v_obj;
    if (obj_ptr == null) {
        log.warn("Cannot get field from null object", .{});
        return error.TvmRuntimeError;
    }

    // get type info via reflection
    const type_info: ?*const c.TVMFFITypeInfo = c.TVMFFIGetTypeInfo(type_index);
    if (type_info == null) {
        log.warn("TVMFFIGetTypeInfo({d}) returned null", .{type_index});
        return error.TvmRuntimeError;
    }

    // search for field by name
    const num_fields: usize = @intCast(type_info.?.num_fields);
    const fields = type_info.?.fields;
    if (fields == null) {
        log.warn("Type has no fields (type_index={d})", .{type_index});
        return error.TvmRuntimeError;
    }

    var found_field: ?*const c.TVMFFIFieldInfo = null;
    for (0..num_fields) |i| {
        const field = &fields[i];
        const field_name = field.name;
        if (field_name.data != null and field_name.size == attr_name.len) {
            const name_slice = field_name.data[0..field_name.size];
            if (std.mem.eql(u8, name_slice, attr_name)) {
                found_field = field;
                break;
            }
        }
    }

    if (found_field == null) {
        // DEBUG: log available fields
        log.warn("Field '{s}' not found in type (type_index={d}, num_fields={d})", .{ attr_name, type_index, num_fields });
        for (0..num_fields) |i| {
            const field = &fields[i];
            const name_data = field.name.data;
            const name_size = field.name.size;
            if (name_data != null and name_size > 0) {
                log.debug("  available field: '{s}'", .{name_data[0..name_size]});
            }
        }
        return error.TvmRuntimeError;
    }

    // compute field pointer: object_ptr + offset
    const offset: usize = @intCast(found_field.?.offset);
    const obj_bytes: [*]u8 = @ptrCast(obj_ptr);
    const field_ptr: *anyopaque = @ptrCast(obj_bytes + offset);

    // call field getter
    const getter = found_field.?.getter;
    if (getter == null) {
        log.warn("Field '{s}' has no getter", .{attr_name});
        return error.TvmRuntimeError;
    }

    var result: c.TVMFFIAny = any_none();
    const ret_code = getter.?(field_ptr, &result);
    if (ret_code != 0) {
        log.warn("Field getter for '{s}' failed with code {d}", .{ attr_name, ret_code });
        return error.TvmRuntimeError;
    }

    return result;
}

/// Helper to create a TVM float value. // NOTE: should this be in ffi module?
fn any_float(value: f64) c.TVMFFIAny {
    var v = std.mem.zeroes(c.TVMFFIAny);
    v.type_index = c.kTVMFFIFloat;
    v.unnamed_1.v_float64 = value;
    return v;
}

/// Lower and build a TIR module using the complete lowering pipeline.
///
/// Applies the full TIR lowering pipeline then builds with the appropriate
/// backend (target.build.cuda for CUDA, target.build.llvm for CPU).
fn lowerAndBuildModule(
    allocator: std.mem.Allocator,
    mod: c.TVMFFIAny,
    target: c.TVMFFIAny,
    target_kind: TargetKind,
    comptime log: anytype,
) !c.TVMFFIAny {
    var lowered_mod = mod;
    if (mod.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectIncRef(@ptrCast(obj));
    }

    // check function attrs before processing
    // MetaSchedule candidates should preserve `global_symbol` from the original module
    log.debug("Input module type_index={d}", .{mod.type_index});

    // apply BindTarget to set target attribute on the module // NOTE: is this comment misplaced?

    // helper to apply a single transform pass w/ no args
    const apply_pass = struct {
        fn f(alloc: std.mem.Allocator, pass_name: []const u8, modp: *c.TVMFFIAny, logger: anytype) !void {
            var pass: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
            ffi_call_global(alloc, pass_name, &.{}, &pass) catch |err| {
                logger.debug("Get pass {s} failed: {s}", .{ pass_name, @errorName(err) });
                return err;
            };
            defer if (pass.unnamed_1.v_obj) |obj| {
                _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
            };

            var result: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
            var run_args = [_]c.TVMFFIAny{ pass, modp.* };
            ffi_call_global(alloc, "transform.RunPass", &run_args, &result) catch |err| {
                logger.debug("RunPass({s}) failed: {s}", .{ pass_name, @errorName(err) });
                return err;
            };

            if (modp.unnamed_1.v_obj != result.unnamed_1.v_obj) {
                if (modp.unnamed_1.v_obj) |obj| {
                    _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
                }
            }
            modp.* = result;
            logger.debug("Applied {s}", .{pass_name});
        }
    }.f;

    // helper to apply a transform pass w/ args
    const apply_pass_with_args = struct { // NOTE: is this a bit repetitive? could have a generic (ie fn (comptime foo: something) type { return struct {...}; }) Same goes for other instances as well where we could consider this pattern.
        fn f(alloc: std.mem.Allocator, pass_name: []const u8, pass_args: []const c.TVMFFIAny, modp: *c.TVMFFIAny, logger: anytype) !void {
            var pass: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
            ffi_call_global(alloc, pass_name, pass_args, &pass) catch |err| {
                logger.debug("Get pass {s} failed: {s}", .{ pass_name, @errorName(err) });
                return err;
            };
            defer if (pass.unnamed_1.v_obj) |obj| {
                _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
            };

            var result: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
            var run_args = [_]c.TVMFFIAny{ pass, modp.* };
            ffi_call_global(alloc, "transform.RunPass", &run_args, &result) catch |err| {
                logger.debug("RunPass({s}) failed: {s}", .{ pass_name, @errorName(err) });
                return err;
            };

            if (modp.unnamed_1.v_obj != result.unnamed_1.v_obj) {
                if (modp.unnamed_1.v_obj) |obj| {
                    _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
                }
            }
            modp.* = result;
            logger.debug("Applied {s}", .{pass_name});
        }
    }.f;

    // apply BindTarget first
    {
        var bind_target_pass: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        var bt_args = [_]c.TVMFFIAny{target};
        ffi_call_global(allocator, "tir.transform.BindTarget", &bt_args, &bind_target_pass) catch |err| {
            log.err("Get BindTarget pass failed: {s}", .{@errorName(err)});
            return err;
        };
        defer if (bind_target_pass.unnamed_1.v_obj) |obj| {
            _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
        };

        var result: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        var run_args = [_]c.TVMFFIAny{ bind_target_pass, lowered_mod };
        ffi_call_global(allocator, "transform.RunPass", &run_args, &result) catch |err| {
            log.debug("BindTarget failed: {s}", .{@errorName(err)});
            return err;
        };

        if (lowered_mod.unnamed_1.v_obj != result.unnamed_1.v_obj) {
            if (lowered_mod.unnamed_1.v_obj) |obj| {
                _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
            }
        }
        lowered_mod = result;
        log.debug("Applied BindTarget", .{});
    }

    // Apply default TIR lowering pipeline (based on TVM's `default_tir_pipeline`)
    // Key passes:
    // 1. LowerCrossThreadReduction - handles cross-thread reductions
    // 2. LowerInitBlock - lowers T.init() blocks (REQUIRED before PlanAndUpdate) // NOTE: document how we know this
    // 3. Buffer allocation passes
    // 4. MakePackedAPI - creates wrapper with empty buffer_map
    // 5. Finalization passes

    // Optional: LowerCrossThreadReduction (only for CUDA) // NOTE: comment unclear. is it intending to say its a no-op if target isnt cuda? clarify and likely remove "optional" and parentheticals
    apply_pass(allocator, "tir.transform.LowerCrossThreadReduction", &lowered_mod, log) catch {};

    // NOTE: LowerInitBlock must come before PlanAndUpdateBufferAllocationLocation // NOTE: documetn how we know this
    // this pass handles T.init() blocks in reductions (like matmul accumulator init)
    try apply_pass(allocator, "tir.transform.LowerInitBlock", &lowered_mod, log);

    // buffer allocation and lowering
    try apply_pass(allocator, "tir.transform.PlanAndUpdateBufferAllocationLocation", &lowered_mod, log);
    try apply_pass(allocator, "tir.transform.ConvertBlocksToOpaque", &lowered_mod, log);
    apply_pass(allocator, "tir.transform.LiftThreadBinding", &lowered_mod, log) catch {};
    // CompactBufferAllocation(is_strict: bool)
    apply_pass_with_args(allocator, "tir.transform.CompactBufferAllocation", &[_]c.TVMFFIAny{any_bool(false)}, &lowered_mod, log) catch {};
    apply_pass(allocator, "tir.transform.LowerMatchBuffer", &lowered_mod, log) catch {};
    try apply_pass(allocator, "tir.transform.LowerOpaqueBlock", &lowered_mod, log);
    try apply_pass(allocator, "tir.transform.FlattenBuffer", &lowered_mod, log);

    // loop transforms and vectorization (critical for MetaSchedule candidates) // NOTE: what is this parenthetical? rephrase and provide more explanation/context if necessary to communicate what is meant.
    // NarrowDataType(target_bits: int) - typically 32 for 32-bit
    apply_pass_with_args(allocator, "tir.transform.NarrowDataType", &[_]c.TVMFFIAny{any_int(32)}, &lowered_mod, log) catch {};
    apply_pass(allocator, "tir.transform.LoopPartition", &lowered_mod, log) catch {};
    // VectorizeLoop(enable_vectorize: bool) - true to enable
    apply_pass_with_args(allocator, "tir.transform.VectorizeLoop", &[_]c.TVMFFIAny{any_bool(true)}, &lowered_mod, log) catch {};
    apply_pass(allocator, "tir.transform.InjectVirtualThread", &lowered_mod, log) catch {};
    apply_pass(allocator, "tir.transform.InjectDoubleBuffer", &lowered_mod, log) catch {};
    apply_pass(allocator, "tir.transform.StorageRewrite", &lowered_mod, log) catch {};

    try apply_pass(allocator, "tir.transform.Simplify", &lowered_mod, log);
    apply_pass(allocator, "tir.transform.RemoveNoOp", &lowered_mod, log) catch {};
    // CommonSubexprElimTIR(enable_cse: bool, enable_equiv_terms: bool)
    apply_pass_with_args(allocator, "tir.transform.CommonSubexprElimTIR", &[_]c.TVMFFIAny{ any_bool(true), any_bool(false) }, &lowered_mod, log) catch {};

    // entry function annotation and host/device split
    apply_pass(allocator, "tir.transform.VerifyMemory", &lowered_mod, log) catch {};
    try apply_pass(allocator, "tir.transform.AnnotateEntryFunc", &lowered_mod, log);

    // CUDA-specific passes before SplitHostDevice (matching TVM python pipeline positions 42-48). // NOTE: requires explanation and/or citation
    // ThreadSync inserts __syncthreads() barriers inferred from shared memory access patterns.
    // AnnotateDeviceRegions wraps thread_extent regions with kTarget attributes — required
    // for SplitHostDevice to detect device code and extract it into a separate kernel.
    if (target_kind == .cuda) {
        apply_pass_with_args(allocator, "tir.transform.ThreadSync", &[_]c.TVMFFIAny{any_raw_str("shared")}, &lowered_mod, log) catch {};
        apply_pass_with_args(allocator, "tir.transform.ThreadSync", &[_]c.TVMFFIAny{any_raw_str("shared.dyn")}, &lowered_mod, log) catch {};
        apply_pass_with_args(allocator, "tir.transform.ThreadSync", &[_]c.TVMFFIAny{any_raw_str("warp")}, &lowered_mod, log) catch {};
        apply_pass(allocator, "tir.transform.InferFragment", &lowered_mod, log) catch {};
        apply_pass(allocator, "tir.transform.LowerThreadAllreduce", &lowered_mod, log) catch {};
        try apply_pass(allocator, "tir.transform.AnnotateDeviceRegions", &lowered_mod, log);
    }

    try apply_pass(allocator, "tir.transform.SplitHostDevice", &lowered_mod, log);
    logModuleFuncCount(allocator, lowered_mod, "after SplitHostDevice", log);
    // MergeSharedMemoryAllocations must follow SplitHostDevice (TVM pipeline requirement) // NOTE: what does this mean? how do we know this? requires explanation and/or citation
    if (target_kind == .cuda) {
        apply_pass(allocator, "tir.transform.MergeSharedMemoryAllocations", &lowered_mod, log) catch {};
    }
    try apply_pass(allocator, "tir.transform.MakePackedAPI", &lowered_mod, log);
    logModuleFuncCount(allocator, lowered_mod, "after MakePackedAPI", log);

    // device kernel launch lowering (needed for proper function structure) // NOTE: what does this mean? how do we know this? requires explanation and/or citation. rephrase to remove parenthetical.
    apply_pass(allocator, "tir.transform.LowerDeviceKernelLaunch", &lowered_mod, log) catch {};
    logModuleFuncCount(allocator, lowered_mod, "after LowerDeviceKernelLaunch", log);

    // After LowerDeviceKernelLaunch, the module has both host and device functions.
    // TVM's python pipeline applies finalization passes AFTER filtering:  // NOTE: requires explanation of what these referenced functions do and citation
    //   finalize_host_passes() on host module only
    //   finalize_device_passes() on device module only
    // This is important because host passes (LowerTVMBuiltin) corrupt device functions. // NOTE: elaborate, is this purely empirical or tacit knowledge?

    defer if (lowered_mod.unnamed_1.v_obj != mod.unnamed_1.v_obj) {
        if (lowered_mod.unnamed_1.v_obj) |obj| {
            _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
        }
    };

    switch (target_kind) {
        .cpu => {
            // CPU: no host/device split needed, apply host finalization directly
            apply_pass(allocator, "tir.transform.LowerTVMBuiltin", &lowered_mod, log) catch {};
            apply_pass(allocator, "tir.transform.LowerCustomDatatypes", &lowered_mod, log) catch {};
            try apply_pass(allocator, "tir.transform.LowerIntrin", &lowered_mod, log);
            apply_pass(allocator, "tir.transform.LowerDeviceStorageAccessInfo", &lowered_mod, log) catch {};
            apply_pass(allocator, "tir.transform.CombineContextCall", &lowered_mod, log) catch {};

            return try tvm_builder.build_cpu_module(allocator, lowered_mod, target);
        },
        .cuda => {
            // CUDA: filter into host/device, finalize each separately, then build+link.
            // matches TVM python: split_host_device_mods → finalize_*_passes → codegen_build  // NOTE: Use of Unicode symbol, remove. Also requires citations
            log.info("CUDA build: filtering and finalizing host/device separately", .{});

            // device path
            log.info("Step 1: Filtering device functions", .{}); // NOTE: when logging a counter there should be a total (i.e., 1/5), and "Step" is too vague, different word or remove
            var device_mod = try tvm_cuda.filter_module_by_target(allocator, lowered_mod, .device);
            defer if (device_mod.unnamed_1.v_obj) |obj| {
                _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
            };

            // device finalization (matching TVM's finalize_device_passes) // NOTE: requires citations. rephrase to remove parenthetical.
            apply_pass(allocator, "tir.transform.LowerWarpMemory", &device_mod, log) catch {};
            apply_pass(allocator, "tir.transform.Simplify", &device_mod, log) catch {};
            apply_pass(allocator, "tir.transform.LowerCustomDatatypes", &device_mod, log) catch {};
            apply_pass(allocator, "tir.transform.LowerDeviceStorageAccessInfo", &device_mod, log) catch {};
            apply_pass(allocator, "tir.transform.LowerIntrin", &device_mod, log) catch {};

            log.info("Step 2: Building device kernels (NVRTC)", .{});
            const device_built = try tvm_cuda.build_device_kernels(allocator, device_mod, target);

            // host path
            log.info("Step 3: Filtering host functions", .{});
            var host_mod = try tvm_cuda.filter_module_by_target(allocator, lowered_mod, .host);

            // host finalization (matching TVM's finalize_host_passes) // NOTE: requires citations. rephrase to remove parenthetical.
            apply_pass(allocator, "tir.transform.LowerTVMBuiltin", &host_mod, log) catch {};
            apply_pass(allocator, "tir.transform.LowerCustomDatatypes", &host_mod, log) catch {};
            apply_pass(allocator, "tir.transform.LowerIntrin", &host_mod, log) catch {};
            apply_pass(allocator, "tir.transform.LowerDeviceStorageAccessInfo", &host_mod, log) catch {};
            apply_pass(allocator, "tir.transform.CombineContextCall", &host_mod, log) catch {};

            // create llvm host target
            var host_target: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
            {
                const host_str = try cstr_alloc(allocator, "llvm");
                defer allocator.free(host_str);
                var args = [_]c.TVMFFIAny{any_raw_str(cstr_ptr(host_str))};
                try ffi_call_global(allocator, "target.Target", &args, &host_target);
            }
            defer if (host_target.unnamed_1.v_obj) |obj| {
                _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
            };

            log.info("Step 4: Building host wrapper (LLVM)", .{});
            const host_built = try tvm_cuda.build_host_wrapper(allocator, host_mod, host_target);

            // link device module into host
            log.info("Step 5: Linking device module into host", .{});
            try tvm_cuda.link_device_module(allocator, host_built, device_built);

            log.info("CUDA module built", .{});
            return host_built;
        },
    }
}

/// Builder callback for MetaSchedule - compiles TIR to .so files.
///
/// This function is called by TVM's MetaSchedule infrastructure during autotuning.
/// For each candidate schedule (BuilderInput), it:
/// 1. Extracts the IRModule and target
/// 2. Compiles the module using tvm.build
/// 3. Exports to a temp .so file
/// 4. Returns a BuilderResult with the artifact path
///
/// Signature matches TVM's FBuild: Array<BuilderResult>(Array<BuilderInput>) // NOTE: it is unclear how this signature matches, expand and clarify. also if referenced type is complex then this additionally requires citation.
fn zigBuildCallback( // NOTE: should be snake_case. this applies elsewhere as well. need to update any comments/logs as well.
    _: ?*anyopaque,
    args: [*c]const c.TVMFFIAny,
    num_args: i32,
    result: [*c]c.TVMFFIAny,
) callconv(.c) c_int {
    const log = std.log.scoped(.@"zg/tvm_builder");

    // get global context
    const ctx = g_tune_ctx orelse {
        log.err("zigBuildCallback: no tune context set", .{});
        return -1;
    };

    if (num_args != 1) {
        log.err("zigBuildCallback: expected 1 arg (Array<BuilderInput>), got {d}", .{num_args});
        return -1;
    }

    // args[0] is Array<BuilderInput>
    const inputs_array = args[0];

    // get array length
    var len_result: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    var len_args = [_]c.TVMFFIAny{inputs_array};
    ffi_call_global(ctx.allocator, "ffi.ArraySize", &len_args, &len_result) catch |err| {
        log.err("ffi.ArraySize failed: {s}", .{@errorName(err)});
        return -1;
    };
    const num_inputs: usize = @intCast(len_result.unnamed_1.v_int64);
    log.info("Building {d} candidates", .{num_inputs});

    // create output array
    var results_list = std.ArrayList(c.TVMFFIAny).empty;
    defer results_list.deinit(ctx.allocator);

    for (0..num_inputs) |i| {
        // get BuilderInput[i]
        var input: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        var get_args = [_]c.TVMFFIAny{ inputs_array, any_int(@intCast(i)) };
        ffi_call_global(ctx.allocator, "ffi.ArrayGetItem", &get_args, &input) catch |err| {
            log.err("ffi.ArrayGetItem failed: {s}", .{@errorName(err)});
            return -1;
        };

        // extract mod and target from BuilderInput
        //  BuilderInput has: mod (IRModule), target (Target)
        const mod = ffi_get_attr(ctx.allocator, input, "mod") catch |err| {
            log.err("get mod failed: {s}", .{@errorName(err)});
            // Return error result
            const err_result = createBuilderErrorResult(ctx.allocator, "failed to get mod") catch return -1;
            results_list.append(ctx.allocator, err_result) catch return -1;
            continue;
        };

        // Use the target from context (already set up correctly)
        const target = ctx.target;

        log.info("Processing candidate {d}/{d}", .{ i, num_inputs });

        // apply lowering passes + compile w/ appropriate build backend (cuda/llvm) // NOTE: comment says llvm, elsewhere we say llvm, but we also have code that targets C, this needs to be investigated
        const built_mod = lowerAndBuildModule(ctx.allocator, mod, target, ctx.target_kind, log) catch |err| {
            log.err("lowerAndBuildModule failed: {s}", .{@errorName(err)});
            const err_result = createBuilderErrorResult(ctx.allocator, "compilation failed") catch return -1;
            results_list.append(ctx.allocator, err_result) catch return -1;
            continue;
        };

        log.info("Candidate {d} built successfully, exporting to .so", .{i});

        // generate unique file paths for this candidate
        const build_id = @atomicRmw(u32, &ctx.build_counter, .Add, 1, .seq_cst);

        // export module to .o then link to .so
        const obj_path = std.fmt.allocPrint(ctx.allocator, "{s}/candidate_{d}.o", .{ ctx.work_dir, build_id }) catch return -1;
        defer ctx.allocator.free(obj_path);
        const so_path = std.fmt.allocPrintSentinel(ctx.allocator, "{s}/candidate_{d}.so", .{ ctx.work_dir, build_id }, 0) catch return -1;
        // so_path is kept (not freed) as it's passed to BuilderResult // NOTE: this comment should be more clear about lifetime and ownership

        log.info("Writing module to {s} (target={s})", .{
            obj_path,
            if (ctx.target_kind == .cuda) "cuda" else "cpu",
        });

        // write host module to object file
        module_write_to_file(ctx.allocator, built_mod, obj_path, "o") catch |err| {
            log.err("Failed to write module to {s}: {s}", .{ obj_path, @errorName(err) });
            ctx.allocator.free(so_path);
            const err_result = createBuilderErrorResult(ctx.allocator, "write_to_file failed") catch return -1;
            results_list.append(ctx.allocator, err_result) catch return -1;
            continue;
        };
        log.info("Write host .o succeeded", .{});

        // For CUDA: pack imported device modules into a separate .o, then link both.
        //  ModulePackImportsToLLVM serializes the CUDA/PTX device module into an LLVM
        //   module containing the data blob. TVM's loader deserializes this on module_load.
        if (ctx.target_kind == .cuda) {
            const devc_obj_path = std.fmt.allocPrint(ctx.allocator, "{s}/candidate_{d}_devc.o", .{ ctx.work_dir, build_id }) catch return -1;
            defer ctx.allocator.free(devc_obj_path);

            // pack imports into llvm module // NOTE: explain what this means/does
            const llvm_target_str = cstr_alloc(ctx.allocator, "llvm") catch return -1; // NOTE: same comment about c string here, again I wont comment on all of them but this needs to be addressed as its everywhere
            defer ctx.allocator.free(llvm_target_str);
            const empty_prefix = cstr_alloc(ctx.allocator, "") catch return -1;
            defer ctx.allocator.free(empty_prefix);

            var pack_mod: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
            ffi_call_global(ctx.allocator, "runtime.ModulePackImportsToLLVM", &.{ // NOTE: signature needs to be documented in a comment
                built_mod,
                any_bool(false), // system_lib
                any_raw_str(cstr_ptr(llvm_target_str)),
                any_raw_str(cstr_ptr(empty_prefix)),
            }, &pack_mod) catch |err| {
                log.err("ModulePackImportsToLLVM failed: {s}", .{@errorName(err)});
                ctx.allocator.free(so_path);
                const err_result = createBuilderErrorResult(ctx.allocator, "pack imports failed") catch return -1;
                results_list.append(ctx.allocator, err_result) catch return -1;
                continue;
            };
            defer if (pack_mod.unnamed_1.v_obj) |obj| {
                _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
            };

            // write packed device module to .o
            module_write_to_file(ctx.allocator, pack_mod, devc_obj_path, "o") catch |err| {
                log.err("Failed to write devc module: {s}", .{@errorName(err)});
                ctx.allocator.free(so_path);
                const err_result = createBuilderErrorResult(ctx.allocator, "write devc failed") catch return -1;
                results_list.append(ctx.allocator, err_result) catch return -1;
                continue;
            };
            log.info("Write device .o succeeded", .{});

            // link host.o and devc.o into .so
            link_objects_to_shared(ctx.allocator, &.{ obj_path, devc_obj_path }, so_path) catch |err| {
                log.err("Failed to link CUDA module: {s}", .{@errorName(err)});
                ctx.allocator.free(so_path);
                const err_result = createBuilderErrorResult(ctx.allocator, "linker failed") catch return -1;
                results_list.append(ctx.allocator, err_result) catch return -1;
                continue;
            };
        } else {
            // CPU: single object file
            link_objects_to_shared(ctx.allocator, &.{obj_path}, so_path) catch |err| {
                log.err("Failed to link {s} -> {s}: {s}", .{ obj_path, so_path, @errorName(err) });
                ctx.allocator.free(so_path);
                const err_result = createBuilderErrorResult(ctx.allocator, "linker failed") catch return -1;
                results_list.append(ctx.allocator, err_result) catch return -1;
                continue;
            };
        }
        log.info("Link succeeded: {s}", .{so_path});

        log.debug("Exported module to {s}", .{so_path});

        // create BuilderResult with .so path as artifact_path
        var builder_result: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        var br_args = [_]c.TVMFFIAny{ any_raw_str(so_path), any_none() };
        ffi_call_global(ctx.allocator, "meta_schedule.BuilderResult", &br_args, &builder_result) catch |err| {
            log.err("BuilderResult creation failed: {s}", .{@errorName(err)});
            ctx.allocator.free(so_path);
            return -1;
        };

        results_list.append(ctx.allocator, builder_result) catch return -1;
        log.debug("Built candidate {d} -> {s}", .{ i, so_path });
    }

    // create output Array from results
    var array_result: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    ffi_call_global(ctx.allocator, "ffi.Array", results_list.items, &array_result) catch |err| {
        log.err("ffi.Array creation failed: {s}", .{@errorName(err)});
        return -1;
    };

    result.* = array_result;
    return 0;
}

/// Create a BuilderResult representing an error.
fn createBuilderErrorResult(allocator: std.mem.Allocator, err_msg: []const u8) !c.TVMFFIAny {
    const msg_buf = try cstr_alloc(allocator, err_msg);
    defer allocator.free(msg_buf);

    var result: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    var args = [_]c.TVMFFIAny{ any_none(), any_raw_str(cstr_ptr(msg_buf)) };
    try ffi_call_global(allocator, "meta_schedule.BuilderResult", &args, &result);
    return result;
}

/// Runner callback for MetaSchedule - measures kernel execution time.
///
/// For each RunnerInput (compiled .so artifact), it:
/// 1. Loads the module from the artifact path
/// 2. Allocates test tensors based on args_info
/// 3. Runs warmup iterations
/// 4. Times multiple runs and computes average
/// 5. Returns RunnerFuture wrapping the timing result
///
/// Signature matches TVM's FRun: Array<RunnerFuture>(Array<RunnerInput>) // NOTE: it is unclear how this signature matches, expand and clarify. also if referenced type is complex then this additionally requires citation.
fn zigRunCallback(
    _: ?*anyopaque,
    args: [*c]const c.TVMFFIAny,
    num_args: i32,
    result: [*c]c.TVMFFIAny,
) callconv(.c) c_int {
    const log = std.log.scoped(.@"zg/tvm_runner");

    const ctx = g_tune_ctx orelse {
        log.err("zigRunCallback: no tune context set", .{});
        return -1;
    };

    if (num_args != 1) {
        log.err("zigRunCallback: expected 1 arg, got {d}", .{num_args});
        return -1;
    }

    const inputs_array = args[0];

    // get array length
    var len_result: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    var len_args = [_]c.TVMFFIAny{inputs_array};
    ffi_call_global(ctx.allocator, "ffi.ArraySize", &len_args, &len_result) catch |err| {
        log.err("ffi.ArraySize failed: {s}", .{@errorName(err)});
        return -1;
    };
    const num_inputs: usize = @intCast(len_result.unnamed_1.v_int64);
    log.info("Running {d} candidates", .{num_inputs});

    var results_list = std.ArrayList(c.TVMFFIAny).empty;
    defer results_list.deinit(ctx.allocator);

    for (0..num_inputs) |i| {
        // get RunnerInput[i]
        var input: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        var get_args = [_]c.TVMFFIAny{ inputs_array, any_int(@intCast(i)) };
        ffi_call_global(ctx.allocator, "ffi.ArrayGetItem", &get_args, &input) catch |err| {
            log.err("ffi.ArrayGetItem failed: {s}", .{@errorName(err)});
            return -1;
        };

        // get artifact_path from RunnerInput
        const artifact_path_any = ffi_get_attr(ctx.allocator, input, "artifact_path") catch |err| {
            log.err("get artifact_path failed: {s}", .{@errorName(err)});
            const err_future = createRunnerErrorFuture(ctx.allocator, "failed to get artifact_path") catch return -1;
            results_list.append(ctx.allocator, err_future) catch return -1;
            continue;
        };

        const artifact_path = any_to_string(ctx.allocator, @constCast(&artifact_path_any)) catch |err| {
            log.err("artifact_path to string failed: {s}", .{@errorName(err)});
            const err_future = createRunnerErrorFuture(ctx.allocator, "invalid artifact_path") catch return -1;
            results_list.append(ctx.allocator, err_future) catch return -1;
            continue;
        };
        defer ctx.allocator.free(artifact_path);

        // load module from .so file (artifact_path is the .so path from builder) // NOTE: rephrase to remove parenthetical
        log.debug("Loading module from: {s}", .{artifact_path});
        const loaded_mod = module_load_from_file(ctx.allocator, artifact_path) catch |err| {
            log.err("Failed to load module {s}: {s}", .{ artifact_path, @errorName(err) });
            const err_future = createRunnerErrorFuture(ctx.allocator, "module load failed") catch return -1;
            results_list.append(ctx.allocator, err_future) catch return -1;
            continue;
        };
        defer _ = c.TVMFFIObjectDecRef(loaded_mod);
        log.debug("Loaded module: {*}", .{loaded_mod});

        // get the "main" function from loaded module // NOTE: requires explanation, how do we know "it" is called main, what is main, citation needed.
        const func = module_get_function(ctx.allocator, loaded_mod, "main", true) catch |err| {
            log.err("GetFunction(main) failed: {s}", .{@errorName(err)});
            const err_future = createRunnerErrorFuture(ctx.allocator, "GetFunction failed") catch return -1;
            results_list.append(ctx.allocator, err_future) catch return -1;
            continue;
        };
        defer _ = c.TVMFFIObjectDecRef(func);
        log.debug("Got main function: {*}", .{func});

        // allocate test tensors for matmul: A[M,K], B[K,N], C[M,N]
        const M = ctx.shape.M;
        const N = ctx.shape.N;
        const K = ctx.shape.K;

        const a_data = ctx.allocator.alloc(f32, M * K) catch {
            const err_future = createRunnerErrorFuture(ctx.allocator, "alloc A failed") catch return -1;
            results_list.append(ctx.allocator, err_future) catch return -1;
            continue;
        };
        defer ctx.allocator.free(a_data);

        const b_data = ctx.allocator.alloc(f32, K * N) catch {
            const err_future = createRunnerErrorFuture(ctx.allocator, "alloc B failed") catch return -1;
            results_list.append(ctx.allocator, err_future) catch return -1;
            continue;
        };
        defer ctx.allocator.free(b_data);

        const c_data = ctx.allocator.alloc(f32, M * N) catch {
            const err_future = createRunnerErrorFuture(ctx.allocator, "alloc C failed") catch return -1;
            results_list.append(ctx.allocator, err_future) catch return -1;
            continue;
        };
        defer ctx.allocator.free(c_data);

        // initialize w/ some simple data
        for (a_data, 0..) |*v, idx| v.* = @as(f32, @floatFromInt(idx % 10)) * 0.1;
        for (b_data, 0..) |*v, idx| v.* = @as(f32, @floatFromInt(idx % 10)) * 0.1;
        @memset(c_data, 0);

        // allocate tensors on target device
        const dev_type: i32 = switch (ctx.target_kind) { // NOTE: we switch on `target_kind` quite a few times, discuss considering methods
            .cpu => c.kDLCPU,
            .cuda => c.kDLCUDA,
        };
        var shape_a = [_]i64{ @intCast(M), @intCast(K) };
        var shape_b = [_]i64{ @intCast(K), @intCast(N) };
        var shape_c = [_]i64{ @intCast(M), @intCast(N) };

        const t_a = allocate_tensor(ctx.allocator, a_data, &shape_a, dev_type) catch |err| {
            log.err("allocate_tensor(A) failed: {s}", .{@errorName(err)});
            const err_future = createRunnerErrorFuture(ctx.allocator, "tensor A failed") catch return -1;
            results_list.append(ctx.allocator, err_future) catch return -1;
            continue;
        };
        defer _ = c.TVMFFIObjectDecRef(t_a);

        const t_b = allocate_tensor(ctx.allocator, b_data, &shape_b, dev_type) catch |err| {
            log.err("allocate_tensor(B) failed: {s}", .{@errorName(err)});
            const err_future = createRunnerErrorFuture(ctx.allocator, "tensor B failed") catch return -1;
            results_list.append(ctx.allocator, err_future) catch return -1;
            continue;
        };
        defer _ = c.TVMFFIObjectDecRef(t_b);

        const t_c = allocate_tensor(ctx.allocator, c_data, &shape_c, dev_type) catch |err| {
            log.err("allocate_tensor(C) failed: {s}", .{@errorName(err)});
            const err_future = createRunnerErrorFuture(ctx.allocator, "tensor C failed") catch return -1;
            results_list.append(ctx.allocator, err_future) catch return -1;
            continue;
        };
        defer _ = c.TVMFFIObjectDecRef(t_c);

        // Warmup run
        var call_args = [_]c.TVMFFIAny{
            any_obj(t_a, c.kTVMFFITensor),
            any_obj(t_b, c.kTVMFFITensor),
            any_obj(t_c, c.kTVMFFITensor),
        };
        var call_res: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        ffi_call(ctx.allocator, func, &call_args, &call_res) catch |err| {
            const tvm_err = get_last_error_message(ctx.allocator) catch "?";
            log.err("warmup call failed: {s} — {s}", .{ @errorName(err), tvm_err });
            const err_future = createRunnerErrorFuture(ctx.allocator, "warmup failed") catch return -1;
            results_list.append(ctx.allocator, err_future) catch return -1;
            continue;
        };

        // timed runs (5 iterations, take median) // NOTE: hard coded, intentional? whats the rationale behind this number?
        const num_runs: usize = 5;
        var times: [5]f64 = undefined; // NOTE: audit use of undefined
        for (0..num_runs) |run_idx| {
            const start = std.time.nanoTimestamp();
            ffi_call(ctx.allocator, func, &call_args, &call_res) catch |err| {
                log.err("timed call failed: {s}", .{@errorName(err)});
                const err_future = createRunnerErrorFuture(ctx.allocator, "timed call failed") catch return -1;
                results_list.append(ctx.allocator, err_future) catch return -1;
                break;
            };
            const end = std.time.nanoTimestamp();
            times[run_idx] = @as(f64, @floatFromInt(end - start)) / 1e9;
        }

        // get median
        std.mem.sort(f64, &times, {}, std.sort.asc(f64));
        const run_time_secs = times[num_runs / 2];

        const runner_future = createRunnerSuccessFuture(ctx.allocator, run_time_secs) catch |err| {
            log.err("create runner future failed: {s}", .{@errorName(err)});
            return -1;
        };
        results_list.append(ctx.allocator, runner_future) catch return -1;
        log.debug("Measured candidate {d}: {d:.6}s", .{ i, run_time_secs });
    }

    // create output Array
    var array_result: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    ffi_call_global(ctx.allocator, "ffi.Array", results_list.items, &array_result) catch |err| { // NOTE: we have a few instances of repeated verbose ffi patterns, like this one of creating an array, discuss considering abstraction in ffi module
        log.err("ffi.Array creation failed: {s}", .{@errorName(err)});
        return -1;
    };

    result.* = array_result;
    return 0;
}

/// Context passed to RunnerFuture callbacks.
/// Stores the pre-computed RunnerResult that f_result should return.
const RunnerFutureCtx = struct {
    result: c.TVMFFIAny,
};

/// Callback for RunnerFuture.f_done - always returns true (immediate future). // NOTE: explain
fn zigRunnerFutureDone(
    _: ?*anyopaque,
    _: [*c]const c.TVMFFIAny,
    _: i32,
    result: [*c]c.TVMFFIAny,
) callconv(.c) c_int {
    result[0] = any_bool(true);
    return 0;
}

/// Callback for RunnerFuture.f_result - returns the pre-stored RunnerResult. // NOTE: explain
fn zigRunnerFutureResult(
    data: ?*anyopaque,
    _: [*c]const c.TVMFFIAny,
    _: i32,
    result: [*c]c.TVMFFIAny,
) callconv(.c) c_int {
    const ctx: *RunnerFutureCtx = @ptrCast(@alignCast(data));
    result[0] = ctx.result;
    return 0;
}

/// Deleter for RunnerFutureCtx - called when the TVM function is destroyed. // NOTE: document lifetime and ownership
fn zigRunnerFutureCtxDeleter(data: ?*anyopaque) callconv(.c) void {
    if (data) |ptr| {
        const ctx: *RunnerFutureCtx = @ptrCast(@alignCast(ptr));
        // release stored RunnerResult
        if (ctx.result.unnamed_1.v_obj) |obj| {
            _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
        }
        // free the context (allocated by the global allocator in tune context) // NOTE: rephrase to remove parenthetical
        if (g_tune_ctx) |tune_ctx| {
            tune_ctx.allocator.destroy(ctx);
        }
    }
}

/// Create a RunnerFuture from a RunnerResult with Zig callbacks.
/// RunnerFuture(f_done, f_result) where:
///   - f_done() -> bool (always true for immediate future)
///   - f_result() -> RunnerResult (returns the pre-stored result)
fn createRunnerFuture(allocator: std.mem.Allocator, runner_result: c.TVMFFIAny) !c.TVMFFIAny {
    // allocate context to store the result (will be freed by deleter) // NOTE: rephrase to remove parenthetical
    const ctx = try allocator.create(RunnerFutureCtx);
    ctx.result = runner_result;
    // increment ref count since we arre storing it
    if (runner_result.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectIncRef(@ptrCast(obj));
    }

    // create f_done callback (no context needed, always returns true) // NOTE: rephrase to remove parenthetical
    var f_done: c.TVMFFIObjectHandle = null;
    var ret = c.TVMFFIFunctionCreate(null, zigRunnerFutureDone, null, &f_done);
    if (ret != 0 or f_done == null) {
        allocator.destroy(ctx);
        return error.TvmRuntimeError;
    }
    defer _ = c.TVMFFIObjectDecRef(f_done);

    // create f_result callback w/ context
    var f_result: c.TVMFFIObjectHandle = null;
    ret = c.TVMFFIFunctionCreate(ctx, zigRunnerFutureResult, zigRunnerFutureCtxDeleter, &f_result);
    if (ret != 0 or f_result == null) {
        allocator.destroy(ctx);
        return error.TvmRuntimeError;
    }
    defer _ = c.TVMFFIObjectDecRef(f_result);

    // create RunnerFuture(f_done, f_result)
    var future: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    var args = [_]c.TVMFFIAny{
        any_obj(f_done, c.kTVMFFIFunction),
        any_obj(f_result, c.kTVMFFIFunction),
    };
    try ffi_call_global(allocator, "meta_schedule.RunnerFuture", &args, &future);
    return future;
}

/// create a RunnerFuture representing an error.
fn createRunnerErrorFuture(allocator: std.mem.Allocator, err_msg: []const u8) !c.TVMFFIAny {
    const msg_buf = try cstr_alloc(allocator, err_msg);
    defer allocator.free(msg_buf);

    // create RunnerResult with error (run_secs=None, error_msg=msg)
    var runner_result: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    var rr_args = [_]c.TVMFFIAny{ any_none(), any_raw_str(cstr_ptr(msg_buf)) };
    try ffi_call_global(allocator, "meta_schedule.RunnerResult", &rr_args, &runner_result);
    defer if (runner_result.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    return createRunnerFuture(allocator, runner_result);
}

/// Create a RunnerFuture representing a successful measurement.
fn createRunnerSuccessFuture(allocator: std.mem.Allocator, run_secs: f64) !c.TVMFFIAny {
    // create Array of run times (single measurement) // NOTE: elaborate and rephrase to remove parenthetical
    var times_array: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    var times_args = [_]c.TVMFFIAny{any_float(run_secs)};
    try ffi_call_global(allocator, "ffi.Array", &times_args, &times_array);
    defer if (times_array.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    // create RunnerResult(run_secs=times, error_msg=None)
    var runner_result: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    var rr_args = [_]c.TVMFFIAny{ times_array, any_none() };
    try ffi_call_global(allocator, "meta_schedule.RunnerResult", &rr_args, &runner_result);
    defer if (runner_result.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    return createRunnerFuture(allocator, runner_result);
}

/// Build a TIR matmul module for autotuning.
///
/// Creates: C[M,N] = A[M,K] @ B[K,N]
/// Uses the same TE API pattern as build_and_run_vec_add but for matmul.
pub fn build_matmul_tir(allocator: std.mem.Allocator, M: usize, N: usize, K: usize) !c.TVMFFIAny {
    if (!build_options.enable_tvm) {
        return error.TvmDisabled;
    }
    const log = std.log.scoped(.@"zg/tvm_matmul");

    try ensureTvmCompilerLoaded(allocator);

    const m_i64: i64 = @intCast(M);
    const n_i64: i64 = @intCast(N);
    const k_i64: i64 = @intCast(K);

    // create shapes for A[M,K], B[K,N], C[M,N]
    var shape_a: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{ any_int(m_i64), any_int(k_i64) };
        try ffi_call_global(allocator, "ffi.Array", &args, &shape_a);
    }
    defer if (shape_a.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    var shape_b: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{ any_int(k_i64), any_int(n_i64) };
        try ffi_call_global(allocator, "ffi.Array", &args, &shape_b);
    }
    defer if (shape_b.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    var shape_c: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{ any_int(m_i64), any_int(n_i64) };
        try ffi_call_global(allocator, "ffi.Array", &args, &shape_c);
    }
    defer if (shape_c.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    // create placeholder tensors
    const dtype_buf = try cstr_alloc(allocator, "float32");
    defer allocator.free(dtype_buf);
    const name_a_buf = try cstr_alloc(allocator, "A");
    defer allocator.free(name_a_buf);
    const name_b_buf = try cstr_alloc(allocator, "B");
    defer allocator.free(name_b_buf);

    var tensor_a: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{
            shape_a,
            any_raw_str(cstr_ptr(dtype_buf)),
            any_raw_str(cstr_ptr(name_a_buf)),
        };
        try ffi_call_global(allocator, "te.Placeholder", &args, &tensor_a);
    }
    log.debug("Created placeholder A[{d},{d}]", .{ M, K });

    var tensor_b: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{
            shape_b,
            any_raw_str(cstr_ptr(dtype_buf)),
            any_raw_str(cstr_ptr(name_b_buf)),
        };
        try ffi_call_global(allocator, "te.Placeholder", &args, &tensor_b);
    }
    log.debug("Created placeholder B[{d},{d}]", .{ K, N });

    // Use te.compute w/ a reduction to create matmul.
    // The TE API requires creating a fcompute lambda, which is tricky via FFI. // NOTE: is this limitation tracked? any prospective paths forward? must discuss implications before we can make a ruling on TVM integration into Zigrad
    // Instead, we can use topi.nn.matmul which is a pre-built TE schedule.

    // try using topi.matmul(A, B, transpose_a=False, transpose_b=False) fallback to topi.nn.matmul
    var tensor_c: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{
            tensor_a,
            tensor_b,
            any_bool(false), // transpose_a
            any_bool(false), // transpose_b
        };
        ffi_call_global(allocator, "topi.matmul", &args, &tensor_c) catch |err1| {
            log.warn("topi.matmul failed ({s}), trying topi.nn.matmul", .{@errorName(err1)});
            // fallback to topi.nn.matmul
            var fallback_args = [_]c.TVMFFIAny{ tensor_a, tensor_b };
            ffi_call_global(allocator, "topi.nn.matmul", &fallback_args, &tensor_c) catch |err2| {
                log.err("topi.nn.matmul also failed: {s}", .{@errorName(err2)});
                return err2;
            };
        };
    }
    log.info("Created matmul C[{d},{d}] = A[{d},{d}] @ B[{d},{d}]", .{ M, N, M, K, K, N });

    // create PrimFunc from tensors
    var tensors_array: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{ tensor_a, tensor_b, tensor_c };
        try ffi_call_global(allocator, "ffi.Array", &args, &tensors_array);
    }

    var prim_func: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{ tensors_array, any_none() };
        try ffi_call_global(allocator, "te.CreatePrimFunc", &args, &prim_func);
    }

    // add global_symbol attribute
    const global_symbol_buf = try cstr_alloc(allocator, "global_symbol");
    defer allocator.free(global_symbol_buf);
    const main_name_buf = try cstr_alloc(allocator, "main");
    defer allocator.free(main_name_buf);

    var prim_func_with_attr: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{
            prim_func,
            any_raw_str(cstr_ptr(global_symbol_buf)),
            any_raw_str(cstr_ptr(main_name_buf)),
        };
        try ffi_call_global(allocator, "ir.BaseFuncWithAttr", &args, &prim_func_with_attr);
    }
    if (prim_func.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    }

    // wrap in IRModule // NOTE: this is very verbose and non-trivial, how often do we do this? discuss considering abstracting into zig ffi module
    const main_buf = try cstr_alloc(allocator, "main");
    defer allocator.free(main_buf);

    var global_var: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{any_raw_str(cstr_ptr(main_buf))};
        try ffi_call_global(allocator, "ir.GlobalVar", &args, &global_var);
    }

    var func_map: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{ global_var, prim_func_with_attr };
        try ffi_call_global(allocator, "ffi.Map", &args, &func_map);
    }

    var empty_map: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    try ffi_call_global(allocator, "ffi.Map", &.{}, &empty_map);

    var ir_mod: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{ func_map, any_none(), empty_map };
        try ffi_call_global(allocator, "ir.IRModule", &args, &ir_mod);
    }

    log.info("Created matmul IRModule", .{});
    return ir_mod;
}

/// Run TVM MetaSchedule autotuning on a TIR module.
///
/// This function orchestrates the tuning process:
/// 1. Creates MetaSchedule components (space generator, search strategy, database)
/// 2. Registers Zig builder/runner callbacks
/// 3. Runs the tuning loop
/// 4. Saves best schedule to the database
pub fn tune(allocator: std.mem.Allocator, ir_mod: c.TVMFFIAny, target_kind: TargetKind, shape: MatmulShape, opts: TuneOpts) !void {
    if (!build_options.enable_tvm) {
        return error.TvmDisabled;
    }
    const log = std.log.scoped(.@"zg/tvm_tune");

    try ensureTvmCompilerLoaded(allocator);

    // load CUDA intrinsics if targeting CUDA
    //   these are required for MetaSchedule CUDA schedule rules (WMMA, MMA, etc.)
    if (target_kind == .cuda) {
        try tvm_cuda.load_intrinsics(allocator);
    }

    // Per-target work directory to prevent database contamination across targets.
    //   CPU traces use SSRSRS tiling + parallel/vectorize which CUDA postprocessors reject.
    const target_suffix: []const u8 = switch (target_kind) {
        .cpu => "cpu",
        .cuda => "cuda",
    };
    const work_dir = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ opts.work_dir, target_suffix });
    defer allocator.free(work_dir);

    log.info("Starting MetaSchedule tuning", .{});
    log.info("  work_dir: {s}", .{work_dir});
    log.info("  max_trials: {d}", .{opts.max_trials});
    log.info("  trials_per_iter: {d}", .{opts.trials_per_iter});

    // create work dir
    std.fs.cwd().makePath(work_dir) catch |err| {
        if (err != error.PathAlreadyExists) {
            log.err("failed to create work_dir: {s}", .{@errorName(err)});
            return err;
        }
    };

    // create target with num-cores for LLVM (required by MetaSchedule)
    const cpu_count: usize = std.Thread.getCpuCount() catch 4;
    const target_str = switch (target_kind) {
        .cpu => try std.fmt.allocPrint(allocator, "llvm -num-cores {d}", .{cpu_count}),
        .cuda => try allocator.dupe(u8, "cuda -arch=sm_86 -max_threads_per_block=1024 -max_shared_memory_per_block=49152"),
    };
    defer allocator.free(target_str);
    const target_buf = try cstr_alloc(allocator, target_str);
    defer allocator.free(target_buf);

    var target: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        // create device target
        var device_target: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        var args = [_]c.TVMFFIAny{any_raw_str(cstr_ptr(target_buf))};
        try ffi_call_global(allocator, "target.Target", &args, &device_target);

        // create host target (always LLVM for CPU codegen)
        const host_str = try cstr_alloc(allocator, "llvm");
        defer allocator.free(host_str);
        var host_target: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        var host_args = [_]c.TVMFFIAny{any_raw_str(cstr_ptr(host_str))};
        ffi_call_global(allocator, "target.Target", &host_args, &host_target) catch |err| { // NOTE: repeated pattern
            if (device_target.unnamed_1.v_obj) |obj| {
                _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
            }
            log.err("target.Target(host) failed: {s}", .{@errorName(err)});
            return err;
        };
        defer if (host_target.unnamed_1.v_obj) |obj| {
            _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
        };

        // Set host on target - CRITICAL for MakePackedAPI to work // NOTE: poor comment. remove all caps "CRITICAL", rephrase, explain, needs citation.
        var with_host_args = [_]c.TVMFFIAny{ device_target, host_target };
        ffi_call_global(allocator, "target.WithHost", &with_host_args, &target) catch |err| {
            if (device_target.unnamed_1.v_obj) |obj| {
                _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
            }
            log.err("target.WithHost failed: {s}", .{@errorName(err)});
            return err;
        };

        // device_target was consumed by WithHost, release it // NOTE: do we have an understanding of memory contract or is this a guess? comment should clarify ownership and lifetime and explain what we know and how we know it when possible.
        if (device_target.unnamed_1.v_obj) |obj| {
            _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
        }
    }
    log.info("Created target with host: {s}", .{target_str});

    // set up global tune context for callbacks
    var tune_ctx = TuneContext.init(allocator, target, target_kind, work_dir, shape);
    defer tune_ctx.deinit();
    g_tune_ctx = &tune_ctx;
    defer g_tune_ctx = null;

    // Register required helper functions that MetaSchedule expects from python. // NOTE: "required" is redundant as comment clearly states TVM "expects" this.
    // The C++ code looks for packed functions in the global registry.
    // python registers "meta_schedule.cpu_count" but C++ may look for "_cpu_count" variant.
    // We register both variants for compatibility. // NOTE: do we have an understanding? is this a guess? "C++ may look for "_cpu_count" variant"" sounds like speculation, requires clarification.
    const cpu_count_cb = struct {
        /// Returns number of CPUs available for parallel builds // NOTE: explain, are trials built in parallel? is this information applicable to or impact the kernel itself? I assume TVM needs to (or would want to) know what resources are available for kernel gen.
        fn f(_: ?*anyopaque, args: [*c]const c.TVMFFIAny, num_args: i32, result: [*c]c.TVMFFIAny) callconv(.c) c_int {
            // _cpu_count(logical: bool = True) -> int
            _ = num_args;
            _ = args;
            const ncpus: i64 = @intCast(std.Thread.getCpuCount() catch 4);
            result.* = any_int(ncpus);
            return 0;
        }
    }.f;

    var cpu_count_func: c.TVMFFIObjectHandle = null;
    if (c.TVMFFIFunctionCreate(null, cpu_count_cb, null, &cpu_count_func) != 0) {
        log.err("TVMFFIFunctionCreate(cpu_count) failed", .{});
        return error.TvmRuntimeError;
    }
    defer _ = c.TVMFFIObjectDecRef(cpu_count_func);

    // register both name variants that TVM might look for // NOTE: same uncertainty comment as above
    const func_names = [_][]const u8{
        "meta_schedule._cpu_count",
        "meta_schedule.cpu_count",
    };
    for (func_names) |func_name_str| {
        // Use the original string directly, NOT cstr_alloc which adds null terminator to length // NOTE: remove all caps "NOT" and provide justification
        // TVMFFIByteArray size should NOT include the null terminator
        var name_arr: c.TVMFFIByteArray = .{ .data = func_name_str.ptr, .size = func_name_str.len };
        // use override=1 to replace any existing registration // NOTE: rephrase, sounds like author is instructing the user on how to use something, not declaring what is happening
        if (c.TVMFFIFunctionSetGlobal(&name_arr, cpu_count_func, 1) != 0) {
            const msg = try get_last_error_message(allocator);
            defer allocator.free(msg);
            log.warn("TVMFFIFunctionSetGlobal({s}) failed: {s}", .{ func_name_str, msg });
            // continue to try other names // NOTE: control flow unclear, loop/block requires better documentation
        } else {
            log.debug("Registered {s}", .{func_name_str});

            // verify registration by immediately trying to retrieve the function
            var retrieved: c.TVMFFIObjectHandle = null;
            if (c.TVMFFIFunctionGetGlobal(&name_arr, &retrieved) != 0) {
                log.warn("VERIFICATION FAILED: Could not retrieve {s} right after registration.", .{func_name_str});
            } else if (retrieved == null) {
                log.warn("VERIFICATION FAILED: Retrieved null for {s}", .{func_name_str});
            } else {
                log.info("VERIFIED: {s} is retrievable via TVMFFIFunctionGetGlobal", .{func_name_str});
                // clean up retrieved handle
                _ = c.TVMFFIObjectDecRef(retrieved);
            }
        }
    }

    // verify functions are still retrievable after registration loop using ffi_get_global // NOTE: looks redundant, its okay to be sure but intent should be explained as repetition is a code smell, looks like debug code. also noisy logs.
    // This uses the same path as ffi_call_global will use
    log.debug("Verifying functions via ffi_get_global...", .{});
    for (func_names) |func_name_str| {
        if (ffi_get_global(allocator, func_name_str)) |handle| {
            log.info("POST-LOOP CHECK OK: {s} retrievable via ffi_get_global", .{func_name_str}); // NOTE: remove all caps, unprofessional, this goes for other locations as well.
            _ = c.TVMFFIObjectDecRef(handle);
        } else |_| {
            log.warn("POST-LOOP CHECK FAILED: {s} not found via ffi_get_global", .{func_name_str});
        }
    }

    // Test calling the registered function through our FFI mechanism
    log.debug("Testing function call through ffi_call_global...", .{});
    {
        var cpu_result: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
        var call_args = [_]c.TVMFFIAny{any_bool(true)};
        if (ffi_call_global(allocator, "meta_schedule.cpu_count", &call_args, &cpu_result)) {
            if (cpu_result.type_index == c.kTVMFFIInt) {
                log.info("TEST CALL SUCCESS: meta_schedule.cpu_count returned {d}", .{cpu_result.unnamed_1.v_int64});
            } else {
                log.warn("TEST CALL: unexpected return type {d}", .{cpu_result.type_index});
            }
        } else |err| {
            log.warn("TEST CALL FAILED: meta_schedule.cpu_count: {s}", .{@errorName(err)});
        }
    }

    // Create MetaSchedule components
    //
    // MetaSchedule's SpaceGeneratorPostOrderApply and other components require
    // complex callbacks that are typically provided by python wrappers.
    // The FFI signature expects:
    //   SpaceGeneratorPostOrderApply(
    //     ffi.Function,                             -- schedule rule generator
    //     Optional<Array<ScheduleRule>>,            -- explicit rules
    //     Optional<Array<Postproc>>,                -- post-processors
    //     Optional<Map<Mutator, FloatImm>>          -- mutators
    //   )
    //
    // For now, we use the default schedule rules by passing None/empty values. // NOTE: is this still true? requires explanation, justification, and citation. is this a limitation we are tracking?

    // 1. Space generator - generates candidate schedules
    //   use ScheduleRuleDefault{LLVM,CUDA}() which returns default rules for the target kind
    var schedule_rules: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        const ffi_func_name = switch (target_kind) {
            .cpu => "meta_schedule.ScheduleRuleDefaultLLVM",
            .cuda => "meta_schedule.ScheduleRuleDefaultCUDA",
        };
        try ffi_call_global(allocator, ffi_func_name, &.{}, &schedule_rules);
    }
    log.debug("Created ScheduleRules for {s}", .{@tagName(target_kind)});

    var space_gen: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        // SpaceGeneratorPostOrderApply(sch_rules, postprocs, mutator_probs)
        //   pass None for f_block_filter, and explicit arrays for others // NOTE: rephrase, sounds like author is instructing the user on how to use something, not declaring what is happening
        var args = [_]c.TVMFFIAny{
            any_none(), // f_block_filter - use default
            schedule_rules, // sch_rules
            any_none(), // postprocs - use default
            any_none(), // mutator_probs - use default
        };
        try ffi_call_global(allocator, "meta_schedule.SpaceGeneratorPostOrderApply", &args, &space_gen);
    }
    log.debug("Created SpaceGenerator", .{});

    // 2. Search strategy - evolutionary search  // NOTE: Good comment block!
    // EvolutionarySearch(
    //   population_size: int,         -- population for evolutionary algorithm
    //   init_measured_ratio: float,   -- ratio of initial measured samples
    //   init_min_unmeasured: int,     -- min unmeasured samples in init
    //   max_fail_count: int,          -- max consecutive failures before stop
    //   genetic_num_iters: int,       -- number of genetic iterations
    //   genetic_mutate_prob: float,   -- probability of mutation
    //   genetic_max_fail_count: int,  -- max failures in genetic phase
    //   eps_greedy: float             -- epsilon for epsilon-greedy exploration
    // )
    var search_strategy: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{
            any_int(512), // population_size
            any_float(0.2), // init_measured_ratio
            any_int(50), // init_min_unmeasured
            any_int(5), // max_fail_count
            any_int(3), // genetic_num_iters
            any_float(0.85), // genetic_mutate_prob
            any_int(10), // genetic_max_fail_count
            any_float(0.05), // eps_greedy
        };
        try ffi_call_global(allocator, "meta_schedule.SearchStrategyEvolutionarySearch", &args, &search_strategy);
    }
    log.debug("Created SearchStrategy", .{});

    // 3. Database - stores tuning records
    const workload_path = try std.fmt.allocPrint(allocator, "{s}/workload.json", .{work_dir});
    defer allocator.free(workload_path);
    const record_path = try std.fmt.allocPrint(allocator, "{s}/tuning_record.json", .{work_dir});
    defer allocator.free(record_path);

    const workload_buf = try cstr_alloc(allocator, workload_path);
    defer allocator.free(workload_buf);
    const record_buf = try cstr_alloc(allocator, record_path);
    defer allocator.free(record_buf);
    const structural_buf = try cstr_alloc(allocator, "structural");
    defer allocator.free(structural_buf);

    var database: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{
            any_raw_str(cstr_ptr(workload_buf)),
            any_raw_str(cstr_ptr(record_buf)),
            any_bool(true), // allow_missing
            any_raw_str(cstr_ptr(structural_buf)),
        };
        try ffi_call_global(allocator, "meta_schedule.DatabaseJSONDatabase", &args, &database);
    }
    log.debug("Created JSONDatabase", .{});

    // 4. Create TuneContext
    // TuneContext(
    //   mod: Optional<IRModule>,
    //   target: Optional<Target>,
    //   space_generator: Optional<SpaceGenerator>,
    //   search_strategy: Optional<SearchStrategy>,
    //   task_name: Optional<str>,
    //   num_threads: int,
    //   rand_state: int,  -- random seed
    //   logger: ffi.Function  -- logging callback
    // )
    const main_buf = try cstr_alloc(allocator, "main");
    defer allocator.free(main_buf);

    // create a no-op logger callback // NOTE: this does not obviously log anything, clarify
    const logger_noop = struct {
        fn f(_: ?*anyopaque, _: [*c]const c.TVMFFIAny, _: i32, result: [*c]c.TVMFFIAny) callconv(.c) c_int {
            result.* = any_none();
            return 0;
        }
    }.f;
    var logger_func: c.TVMFFIObjectHandle = null;
    if (c.TVMFFIFunctionCreate(null, logger_noop, null, &logger_func) != 0) {
        log.err("TVMFFIFunctionCreate(logger) failed", .{});
        return error.TvmRuntimeError;
    }
    defer _ = c.TVMFFIObjectDecRef(logger_func);

    var tune_context: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{
            ir_mod, // mod
            target, // target
            space_gen, // space_generator
            search_strategy, // search_strategy
            any_raw_str(cstr_ptr(main_buf)), // task_name
            any_int(1), // num_threads
            any_int(42), // rand_state (seed)
            any_obj(logger_func, c.kTVMFFIFunction), // logger
        };
        try ffi_call_global(allocator, "meta_schedule.TuneContext", &args, &tune_context);
    }
    log.debug("Created TuneContext", .{});

    // 5. Register Zig builder callback // NOTE: builder of what? vague.
    var builder_func: c.TVMFFIObjectHandle = null;
    if (c.TVMFFIFunctionCreate(null, zigBuildCallback, null, &builder_func) != 0) {
        log.err("TVMFFIFunctionCreate(builder) failed", .{});
        return error.TvmRuntimeError;
    }
    defer _ = c.TVMFFIObjectDecRef(builder_func);

    var builder: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{any_obj(builder_func, c.kTVMFFIFunction)};
        try ffi_call_global(allocator, "meta_schedule.BuilderPyBuilder", &args, &builder);
    }
    log.debug("Created PyBuilder with Zig callback", .{});

    // 6. Register Zig runner callback
    var runner_func: c.TVMFFIObjectHandle = null;
    if (c.TVMFFIFunctionCreate(null, zigRunCallback, null, &runner_func) != 0) {
        log.err("TVMFFIFunctionCreate(runner) failed", .{});
        return error.TvmRuntimeError;
    }
    defer _ = c.TVMFFIObjectDecRef(runner_func);

    var runner: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{any_obj(runner_func, c.kTVMFFIFunction)};
        try ffi_call_global(allocator, "meta_schedule.RunnerPyRunner", &args, &runner);
    }
    log.debug("Created PyRunner with Zig callback", .{});

    // 7. Create cost model using PyCostModel with Zig callbacks
    // PyCostModel(f_load, f_save, f_update, f_predict, f_as_string)
    // We implement a simple random cost model that returns random scores. // NOTE: I understand the rationale, but it should be justified here.

    // no-op callbacks for load/save // NOTE: this doesnt obviously load/save anything, require explanation
    const noop_cb = struct {
        fn f(_: ?*anyopaque, _: [*c]const c.TVMFFIAny, _: i32, result: [*c]c.TVMFFIAny) callconv(.c) c_int {
            result.* = any_none();
            return 0;
        }
    }.f;

    // update callback - no-op for random model // NOTE: clarify intended role. it is clear what we are doing (nothing) it is not clear what the TVM semantics are (ie what is TVM calling this for?).
    var update_func: c.TVMFFIObjectHandle = null;
    if (c.TVMFFIFunctionCreate(null, noop_cb, null, &update_func) != 0) {
        return error.TvmRuntimeError;
    }
    defer _ = c.TVMFFIObjectDecRef(update_func);

    // Predict callback - returns random scores
    // f_predict(context, candidates, return_ptr) -> None
    // return_ptr is a pointer to a double array where we write scores // NOTE: ambiguous, could mean an array of doubles, or double pointer, etc.
    const predict_cb = struct {
        fn f(_: ?*anyopaque, args: [*c]const c.TVMFFIAny, num_args: i32, result: [*c]c.TVMFFIAny) callconv(.c) c_int {
            const predict_log = std.log.scoped(.@"zg/tvm_predict");

            if (num_args < 3) {
                predict_log.err("predict callback: expected 3 args, got {d}", .{num_args});
                result.* = any_none();
                return -1;
            }

            // args[0] = TuneContext, args[1] = Array<MeasureCandidate>, args[2] = void* (double*)
            const candidates = args[1];
            const return_ptr = args[2];

            // get number of candidates
            var n: usize = 0;
            if (candidates.type_index >= c.kTVMFFIStaticObjectBegin and candidates.unnamed_1.v_obj != null) {
                var len_result: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
                var len_args = [_]c.TVMFFIAny{candidates};
                if (ffi_call_global_noerr("ffi.ArraySize", &len_args, &len_result)) |err_msg| {
                    predict_log.err("ffi.ArraySize failed: {s}", .{err_msg});
                    result.* = any_none();
                    return -1;
                }
                if (len_result.type_index != c.kTVMFFIInt) {
                    predict_log.err("ffi.ArraySize returned unexpected type_index={d}", .{len_result.type_index});
                    result.* = any_none();
                    return -1;
                }
                n = @intCast(len_result.unnamed_1.v_int64);
            } else {
                predict_log.err("candidates is not a TVM object (type_index={d})", .{candidates.type_index});
                result.* = any_none();
                return -1;
            }

            predict_log.debug("scoring {d} candidates", .{n});

            // write random scores to the pre-allocated double* buffer
            if (return_ptr.type_index == c.kTVMFFIOpaquePtr and return_ptr.unnamed_1.v_ptr != null) {
                const scores: [*]f64 = @ptrCast(@alignCast(return_ptr.unnamed_1.v_ptr));
                var prng = std.Random.DefaultPrng.init(42);
                for (0..n) |i| {
                    scores[i] = prng.random().float(f64);
                }
            } else {
                predict_log.err("predict callback: return_ptr is not an opaque pointer (type_index={d})", .{return_ptr.type_index});
                result.* = any_none();
                return -1;
            }

            result.* = any_none();
            return 0;
        }

        /// Call a TVM global function without requiring an allocator.
        /// Returns null on success, or a static error description on failure.
        ///
        /// TVM's SafeCall checks `result->type_index < kTVMFFIStaticObjectBegin`
        /// before executing — the output must be pre-initialized to a POD/None value,
        /// otherwise uninitialized stack memory can trigger a spurious CHECK failure.
        fn ffi_call_global_noerr(name: []const u8, args: []c.TVMFFIAny, out: *c.TVMFFIAny) ?[]const u8 {
            out.* = any_none();
            var name_arr: c.TVMFFIByteArray = .{ .data = name.ptr, .size = name.len };
            var func_handle: c.TVMFFIObjectHandle = null;
            if (c.TVMFFIFunctionGetGlobal(&name_arr, &func_handle) != 0 or func_handle == null) {
                return "function not found in global registry";
            }
            defer _ = c.TVMFFIObjectDecRef(func_handle);

            if (c.TVMFFIFunctionCall(func_handle, args.ptr, @intCast(args.len), out) != 0) {
                return "TVMFFIFunctionCall returned error";
            }
            return null;
        }
    }.f;

    var predict_func: c.TVMFFIObjectHandle = null;
    if (c.TVMFFIFunctionCreate(null, predict_cb, null, &predict_func) != 0) {
        return error.TvmRuntimeError;
    }
    defer _ = c.TVMFFIObjectDecRef(predict_func);

    // as_string callback - returns model name
    const as_string_cb = struct {
        fn f(_: ?*anyopaque, _: [*c]const c.TVMFFIAny, _: i32, result: [*c]c.TVMFFIAny) callconv(.c) c_int {
            // Return a string "ZigRandomModel"
            const name = "ZigRandomModel";
            var name_arr: c.TVMFFIByteArray = .{ .data = name.ptr, .size = name.len };
            var str_obj: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
            if (c.TVMFFIStringFromByteArray(&name_arr, &str_obj) != 0) {
                result.* = any_none();
                return -1;
            }
            result.* = str_obj;
            return 0;
        }
    }.f;

    var as_string_func: c.TVMFFIObjectHandle = null;
    if (c.TVMFFIFunctionCreate(null, as_string_cb, null, &as_string_func) != 0) {
        return error.TvmRuntimeError;
    }
    defer _ = c.TVMFFIObjectDecRef(as_string_func);

    // create PyCostModel
    var cost_model: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{
            any_obj(update_func, c.kTVMFFIFunction), // f_load (reuse noop)
            any_obj(update_func, c.kTVMFFIFunction), // f_save (reuse noop)
            any_obj(update_func, c.kTVMFFIFunction), // f_update
            any_obj(predict_func, c.kTVMFFIFunction), // f_predict
            any_obj(as_string_func, c.kTVMFFIFunction), // f_as_string
        };
        try ffi_call_global(allocator, "meta_schedule.CostModelPyCostModel", &args, &cost_model);
    }
    log.debug("Created PyCostModel with Zig random predictor", .{});

    // 8. Create task scheduler
    // TaskSchedulerGradientBased(
    //   f_logging: ffi.Function,  -- logging callback
    //   alpha: float,             -- gradient weight
    //   window_size: int,         -- window for gradient estimation
    //   seed: int                 -- random seed
    // )
    var task_scheduler: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        // Use the same logger as TuneContext
        var args = [_]c.TVMFFIAny{
            any_obj(logger_func, c.kTVMFFIFunction), // f_logging
            any_float(0.8), // alpha
            any_int(3), // window_size
            any_int(42), // seed
        };
        try ffi_call_global(allocator, "meta_schedule.TaskSchedulerGradientBased", &args, &task_scheduler);
    }
    log.debug("Created TaskScheduler", .{});

    // 8. Run tuning
    log.info("Starting tuning with {d} max trials...", .{opts.max_trials});

    // create contexts array (single task) // NOTE: lacking context, what is single task? what is a task? why only a single task?
    var contexts_array: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{tune_context};
        try ffi_call_global(allocator, "ffi.Array", &args, &contexts_array);
    }

    // create weights array // NOTE: lacking context, what are "weights" is this a TVM concept?
    var weights_array: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{any_float(1.0)};
        try ffi_call_global(allocator, "ffi.Array", &args, &weights_array);
    }

    // Create AddToDatabase callback - this writes tuning records to the JSON database.
    //  We only use AddToDatabase, not the full default set (which includes RemoveBuildArtifact
    //   and UpdateCostModel that require python helpers we havent registered). // NOTE: requires brief explanation/justification
    var add_to_db_callback: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    try ffi_call_global(allocator, "meta_schedule.MeasureCallbackAddToDatabase", &.{}, &add_to_db_callback);
    log.debug("Created AddToDatabase callback", .{});

    var callbacks_array: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{add_to_db_callback};
        try ffi_call_global(allocator, "ffi.Array", &args, &callbacks_array);
    }

    // run TaskScheduler.Tune // NOTE: requires documentation on what happens here and how it works
    var tune_result: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        var args = [_]c.TVMFFIAny{
            task_scheduler,
            contexts_array,
            weights_array,
            any_int(@intCast(opts.max_trials)),
            any_int(@intCast(opts.max_trials)),
            any_int(@intCast(opts.trials_per_iter)),
            builder,
            runner,
            callbacks_array,
            database,
            cost_model, // RandomModel cost model
        };
        ffi_call_global(allocator, "meta_schedule.TaskSchedulerTune", &args, &tune_result) catch |err| {
            // MetaSchedule requires additional python-registered functions. // NOTE: this doesnt make sense, implies this path will always fail, is that true? explain
            //  The core callback infrastructure works (Builder/Runner registered),
            //   but full tuning requires python for _cpu_count and other helpers.
            log.err("TaskSchedulerTune failed: {s}", .{@errorName(err)});
            log.err("MetaSchedule tuning currently requires python for helper functions.", .{});
            log.err("Use: task python -- scripts/tvm_autotune.py matmul --shape={s}x{s}x{s}", .{
                "M", "N", "K",
            });
            return err;
        };
    }

    log.info("Tuning complete.", .{});
    log.info("Results saved to:", .{});
    log.info("  Workloads: {s}", .{workload_path});
    log.info("  Records: {s}", .{record_path});
}

// ============================================================================
// Best Schedule Replay: Load and run tuned kernels
// ============================================================================

/// Options for loading a tuned module.
pub const LoadTunedOpts = struct {
    /// Work directory containing tuning records and compiled .so files. // NOTE: confusing phrasing, rephrase.
    ///  Tuning stores results in per-target subdirs (e.g. `artifacts/tvm_cache/cpu/`).
    work_dir: []const u8 = "artifacts/tvm_cache/cpu",
};

/// Result of loading a tuned module.
pub const TunedModule = struct {
    module_handle: c.TVMFFIObjectHandle, // NOTE: we made a zig wrapper type, we should consider using zig types, ie optional fields being zig optional types is safer and carries clearer semantics
    main_func: c.TVMFFIObjectHandle, // NOTE: we made a zig wrapper type, we should consider using zig types, ie optional fields being zig optional types is safer and carries clearer semantics
    best_candidate: usize,
    best_time_us: f64,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *TunedModule) void {
        if (self.main_func) |f| _ = c.TVMFFIObjectDecRef(f);
        if (self.module_handle) |m| _ = c.TVMFFIObjectDecRef(m);
        self.* = undefined;
    }
};

/// Parse tuning_record.json and find the best candidate index.
/// Returns (best_index, best_time_seconds) or error if no records found.
fn find_best_candidate(allocator: std.mem.Allocator, record_path: []const u8) !struct { usize, f64 } {
    const log = std.log.scoped(.@"zg/tvm_loader");

    const file = std.fs.cwd().openFile(record_path, .{}) catch |err| {
        log.err("Failed to open tuning records at {s}: {s}", .{ record_path, @errorName(err) });
        return error.NoTuningRecords;
    };
    defer file.close();

    // read entire file (tuning records are typically small) // NOTE: rephrase to remove parenthetical
    const file_size = try file.getEndPos();
    if (file_size == 0) {
        log.err("Empty tuning records file: {s}", .{record_path});
        return error.NoTuningRecords;
    }
    const contents = try allocator.alloc(u8, file_size);
    defer allocator.free(contents);
    const bytes_read = try file.readAll(contents);

    var best_idx: usize = 0;
    var best_time: f64 = std.math.inf(f64);
    var line_num: usize = 0;

    // split by newlines
    var lines = std.mem.splitScalar(u8, contents[0..bytes_read], '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;

        // Parse json to extract run_secs // NOTE: explain or state why we arent using std lib for json
        // Format: [workload_id, [[trace, decisions], [run_secs], target, args]]
        // The run_secs comes after the decisions array closes: ]],[run_secs],{
        // Look for pattern "]],[" followed by a float (not integer like tile sizes)
        var run_secs: ?f64 = null;
        var i: usize = 0;
        while (i + 10 < line.len) : (i += 1) {
            // Look for pattern "]],[" (double close bracket) followed by a digit
            if (i + 4 < line.len and
                line[i] == ']' and line[i + 1] == ']' and
                line[i + 2] == ',' and line[i + 3] == '[')
            {
                const start = i + 4;
                // Check if next char is a digit (floats start with digit, e.g., 1.14e-05)
                if (start < line.len and std.ascii.isDigit(line[start])) {
                    // Find end of number (until ])
                    var end = start;
                    while (end < line.len and line[end] != ']') : (end += 1) {}
                    if (end > start) {
                        const num_str = line[start..end];
                        // Only accept if it looks like a float (contains 'e' or '.')
                        if (std.mem.indexOfScalar(u8, num_str, 'e') != null or
                            std.mem.indexOfScalar(u8, num_str, '.') != null)
                        {
                            run_secs = std.fmt.parseFloat(f64, num_str) catch null;
                            if (run_secs != null) break;
                        }
                    }
                }
            }
        }

        if (run_secs) |t| {
            log.debug("Record {d}: {d:.2} µs", .{ line_num, t * 1e6 });
            if (t < best_time) {
                best_time = t;
                best_idx = line_num;
            }
        } else {
            log.warn("Could not parse run_secs from record {d}", .{line_num});
        }

        line_num += 1;
    }

    if (line_num == 0) {
        log.err("No tuning records found in {s}", .{record_path});
        return error.NoTuningRecords;
    }

    log.info("Best candidate: {d} ({d:.2} µs)", .{ best_idx, best_time * 1e6 });
    return .{ best_idx, best_time };
}

/// Load the best tuned module from a previous tuning run.
///
/// Parses tuning_record.json to find the fastest candidate, then loads
/// the corresponding .so file and returns a handle to the main function.
pub fn load_tuned_module(allocator: std.mem.Allocator, opts: LoadTunedOpts) !TunedModule {
    const log = std.log.scoped(.@"zg/tvm_loader");

    // TVM runtime needs to be initialized since we need full compiler module execution
    try ensureTvmCompilerLoaded(allocator);

    // find best candidate from tuning records
    const record_path = try std.fmt.allocPrint(allocator, "{s}/tuning_record.json", .{opts.work_dir});
    defer allocator.free(record_path);

    const best_idx, const best_time = try find_best_candidate(allocator, record_path);

    // load the corresponding .so file
    const so_path = try std.fmt.allocPrint(allocator, "{s}/candidate_{d}.so", .{ opts.work_dir, best_idx });
    defer allocator.free(so_path);

    log.info("Loading tuned module: {s}", .{so_path});

    const module_handle = try module_load_from_file(allocator, so_path);
    errdefer _ = c.TVMFFIObjectDecRef(module_handle);

    // get main function
    const main_func = try module_get_function(allocator, module_handle, "main", true);

    log.info("Loaded tuned module successfully", .{});

    return TunedModule{
        .module_handle = module_handle,
        .main_func = main_func,
        .best_candidate = best_idx,
        .best_time_us = best_time * 1e6,
        .allocator = allocator,
    };
}

/// Run a tuned matmul with random test data and verify correctness.
pub fn run_tuned_matmul(
    allocator: std.mem.Allocator,
    M: usize,
    N: usize,
    K: usize,
    opts: LoadTunedOpts,
) !void {
    const log = std.log.scoped(.@"zg/tvm_run");

    log.info("Running tuned matmul: [{d},{d}] x [{d},{d}] -> [{d},{d}]", .{ M, K, K, N, M, N });

    var tuned = try load_tuned_module(allocator, opts);
    defer tuned.deinit();

    log.info("Using candidate {d} (tuned time: {d:.2} µs)", .{ tuned.best_candidate, tuned.best_time_us });

    // set up random test data
    const a_data = try allocator.alloc(f32, M * K);
    defer allocator.free(a_data);
    const b_data = try allocator.alloc(f32, K * N);
    defer allocator.free(b_data);
    const c_data = try allocator.alloc(f32, M * N);
    defer allocator.free(c_data);

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();
    for (a_data) |*v| v.* = rand.float(f32);
    for (b_data) |*v| v.* = rand.float(f32);
    @memset(c_data, 0);

    // create DLPack tensors
    var shape_a = [_]i64{ @intCast(M), @intCast(K) };
    var shape_b = [_]i64{ @intCast(K), @intCast(N) };
    var shape_c = [_]i64{ @intCast(M), @intCast(N) };

    var dl_a = c.DLManagedTensor{ // NOTE: should we consider DLPack bindings/Zig wrapper types going forward? Should discuss the possibility of DLPack as a Zigrad dependency, irrespective of TVM.
        .dl_tensor = make_dl_tensor_f32(a_data, &shape_a),
        .manager_ctx = null,
        .deleter = dlpack_noop_deleter,
    };
    var dl_b = c.DLManagedTensor{
        .dl_tensor = make_dl_tensor_f32(b_data, &shape_b),
        .manager_ctx = null,
        .deleter = dlpack_noop_deleter,
    };
    var dl_c = c.DLManagedTensor{
        .dl_tensor = make_dl_tensor_f32(c_data, &shape_c),
        .manager_ctx = null,
        .deleter = dlpack_noop_deleter,
    };

    // convert to TVM tensors
    const t_a = try tensor_from_dlpack(allocator, &dl_a);
    defer _ = c.TVMFFIObjectDecRef(t_a);
    const t_b = try tensor_from_dlpack(allocator, &dl_b);
    defer _ = c.TVMFFIObjectDecRef(t_b);
    const t_c = try tensor_from_dlpack(allocator, &dl_c);
    defer _ = c.TVMFFIObjectDecRef(t_c);

    var call_args = [_]c.TVMFFIAny{
        any_obj(t_a, c.kTVMFFITensor),
        any_obj(t_b, c.kTVMFFITensor),
        any_obj(t_c, c.kTVMFFITensor),
    };
    var call_res: c.TVMFFIAny = undefined; // NOTE: audit use of undefined

    log.info("Executing tuned kernel...", .{});

    const warmup_iters = 10;
    const bench_iters = 100;

    for (0..warmup_iters) |_| {
        try ffi_call(allocator, tuned.main_func, &call_args, &call_res);
    }

    // benchmark
    const start = std.time.nanoTimestamp();
    for (0..bench_iters) |_| {
        try ffi_call(allocator, tuned.main_func, &call_args, &call_res);
    }
    const end = std.time.nanoTimestamp();
    const elapsed_ns: u64 = @intCast(end - start);
    const avg_us = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(bench_iters)) / 1000.0;

    log.info("Benchmark: {d:.2} µs/iter (tuned prediction: {d:.2} µs)", .{ avg_us, tuned.best_time_us });

    // Verify correctness (simple check: compute reference matmul) // NOTE: rephrase to remove parenthetical
    log.info("Verifying correctness...", .{});
    var max_diff: f32 = 0;
    for (0..M) |i| {
        for (0..N) |j| {
            var expected: f32 = 0;
            for (0..K) |k| {
                expected += a_data[i * K + k] * b_data[k * N + j];
            }
            const actual = c_data[i * N + j];
            const diff = @abs(actual - expected);
            if (diff > max_diff) max_diff = diff;
        }
    }

    if (max_diff < 1e-4) {
        log.info("✓ Verification passed (max diff: {e:.2})", .{max_diff});
    } else {
        log.err("✗ Verification failed (max diff: {e:.2})", .{max_diff});
        return error.VerificationFailed;
    }

    const flops = 2.0 * @as(f64, @floatFromInt(M * N * K)); // 2 ops per multiply-add
    const gflops = flops / (avg_us * 1000.0);
    log.info("Performance: {d:.2} GFLOP/s", .{gflops});
}
