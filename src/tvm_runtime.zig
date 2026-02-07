const std = @import("std");
const build_options = @import("build_options");

const c = if (build_options.enable_tvm)
    @cImport({
        @cInclude("tvm/ffi/c_api.h");
    })
else
    struct {};

/// Handle to the dynamically loaded libtvm.so (full compiler with TE/codegen).
/// This is loaded on-demand since the Zig linker drops it (no direct symbol refs).
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

    var read_buf: [8192]u8 = undefined;
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
const RTLD_GLOBAL: c_int = 0x100;
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

fn ffi_fail(comptime what: []const u8) !noreturn {
    _ = what;
    return error.TvmRuntimeError;
}

fn any_none() c.TVMFFIAny {
    var v = std.mem.zeroes(c.TVMFFIAny);
    v.type_index = c.kTVMFFINone;
    return v;
}

fn any_int(value: i64) c.TVMFFIAny {
    var v = std.mem.zeroes(c.TVMFFIAny);
    v.type_index = c.kTVMFFIInt;
    v.unnamed_1.v_int64 = value;
    return v;
}

fn any_bool(value: bool) c.TVMFFIAny {
    var v = std.mem.zeroes(c.TVMFFIAny);
    v.type_index = c.kTVMFFIBool;
    v.unnamed_1.v_int64 = if (value) 1 else 0;
    return v;
}

fn any_raw_str(cstr: [*:0]const u8) c.TVMFFIAny {
    var v = std.mem.zeroes(c.TVMFFIAny);
    v.type_index = c.kTVMFFIRawStr;
    v.unnamed_1.v_c_str = cstr;
    return v;
}

fn any_obj(handle: c.TVMFFIObjectHandle, type_index: i32) c.TVMFFIAny {
    var v = std.mem.zeroes(c.TVMFFIAny);
    v.type_index = type_index;
    v.unnamed_1.v_obj = @ptrCast(@alignCast(handle));
    return v;
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

fn any_to_string(allocator: std.mem.Allocator, v: *c.TVMFFIAny) ![]u8 {
    // Small string is stored inline.
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

fn ffi_call_global(allocator: std.mem.Allocator, name: []const u8, args: []c.TVMFFIAny, out: *c.TVMFFIAny) !void {
    const func = try ffi_get_global(allocator, name);
    defer _ = c.TVMFFIObjectDecRef(func);
    try ffi_call(allocator, func, args, out);
}

fn cstr_alloc(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    const buf = try allocator.alloc(u8, s.len + 1);
    std.mem.copyForwards(u8, buf[0..s.len], s);
    buf[s.len] = 0;
    return buf;
}

fn cstr_ptr(buf: []u8) [*:0]const u8 {
    return @ptrCast(buf.ptr);
}

fn module_load_from_file(allocator: std.mem.Allocator, path: []const u8, format: []const u8) !c.TVMFFIObjectHandle {
    const path_buf = try cstr_alloc(allocator, path);
    defer allocator.free(path_buf);
    const fmt_buf = try cstr_alloc(allocator, format);
    defer allocator.free(fmt_buf);
    const path_z = cstr_ptr(path_buf);
    const fmt_z = cstr_ptr(fmt_buf);

    var out: c.TVMFFIAny = undefined;
    var args = [_]c.TVMFFIAny{ any_raw_str(path_z), any_raw_str(fmt_z) };
    try ffi_call_global(allocator, "ffi.ModuleLoadFromFile", &args, &out);
    if (out.type_index != c.kTVMFFIModule or out.unnamed_1.v_obj == null) {
        std.log.err("unexpected ModuleLoadFromFile return type: {d}", .{out.type_index});
        return error.UnexpectedTvmType;
    }
    return @ptrCast(out.unnamed_1.v_obj);
}

fn module_get_function(
    allocator: std.mem.Allocator,
    module: c.TVMFFIObjectHandle,
    name: []const u8,
    query_imports: bool,
) !c.TVMFFIObjectHandle {
    const name_buf = try cstr_alloc(allocator, name);
    defer allocator.free(name_buf);
    const name_z = cstr_ptr(name_buf);

    var out: c.TVMFFIAny = undefined;
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

fn make_dl_tensor(data: []f16, shape: []i64) c.DLTensor {
    return .{
        .data = @ptrCast(data.ptr),
        .device = .{ .device_type = c.kDLCPU, .device_id = 0 },
        .ndim = @intCast(shape.len),
        .dtype = .{ .code = c.kDLFloat, .bits = 16, .lanes = 1 },
        .shape = shape.ptr,
        .strides = null,
        .byte_offset = 0,
    };
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

    var res0: c.TVMFFIAny = undefined;
    try ffi_call0(allocator, factory, &res0);
    if (res0.type_index != c.kTVMFFIFunction or res0.unnamed_1.v_obj == null) {
        std.log.err("unexpected return type from {s}(): type_index={d}", .{ name, res0.type_index });
        return error.UnexpectedTvmType;
    }
    const functor: c.TVMFFIObjectHandle = @ptrCast(res0.unnamed_1.v_obj);
    defer _ = c.TVMFFIObjectDecRef(functor);

    var res_len: c.TVMFFIAny = undefined;
    try ffi_call1_i64(allocator, functor, -1, &res_len);
    if (res_len.type_index != c.kTVMFFIInt) {
        std.log.err("unexpected len return type: type_index={d}", .{res_len.type_index});
        return error.UnexpectedTvmType;
    }
    const count: usize = @intCast(res_len.unnamed_1.v_int64);

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const out = &stdout_writer.interface;
    defer out.flush() catch {};

    try out.print("TVM FFI global functions: {d}\n", .{count});
    for (0..count) |i| {
        var res_name: c.TVMFFIAny = undefined;
        try ffi_call1_i64(allocator, functor, @intCast(i), &res_name);
        const s = try any_to_string(allocator, &res_name);
        defer allocator.free(s);
        try out.print("  {s}\n", .{s});
    }
}

pub const TargetKind = enum { cpu, cuda };

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

    // --- Step 1: Create shape as Array<ir.PrimExpr> ---
    const n_i64: i64 = @intCast(n);

    // te.Placeholder expects shape as Array<ir.PrimExpr>
    // Integers can be used directly as PrimExpr in TVM
    var shape_array: c.TVMFFIAny = undefined;
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

    // --- Step 2: Create placeholder tensors A and B ---
    // te.Placeholder(shape: Array<PrimExpr>, dtype: DataType, name: string)
    const name_a_buf = try cstr_alloc(allocator, "A");
    defer allocator.free(name_a_buf);
    const name_b_buf = try cstr_alloc(allocator, "B");
    defer allocator.free(name_b_buf);
    // Use float32 for now (standard C support, CUDA with fp32 also works)
    // TODO: switch to float16 for CUDA when we get the include paths sorted out
    const dtype_buf = try cstr_alloc(allocator, "float32");
    defer allocator.free(dtype_buf);

    var tensor_a: c.TVMFFIAny = undefined;
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

    var tensor_b: c.TVMFFIAny = undefined;
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

    // --- Step 3: Create iteration variable for the compute ---
    const name_i_buf = try cstr_alloc(allocator, "i");
    defer allocator.free(name_i_buf);
    const int32_buf = try cstr_alloc(allocator, "int32");
    defer allocator.free(int32_buf);

    // Create a tir.Var for the loop index
    // tir.Var(name: str, dtype: AnyView, span: ir.Span)
    var iter_var_i: c.TVMFFIAny = undefined;
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

    // --- Step 4: Create load expressions A[i] and B[i] ---
    // tir.ProducerLoad(producer, indices)
    var indices_array: c.TVMFFIAny = undefined;
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

    var load_a: c.TVMFFIAny = undefined;
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

    var load_b: c.TVMFFIAny = undefined;
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

    // --- Step 5: Create Add expression: A[i] + B[i] ---
    var add_expr: c.TVMFFIAny = undefined;
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

    // --- Step 6: Create IterVar for the compute axis ---
    // tir.IterVar(dom, var, iter_type, thread_tag)
    var iter_dom: c.TVMFFIAny = undefined;
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

    var axis_iter_var: c.TVMFFIAny = undefined;
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

    // --- Step 7: Create ComputeOp ---
    // te.ComputeOp(name, tag, attrs, axis, body)
    const name_c_buf = try cstr_alloc(allocator, "C");
    defer allocator.free(name_c_buf);
    const tag_buf = try cstr_alloc(allocator, "");
    defer allocator.free(tag_buf);

    var axis_array: c.TVMFFIAny = undefined;
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

    var body_array: c.TVMFFIAny = undefined;
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

    var compute_op: c.TVMFFIAny = undefined;
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

    // --- Step 8: Get output tensor C from ComputeOp ---
    var tensor_c: c.TVMFFIAny = undefined;
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

    // --- Step 9: Create PrimFunc from tensors ---
    // te.CreatePrimFunc(tensors: Array[Tensor]) -> tir.PrimFunc
    var tensors_array: c.TVMFFIAny = undefined;
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

    var prim_func: c.TVMFFIAny = undefined;
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

        var prim_func_with_attr: c.TVMFFIAny = undefined;
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

    // --- Step 10: Create target with host ---
    // For CPU: use 'c' target to avoid LLVM JIT conflicts with SDK's LLVM 22.
    // For CUDA: device is 'cuda', host is 'c' (generates C source compiled with gcc).
    // The 'c' target avoids LLVM entirely, using external C compiler instead.
    const host_str = try cstr_alloc(allocator, "c");
    defer allocator.free(host_str);

    // Device target depends on target_kind
    const device_str = switch (target_kind) {
        .cpu => try cstr_alloc(allocator, "c"),
        .cuda => try cstr_alloc(allocator, "cuda"),
    };
    defer allocator.free(device_str);

    var target: c.TVMFFIAny = undefined;
    {
        // Create device target
        var device_target: c.TVMFFIAny = undefined;
        {
            var args = [_]c.TVMFFIAny{any_raw_str(cstr_ptr(device_str))};
            ffi_call_global(allocator, "target.Target", &args, &device_target) catch |err| {
                log.err("target.Target({s}) failed: {s}", .{ device_str, @errorName(err) });
                return err;
            };
        }
        // Create host target
        var host_target: c.TVMFFIAny = undefined;
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

    // --- Step 11: Build module ---
    // target.Build(IRModule, Target) -> runtime.Module
    // First wrap prim_func in an IRModule
    var ir_mod: c.TVMFFIAny = undefined;
    {
        // Create GlobalVar for the function name
        const main_buf = try cstr_alloc(allocator, "main");
        defer allocator.free(main_buf);

        var global_var: c.TVMFFIAny = undefined;
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
        var func_map: c.TVMFFIAny = undefined;
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
        var empty_map: c.TVMFFIAny = undefined;
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

    // --- Step 11b: Apply TIR lowering passes ---
    // The codegen requires buffer_map to be empty. We must run lowering passes first.
    // Minimal pipeline: BindTarget → FlattenBuffer → MakePackedAPI → finalization
    var lowered_mod = ir_mod;

    // Create and apply sequential passes
    {
        // 1. BindTarget - binds target to all functions
        var bind_pass: c.TVMFFIAny = undefined;
        {
            var args = [_]c.TVMFFIAny{target};
            try ffi_call_global(allocator, "tir.transform.BindTarget", &args, &bind_pass);
        }
        defer if (bind_pass.unnamed_1.v_obj) |obj| {
            _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
        };

        var pass_result: c.TVMFFIAny = undefined;
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

    // Helper to apply a nullary transform pass
    const apply_pass = struct {
        fn f(alloc: std.mem.Allocator, pass_name: []const u8, mod: *c.TVMFFIAny, logger: anytype) !void {
            var pass: c.TVMFFIAny = undefined;
            ffi_call_global(alloc, pass_name, &.{}, &pass) catch |err| {
                logger.err("{s}() failed: {s}", .{ pass_name, @errorName(err) });
                return err;
            };
            defer if (pass.unnamed_1.v_obj) |obj| {
                _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
            };

            var result: c.TVMFFIAny = undefined;
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

    // --- Step 12: Build runtime module(s) ---
    var built_mod: c.TVMFFIAny = undefined;

    if (target_kind == .cpu) {
        // CPU: use 'c' target to avoid LLVM JIT conflicts
        var args = [_]c.TVMFFIAny{ lowered_mod, target };
        ffi_call_global(allocator, "target.build.c", &args, &built_mod) catch |err| {
            log.err("target.build.c failed: {s}", .{@errorName(err)});
            return err;
        };
        log.info("Built CPU module (C target): type_index={d}", .{built_mod.type_index});
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

        const filter_cb = struct {
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
        var host_only_target: c.TVMFFIAny = undefined;
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

        var device_only_target: c.TVMFFIAny = undefined;
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

        // Use tir.transform.Filter to create host and device modules
        // We need to register a callback function with TVM
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

        // Create Filter passes
        var host_filter_pass: c.TVMFFIAny = undefined;
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

        var device_filter_pass: c.TVMFFIAny = undefined;
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

        // Apply host filter to get host-only module
        var host_ir_mod: c.TVMFFIAny = undefined;
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

        // Apply device filter to get device-only module
        var device_ir_mod: c.TVMFFIAny = undefined;
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

        // Build host module with C backend (avoids LLVM JIT conflicts)
        var host_mod: c.TVMFFIAny = undefined;
        {
            var args = [_]c.TVMFFIAny{ host_ir_mod, host_only_target };
            ffi_call_global(allocator, "target.build.c", &args, &host_mod) catch |err| {
                log.err("target.build.c(host) failed: {s}", .{@errorName(err)});
                return err;
            };
        }
        log.info("Built host module (C target): type_index={d}", .{host_mod.type_index});

        // Build device module with CUDA
        var device_mod: c.TVMFFIAny = undefined;
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

        // Import device module into host module
        {
            var args = [_]c.TVMFFIAny{ host_mod, device_mod };
            var result: c.TVMFFIAny = undefined;
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

    // --- Step 13: Debug module info and export if needed ---
    // The 'c' target creates a CSourceModule with C source code that needs compilation.
    // Check module kind and supported formats.
    {
        var kind: c.TVMFFIAny = undefined;
        var args = [_]c.TVMFFIAny{built_mod};
        ffi_call_global(allocator, "ffi.ModuleGetKind", &args, &kind) catch |err| {
            log.warn("ffi.ModuleGetKind failed: {s}", .{@errorName(err)});
        };
        if (kind.type_index != c.kTVMFFINone) {
            if (any_to_string(allocator, &kind)) |kind_str| {
                defer allocator.free(kind_str);
                log.info("Module kind: {s}", .{kind_str});
            } else |_| {
                log.info("Module kind: (could not convert to string)", .{});
            }
        }

        var formats: c.TVMFFIAny = undefined;
        ffi_call_global(allocator, "ffi.ModuleGetWriteFormats", &args, &formats) catch |err| {
            log.warn("ffi.ModuleGetWriteFormats failed: {s}", .{@errorName(err)});
        };
        if (formats.type_index != c.kTVMFFINone) {
            if (any_to_string(allocator, &formats)) |fmt_str| {
                defer allocator.free(fmt_str);
                log.info("Write formats: {s}", .{fmt_str});
            } else |_| {}
        }

        // Check the source code in the module (format is required, use "c" for C source)
        var source: c.TVMFFIAny = undefined;
        const fmt_c_buf = try cstr_alloc(allocator, "c");
        defer allocator.free(fmt_c_buf);
        var src_args = [_]c.TVMFFIAny{ built_mod, any_raw_str(cstr_ptr(fmt_c_buf)) };
        ffi_call_global(allocator, "ffi.ModuleInspectSource", &src_args, &source) catch |err| {
            log.warn("ffi.ModuleInspectSource failed: {s}", .{@errorName(err)});
        };
        if (source.type_index != c.kTVMFFINone) {
            if (any_to_string(allocator, &source)) |src_str| {
                defer allocator.free(src_str);
                // Print first 500 chars of source
                const len = @min(src_str.len, 500);
                log.info("Module source preview:\n{s}...", .{src_str[0..len]});
            } else |_| {}
        }
    }

    // For 'c' target, we need to export to .c and compile with an external compiler.
    // Both CPU and CUDA use 'c' target for host code, so always export and compile.
    {
        const c_path = "/tmp/tvm_module.c";
        const so_path = "/tmp/tvm_module.so";
        const c_path_buf = try cstr_alloc(allocator, c_path);
        defer allocator.free(c_path_buf);
        const format_c = try cstr_alloc(allocator, "c");
        defer allocator.free(format_c);

        // Export C source
        {
            var result: c.TVMFFIAny = undefined;
            var args = [_]c.TVMFFIAny{
                built_mod,
                any_raw_str(cstr_ptr(c_path_buf)),
                any_raw_str(cstr_ptr(format_c)),
            };
            ffi_call_global(allocator, "ffi.ModuleWriteToFile", &args, &result) catch |err| {
                log.err("ffi.ModuleWriteToFile(.c) failed: {s}", .{@errorName(err)});
                return err;
            };
        }
        log.info("Exported C source to {s}", .{c_path});

        // Compile with gcc
        // We need TVM runtime includes from the TVM library path
        const lib_path = try findTvmLibPath(allocator);
        defer if (lib_path) |p| allocator.free(p);

        var tvm_include_dir: []const u8 = "";
        var tvm_lib_dir: []const u8 = "";
        if (lib_path) |p| {
            // Extract directories: /nix/.../lib/libtvm.so -> include and lib dirs
            if (std.mem.lastIndexOf(u8, p, "/lib/")) |idx| {
                tvm_include_dir = try std.fmt.allocPrint(allocator, "{s}/include", .{p[0..idx]});
                tvm_lib_dir = try std.fmt.allocPrint(allocator, "{s}/lib", .{p[0..idx]});
            }
        }
        defer if (tvm_include_dir.len > 0) allocator.free(tvm_include_dir);
        defer if (tvm_lib_dir.len > 0) allocator.free(tvm_lib_dir);

        // Compile C source to .so using zig cc (Zig's built-in C compiler)
        // Link against TVM runtime for symbols like TVMBackendGetFuncFromEnv
        var argv_buf: [24][]const u8 = undefined;
        var argc: usize = 0;
        argv_buf[argc] = "zig";
        argc += 1;
        argv_buf[argc] = "cc";
        argc += 1;
        argv_buf[argc] = "-shared";
        argc += 1;
        argv_buf[argc] = "-fPIC";
        argc += 1;
        argv_buf[argc] = "-O2";
        argc += 1;
        argv_buf[argc] = "-o";
        argc += 1;
        argv_buf[argc] = so_path;
        argc += 1;
        argv_buf[argc] = c_path;
        argc += 1;
        if (tvm_include_dir.len > 0) {
            argv_buf[argc] = "-I";
            argc += 1;
            argv_buf[argc] = tvm_include_dir;
            argc += 1;
        }
        // Note: Don't link against libtvm_runtime here - symbols are already available
        // from the TVM libraries loaded in the main process. Linking again would cause
        // duplicate global function registration errors.
        // The .so will resolve TVM symbols at load time from the already-loaded libraries.

        // Debug: print compile command (join argv)
        if (false) { // disabled - too verbose
            var cmd_buf: [1024]u8 = undefined;
            var cmd_len: usize = 0;
            for (argv_buf[0..argc]) |arg| {
                if (cmd_len + arg.len + 1 < cmd_buf.len) {
                    @memcpy(cmd_buf[cmd_len..][0..arg.len], arg);
                    cmd_len += arg.len;
                    cmd_buf[cmd_len] = ' ';
                    cmd_len += 1;
                }
            }
            log.debug("Compiling: {s}", .{cmd_buf[0..cmd_len]});
        }
        const compile_result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = argv_buf[0..argc],
        }) catch |err| {
            log.err("cc compilation failed: {s}", .{@errorName(err)});
            return err;
        };
        defer allocator.free(compile_result.stdout);
        defer allocator.free(compile_result.stderr);

        switch (compile_result.term) {
            .Exited => |code| {
                if (code != 0) {
                    log.err("cc failed with exit code {d}: {s}", .{ code, compile_result.stderr });
                    return error.TvmRuntimeError;
                }
            },
            else => {
                log.err("cc terminated abnormally", .{});
                return error.TvmRuntimeError;
            },
        }
        log.info("Compiled {s} to {s}", .{ c_path, so_path });

        // Load the compiled .so
        const so_path_buf = try cstr_alloc(allocator, so_path);
        defer allocator.free(so_path_buf);

        var loaded_mod: c.TVMFFIAny = undefined;
        {
            var args = [_]c.TVMFFIAny{
                any_raw_str(cstr_ptr(so_path_buf)),
            };
            ffi_call_global(allocator, "ffi.ModuleLoadFromFile", &args, &loaded_mod) catch |err| {
                log.err("ffi.ModuleLoadFromFile(.so) failed: {s}", .{@errorName(err)});
                return err;
            };
        }
        log.info("Loaded compiled module: type_index={d}", .{loaded_mod.type_index});

        // Replace built_mod with loaded_mod
        if (built_mod.unnamed_1.v_obj) |obj| {
            _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
        }
        built_mod = loaded_mod;
    }

    // --- Step 14: Get the compiled function ---
    var compiled_func: c.TVMFFIAny = undefined;
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

    // --- Step 15: Execute with test data ---
    // Use f32 for CPU (standard C support)
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

    // For CUDA, allocate device tensors and copy data
    var t_a: c.TVMFFIObjectHandle = undefined;
    var t_b: c.TVMFFIObjectHandle = undefined;
    var t_c: c.TVMFFIObjectHandle = undefined;

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
        const float32_buf = try cstr_alloc(allocator, "float32");
        defer allocator.free(float32_buf);

        // Create shape for TVM
        var shape_any: c.TVMFFIAny = undefined;
        {
            var args = [_]c.TVMFFIAny{any_int(n_i64)};
            try ffi_call_global(allocator, "ffi.Shape", &args, &shape_any);
        }
        defer if (shape_any.unnamed_1.v_obj) |obj| {
            _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
        };

        // Allocate device tensors
        // runtime.TVMTensorAllocWithScope(shape, dtype, device, mem_scope)
        const alloc_tensor = struct {
            fn f(alloc: std.mem.Allocator, shape: c.TVMFFIAny, dtype_cstr: [*:0]const u8, dev_type: i32, dev_id: i32) !c.TVMFFIObjectHandle {
                var result: c.TVMFFIAny = undefined;
                const dev_arr = [_]i64{ dev_type, dev_id };
                var dev: c.TVMFFIAny = undefined;
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

        // Copy input data to device
        // runtime.TVMTensorCopyFromBytes(tensor, data_ptr, nbytes)
        const copy_to_device = struct {
            fn f(alloc: std.mem.Allocator, tensor: c.TVMFFIObjectHandle, data: [*]const f32, bytes: usize) !void {
                var result: c.TVMFFIAny = undefined;
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

    // Call the compiled function
    var exec_args = [_]c.TVMFFIAny{
        any_obj(t_a, c.kTVMFFITensor),
        any_obj(t_b, c.kTVMFFITensor),
        any_obj(t_c, c.kTVMFFITensor),
    };
    const func_handle: c.TVMFFIObjectHandle = @ptrCast(compiled_func.unnamed_1.v_obj);
    var exec_result: c.TVMFFIAny = undefined;
    try ffi_call(allocator, func_handle, &exec_args, &exec_result);

    log.info("Executed vec_add successfully!", .{});

    // Copy result back from device if CUDA
    if (target_kind == .cuda) {
        var result: c.TVMFFIAny = undefined;
        var args = [_]c.TVMFFIAny{
            any_obj(t_c, c.kTVMFFITensor),
            .{ .type_index = c.kTVMFFIOpaquePtr, .unnamed_0 = .{ .small_str_len = 0 }, .unnamed_1 = .{ .v_ptr = @ptrCast(c_data.ptr) } },
            any_int(@intCast(nbytes)),
        };
        try ffi_call_global(allocator, "runtime.TVMTensorCopyToBytes", &args, &result);
        log.info("Copied result from GPU", .{});
    }

    // Print results
    var stdout_buffer: [8192]u8 = undefined;
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

    const module = try module_load_from_file(allocator, module_path, "so");
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
    var res: c.TVMFFIAny = undefined;
    try ffi_call(allocator, func, &args, &res);

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const out = &stdout_writer.interface;
    defer out.flush() catch {};

    try out.writeAll("TVM vec_add result (first 8):\n");
    const limit = @min(n, 8);
    for (c_out[0..limit], 0..) |v, i| {
        try out.print("  [{d}] = {d}\n", .{ i, v });
    }
}
