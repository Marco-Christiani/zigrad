//! Mirage C ABI declarations with runtime symbol loading.
const std = @import("std");

const log = std.log.scoped(.@"zg/mirage_cffi");

pub const MirageContext = opaque {};
pub const MirageGraph = opaque {};
pub const MirageTensor = u32;

pub const MirageStatus = enum(c_int) {
    ok = 0,
    invalid_argument = 1,
    internal_error = 2,
    unsupported = 3,
};

pub const MirageDType = enum(c_int) {
    f16 = 0,
    bf16 = 1,
    f32 = 2,
    f64 = 3,
    i8 = 4,
    i32 = 5,
    i64 = 6,
    u32 = 7,
    u64 = 8,
};

pub const MirageUnaryOp = enum(c_int) {
    exp = 0,
    sqrt = 1,
    silu = 2,
    gelu = 3,
    relu = 4,
    log = 5,
};

pub const MirageBinaryOp = enum(c_int) {
    add = 0,
    mul = 1,
    div = 2,
    pow = 3,
};

pub const BufferDesc = extern struct {
    data: ?*anyopaque,
    dtype: MirageDType,
    dims: [*]const i64,
    rank: usize,
};

pub const DispatchParams = extern struct {
    inputs: ?[*]const BufferDesc,
    num_inputs: usize,
    outputs: ?[*]const BufferDesc,
    num_outputs: usize,
    device_ordinal: i32,
    stream: ?*anyopaque,
};

pub const LaunchInfo = extern struct {
    workspace_bytes: usize,
};

pub const SearchDim3 = extern struct {
    x: i32,
    y: i32,
    z: i32,
};

pub const SuperoptOptions = extern struct {
    max_num_graphs: u32,

    imap_to_explore: ?[*]const SearchDim3,
    num_imaps: usize,
    omap_to_explore: ?[*]const SearchDim3,
    num_omaps: usize,

    grid_dim_to_explore: ?[*]const SearchDim3,
    num_grid_dims: usize,
    block_dim_to_explore: ?[*]const SearchDim3,
    num_block_dims: usize,

    fmap_to_explore: ?[*]const i32,
    num_fmaps: usize,
    frange_to_explore: ?[*]const i32,
    num_franges: usize,

    checkpoint_filename: ?[*:0]const u8,
    verbose: u8,
    is_formal_verified: u8,
};

const FnStatusString = *const fn (MirageStatus) callconv(.c) [*:0]const u8;
const FnContextCreate = *const fn (out_ctx: *?*MirageContext) callconv(.c) MirageStatus;
const FnContextDestroy = *const fn (ctx: ?*MirageContext) callconv(.c) void;
const FnGraphCreate = *const fn (ctx: ?*MirageContext, out_graph: *?*MirageGraph) callconv(.c) MirageStatus;
const FnGraphDestroy = *const fn (graph: ?*MirageGraph) callconv(.c) void;
const FnGraphNewInput = *const fn (
    graph: ?*MirageGraph,
    dims: [*]const i64,
    rank: usize,
    dtype: MirageDType,
    out_tensor: *MirageTensor,
) callconv(.c) MirageStatus;
const FnGraphMatmul = *const fn (
    graph: ?*MirageGraph,
    lhs: MirageTensor,
    rhs: MirageTensor,
    out_tensor: *MirageTensor,
) callconv(.c) MirageStatus;
const FnGraphUnary = *const fn (
    graph: ?*MirageGraph,
    op: MirageUnaryOp,
    input: MirageTensor,
    out_tensor: *MirageTensor,
) callconv(.c) MirageStatus;
const FnGraphBinary = *const fn (
    graph: ?*MirageGraph,
    op: MirageBinaryOp,
    lhs: MirageTensor,
    rhs: MirageTensor,
    out_tensor: *MirageTensor,
) callconv(.c) MirageStatus;
const FnGraphMarkOutput = *const fn (
    graph: ?*MirageGraph,
    tensor: MirageTensor,
) callconv(.c) MirageStatus;
const FnGraphSuperoptimize = *const fn (
    graph: ?*MirageGraph,
    config_name: ?[*:0]const u8,
) callconv(.c) MirageStatus;
const FnGraphSuperoptimizeWithOptions = *const fn (
    graph: ?*MirageGraph,
    config_name: ?[*:0]const u8,
    options: ?*const SuperoptOptions,
) callconv(.c) MirageStatus;
const FnGraphCompile = *const fn (
    ctx: ?*MirageContext,
    graph: ?*MirageGraph,
    target_cc: c_int,
    out_artifact_ptr: *[*]const u8,
    out_artifact_len: *usize,
    out_launch_info: *LaunchInfo,
) callconv(.c) MirageStatus;
const FnExecuteKernel = *const fn (
    ctx: ?*MirageContext,
    artifact_ptr: [*]const u8,
    artifact_len: usize,
    params: *const DispatchParams,
    workspace: ?*anyopaque,
) callconv(.c) MirageStatus;
const FnValidateArtifact = *const fn (
    ctx: ?*MirageContext,
    artifact_ptr: [*]const u8,
    artifact_len: usize,
) callconv(.c) MirageStatus;
const FnReleaseBuffer = *const fn (
    ctx: ?*MirageContext,
    buffer_ptr: [*]const u8,
    buffer_len: usize,
) callconv(.c) void;
const FnReleaseDeviceMemory = *const fn () callconv(.c) void;
const FnDeviceMemInfo = *const fn (
    out_free: *usize,
    out_total: *usize,
) callconv(.c) MirageStatus;

var fn_status_string: ?FnStatusString = null;
var fn_context_create: ?FnContextCreate = null;
var fn_context_destroy: ?FnContextDestroy = null;
var fn_graph_create: ?FnGraphCreate = null;
var fn_graph_destroy: ?FnGraphDestroy = null;
var fn_graph_new_input: ?FnGraphNewInput = null;
var fn_graph_matmul: ?FnGraphMatmul = null;
var fn_graph_unary: ?FnGraphUnary = null;
var fn_graph_binary: ?FnGraphBinary = null;
var fn_graph_mark_output: ?FnGraphMarkOutput = null;
var fn_graph_superoptimize: ?FnGraphSuperoptimize = null;
var fn_graph_superoptimize_with_options: ?FnGraphSuperoptimizeWithOptions = null;
var fn_graph_compile: ?FnGraphCompile = null;
var fn_execute_kernel: ?FnExecuteKernel = null;
var fn_validate_artifact: ?FnValidateArtifact = null;
var fn_release_buffer: ?FnReleaseBuffer = null;
var fn_release_device_memory: ?FnReleaseDeviceMemory = null;
var fn_device_mem_info: ?FnDeviceMemInfo = null;

var load_mutex: std.Thread.Mutex = .{};
var symbols_ready = false;

pub const LoadError = error{
    MirageSymbolMissing,
};

extern "c" fn dlsym(handle: *anyopaque, symbol: [*:0]const u8) ?*anyopaque;
extern "c" fn dlerror() ?[*:0]const u8;

pub fn ensure_loaded(handle: *anyopaque) LoadError!void {
    load_mutex.lock();
    defer load_mutex.unlock();

    if (symbols_ready) return;

    fn_status_string = try load_symbol(FnStatusString, handle, "mirage_status_string");
    fn_context_create = try load_symbol(FnContextCreate, handle, "mirage_context_create");
    fn_context_destroy = try load_symbol(FnContextDestroy, handle, "mirage_context_destroy");
    fn_graph_create = try load_symbol(FnGraphCreate, handle, "mirage_graph_create");
    fn_graph_destroy = try load_symbol(FnGraphDestroy, handle, "mirage_graph_destroy");
    fn_graph_new_input = try load_symbol(FnGraphNewInput, handle, "mirage_graph_new_input");
    fn_graph_matmul = try load_symbol(FnGraphMatmul, handle, "mirage_graph_matmul");
    fn_graph_unary = try load_symbol(FnGraphUnary, handle, "mirage_graph_unary");
    fn_graph_binary = try load_symbol(FnGraphBinary, handle, "mirage_graph_binary");
    fn_graph_mark_output = try load_symbol(FnGraphMarkOutput, handle, "mirage_graph_mark_output");
    fn_graph_superoptimize = try load_symbol(FnGraphSuperoptimize, handle, "mirage_graph_superoptimize");
    fn_graph_superoptimize_with_options = load_symbol_optional(FnGraphSuperoptimizeWithOptions, handle, "mirage_graph_superoptimize_with_options");
    fn_graph_compile = try load_symbol(FnGraphCompile, handle, "mirage_graph_compile");
    fn_execute_kernel = try load_symbol(FnExecuteKernel, handle, "mirage_execute_kernel");
    fn_validate_artifact = try load_symbol(FnValidateArtifact, handle, "mirage_validate_artifact");
    fn_release_buffer = try load_symbol(FnReleaseBuffer, handle, "mirage_release_buffer");
    fn_release_device_memory = load_symbol_optional(FnReleaseDeviceMemory, handle, "mirage_release_device_memory");
    fn_device_mem_info = load_symbol_optional(FnDeviceMemInfo, handle, "mirage_device_mem_info");

    symbols_ready = true;
}

pub fn mirage_status_string(status: MirageStatus) [*:0]const u8 {
    const f = fn_status_string orelse return "unknown_status";
    return f(status);
}

pub fn mirage_context_create(out_ctx: *?*MirageContext) MirageStatus {
    const f = fn_context_create orelse return .internal_error;
    return f(out_ctx);
}

pub fn mirage_context_destroy(ctx: ?*MirageContext) void {
    const f = fn_context_destroy orelse return;
    f(ctx);
}

pub fn mirage_graph_create(ctx: ?*MirageContext, out_graph: *?*MirageGraph) MirageStatus {
    const f = fn_graph_create orelse return .internal_error;
    return f(ctx, out_graph);
}

pub fn mirage_graph_destroy(graph: ?*MirageGraph) void {
    const f = fn_graph_destroy orelse return;
    f(graph);
}

pub fn mirage_graph_new_input(
    graph: ?*MirageGraph,
    dims: [*]const i64,
    rank: usize,
    dtype: MirageDType,
    out_tensor: *MirageTensor,
) MirageStatus {
    const f = fn_graph_new_input orelse return .internal_error;
    return f(graph, dims, rank, dtype, out_tensor);
}

pub fn mirage_graph_matmul(
    graph: ?*MirageGraph,
    lhs: MirageTensor,
    rhs: MirageTensor,
    out_tensor: *MirageTensor,
) MirageStatus {
    const f = fn_graph_matmul orelse return .internal_error;
    return f(graph, lhs, rhs, out_tensor);
}

pub fn mirage_graph_unary(
    graph: ?*MirageGraph,
    op: MirageUnaryOp,
    input: MirageTensor,
    out_tensor: *MirageTensor,
) MirageStatus {
    const f = fn_graph_unary orelse return .internal_error;
    return f(graph, op, input, out_tensor);
}

pub fn mirage_graph_binary(
    graph: ?*MirageGraph,
    op: MirageBinaryOp,
    lhs: MirageTensor,
    rhs: MirageTensor,
    out_tensor: *MirageTensor,
) MirageStatus {
    const f = fn_graph_binary orelse return .internal_error;
    return f(graph, op, lhs, rhs, out_tensor);
}

pub fn mirage_graph_mark_output(graph: ?*MirageGraph, tensor: MirageTensor) MirageStatus {
    const f = fn_graph_mark_output orelse return .internal_error;
    return f(graph, tensor);
}

pub fn mirage_graph_superoptimize(graph: ?*MirageGraph, config_name: ?[*:0]const u8) MirageStatus {
    const f = fn_graph_superoptimize orelse return .internal_error;
    return f(graph, config_name);
}

pub fn mirage_graph_superoptimize_with_options(
    graph: ?*MirageGraph,
    config_name: ?[*:0]const u8,
    options: ?*const SuperoptOptions,
) MirageStatus {
    if (fn_graph_superoptimize_with_options) |f| {
        return f(graph, config_name, options);
    }
    // Older runtimes may only expose `mirage_graph_superoptimize`.
    return mirage_graph_superoptimize(graph, config_name);
}

pub fn mirage_graph_compile(
    ctx: ?*MirageContext,
    graph: ?*MirageGraph,
    target_cc: c_int,
    out_artifact_ptr: *[*]const u8,
    out_artifact_len: *usize,
    out_launch_info: *LaunchInfo,
) MirageStatus {
    const f = fn_graph_compile orelse return .internal_error;
    return f(ctx, graph, target_cc, out_artifact_ptr, out_artifact_len, out_launch_info);
}

pub fn mirage_execute_kernel(
    ctx: ?*MirageContext,
    artifact_ptr: [*]const u8,
    artifact_len: usize,
    params: *const DispatchParams,
    workspace: ?*anyopaque,
) MirageStatus {
    const f = fn_execute_kernel orelse return .internal_error;
    return f(ctx, artifact_ptr, artifact_len, params, workspace);
}

pub fn mirage_validate_artifact(
    ctx: ?*MirageContext,
    artifact_ptr: [*]const u8,
    artifact_len: usize,
) MirageStatus {
    const f = fn_validate_artifact orelse return .internal_error;
    return f(ctx, artifact_ptr, artifact_len);
}

pub fn mirage_release_buffer(ctx: ?*MirageContext, buffer_ptr: [*]const u8, buffer_len: usize) void {
    const f = fn_release_buffer orelse return;
    f(ctx, buffer_ptr, buffer_len);
}

/// Release the Mirage DeviceMemoryManager singleton and its GPU allocations.
/// No-op if the runtime doesn't expose this symbol (older builds).
pub fn mirage_release_device_memory() void {
    const f = fn_release_device_memory orelse {
        log.warn("mirage_release_device_memory symbol not found; skipping device memory release", .{});
        return;
    };
    log.info("releasing mirage device memory", .{});
    f();
}

/// Query device memory from the underlying runtime (e.g. cudaMemGetInfo).
/// Returns null if the symbol is unavailable or the query fails.
pub fn mirage_device_mem_info() ?struct { free: usize, total: usize } {
    const f = fn_device_mem_info orelse return null;
    var free: usize = 0;
    var total: usize = 0;
    const st = f(&free, &total);
    if (st != .ok) return null;
    return .{ .free = free, .total = total };
}

fn load_symbol(comptime T: type, handle: *anyopaque, comptime symbol: [:0]const u8) LoadError!T {
    clear_dlerror();
    const raw = dlsym(handle, symbol.ptr) orelse {
        if (dlerror()) |err| {
            log.err("missing symbol {s}: {s}", .{ symbol, std.mem.span(err) });
        } else {
            log.err("missing symbol {s}", .{symbol});
        }
        return error.MirageSymbolMissing;
    };
    return @ptrCast(raw);
}

fn load_symbol_optional(comptime T: type, handle: *anyopaque, comptime symbol: [:0]const u8) ?T {
    clear_dlerror();
    const raw = dlsym(handle, symbol.ptr) orelse {
        _ = dlerror();
        return null;
    };
    return @ptrCast(raw);
}

fn clear_dlerror() void {
    _ = dlerror();
}
