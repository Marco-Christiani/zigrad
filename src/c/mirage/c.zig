//! Mirage C ABI declarations with runtime symbol loading.
const std = @import("std");

const log = std.log.scoped(.@"zg/mirage_cffi");

pub const MirageContext = opaque {};

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

pub const PayloadV1 = extern struct {
    eqn_count: u32,
    num_inputs: u32,
    num_outputs: u32,
    launcher_so_path_len: u32,
};

const FnStatusString = *const fn (MirageStatus) callconv(.c) [*:0]const u8;
const FnContextCreate = *const fn (out_ctx: *?*MirageContext) callconv(.c) MirageStatus;
const FnContextDestroy = *const fn (ctx: ?*MirageContext) callconv(.c) void;
const FnCompileKernel = *const fn (
    ctx: ?*MirageContext,
    payload_ptr: [*]const u8,
    payload_len: usize,
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

var fn_status_string: ?FnStatusString = null;
var fn_context_create: ?FnContextCreate = null;
var fn_context_destroy: ?FnContextDestroy = null;
var fn_compile_kernel: ?FnCompileKernel = null;
var fn_execute_kernel: ?FnExecuteKernel = null;
var fn_validate_artifact: ?FnValidateArtifact = null;
var fn_release_buffer: ?FnReleaseBuffer = null;

var symbols_ready = false;

pub const LoadError = error{
    MirageSymbolMissing,
};

extern "c" fn dlsym(handle: *anyopaque, symbol: [*:0]const u8) ?*anyopaque;
extern "c" fn dlerror() ?[*:0]const u8;

pub fn ensure_loaded(handle: *anyopaque) LoadError!void {
    if (symbols_ready) return;

    fn_status_string = try load_symbol(FnStatusString, handle, "mirage_status_string");
    fn_context_create = try load_symbol(FnContextCreate, handle, "mirage_context_create");
    fn_context_destroy = try load_symbol(FnContextDestroy, handle, "mirage_context_destroy");
    fn_compile_kernel = try load_symbol(FnCompileKernel, handle, "mirage_compile_kernel");
    fn_execute_kernel = try load_symbol(FnExecuteKernel, handle, "mirage_execute_kernel");
    fn_validate_artifact = try load_symbol(FnValidateArtifact, handle, "mirage_validate_artifact");
    fn_release_buffer = try load_symbol(FnReleaseBuffer, handle, "mirage_release_buffer");

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

pub fn mirage_compile_kernel(
    ctx: ?*MirageContext,
    payload_ptr: [*]const u8,
    payload_len: usize,
    out_artifact_ptr: *[*]const u8,
    out_artifact_len: *usize,
    out_launch_info: *LaunchInfo,
) MirageStatus {
    const f = fn_compile_kernel orelse return .internal_error;
    return f(ctx, payload_ptr, payload_len, out_artifact_ptr, out_artifact_len, out_launch_info);
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

fn load_symbol(comptime T: type, handle: *anyopaque, comptime symbol: [:0]const u8) LoadError!T {
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
