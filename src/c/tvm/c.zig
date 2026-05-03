//! TVM C API FFI bindings.
//!
//! This module always provides TVM C types/constants at compile time (headers
//! must be present), while function symbols are resolved at runtime.

const std = @import("std");

const log = std.log.scoped(.@"zg/tvm_cffi");

const c = @import("c-tvm");

pub const TVMFFIAny = c.TVMFFIAny;
pub const TVMFFIObjectHandle = c.TVMFFIObjectHandle;
pub const TVMFFIByteArray = c.TVMFFIByteArray;
pub const TVMFFIErrorCell = c.TVMFFIErrorCell;
pub const TVMFFIFieldInfo = c.TVMFFIFieldInfo;
pub const TVMFFIObject = c.TVMFFIObject;
pub const TVMFFITypeInfo = c.TVMFFITypeInfo;
pub const DLManagedTensor = c.struct_DLManagedTensor;

pub const kTVMFFINone = c.kTVMFFINone;
pub const kTVMFFIInt = c.kTVMFFIInt;
pub const kTVMFFIFloat = c.kTVMFFIFloat;
pub const kTVMFFIBool = c.kTVMFFIBool;
pub const kTVMFFIRawStr = c.kTVMFFIRawStr;
pub const kTVMFFITensor = c.kTVMFFITensor;
pub const kTVMFFIOpaquePtr = c.kTVMFFIOpaquePtr;
pub const kTVMFFIModule = c.kTVMFFIModule;
pub const kTVMFFIFunction = c.kTVMFFIFunction;
pub const kTVMFFISmallStr = c.kTVMFFISmallStr;
pub const kTVMFFIStr = c.kTVMFFIStr;
pub const kTVMFFIDataType = c.kTVMFFIDataType;
pub const kTVMFFIDevice = c.kTVMFFIDevice;
pub const kTVMFFIStaticObjectBegin = c.kTVMFFIStaticObjectBegin;

const PackedCFunc = *const fn (?*anyopaque, [*c]const TVMFFIAny, i32, [*c]TVMFFIAny) callconv(.c) c_int;
const PackedCFuncFinalizer = *const fn (?*anyopaque) callconv(.c) void;

const FnObjectDecRef = *const fn (TVMFFIObjectHandle) callconv(.c) c_int;
const FnObjectIncRef = *const fn (TVMFFIObjectHandle) callconv(.c) c_int;
const FnFunctionCreate = *const fn (?*anyopaque, ?PackedCFunc, ?PackedCFuncFinalizer, *TVMFFIObjectHandle) callconv(.c) c_int;
const FnFunctionCall = *const fn (TVMFFIObjectHandle, ?[*]TVMFFIAny, i32, *TVMFFIAny) callconv(.c) c_int;
const FnFunctionGetGlobal = *const fn (*TVMFFIByteArray, *TVMFFIObjectHandle) callconv(.c) c_int;
const FnFunctionSetGlobal = *const fn (*TVMFFIByteArray, TVMFFIObjectHandle, i32) callconv(.c) c_int;
const FnErrorMoveFromRaised = *const fn (*TVMFFIObjectHandle) callconv(.c) void;
const FnGetTypeInfo = *const fn (i32) callconv(.c) ?*const TVMFFITypeInfo;
const FnStringFromByteArray = *const fn (*TVMFFIByteArray, *TVMFFIAny) callconv(.c) c_int;
const FnTensorFromDLPack = *const fn ([*c]DLManagedTensor, i32, i32, *TVMFFIObjectHandle) callconv(.c) c_int;

var fn_object_dec_ref: ?FnObjectDecRef = null;
var fn_object_inc_ref: ?FnObjectIncRef = null;
var fn_function_create: ?FnFunctionCreate = null;
var fn_function_call: ?FnFunctionCall = null;
var fn_function_get_global: ?FnFunctionGetGlobal = null;
var fn_function_set_global: ?FnFunctionSetGlobal = null;
var fn_error_move_from_raised: ?FnErrorMoveFromRaised = null;
var fn_get_type_info: ?FnGetTypeInfo = null;
var fn_string_from_byte_array: ?FnStringFromByteArray = null;
var fn_tensor_from_dlpack: ?FnTensorFromDLPack = null;

var symbols_ready = false;

pub const LoadError = error{
    TvmSymbolMissing,
};

extern "c" fn dlsym(handle: *anyopaque, symbol: [*:0]const u8) ?*anyopaque;
extern "c" fn dlerror() ?[*:0]const u8;

pub fn ensure_loaded(handle: *anyopaque) LoadError!void {
    if (symbols_ready) return;

    fn_object_dec_ref = try load_symbol(FnObjectDecRef, handle, "TVMFFIObjectDecRef");
    fn_object_inc_ref = try load_symbol(FnObjectIncRef, handle, "TVMFFIObjectIncRef");
    fn_function_create = try load_symbol(FnFunctionCreate, handle, "TVMFFIFunctionCreate");
    fn_function_call = try load_symbol(FnFunctionCall, handle, "TVMFFIFunctionCall");
    fn_function_get_global = try load_symbol(FnFunctionGetGlobal, handle, "TVMFFIFunctionGetGlobal");
    fn_function_set_global = try load_symbol(FnFunctionSetGlobal, handle, "TVMFFIFunctionSetGlobal");
    fn_error_move_from_raised = try load_symbol(FnErrorMoveFromRaised, handle, "TVMFFIErrorMoveFromRaised");
    fn_get_type_info = try load_symbol(FnGetTypeInfo, handle, "TVMFFIGetTypeInfo");
    fn_string_from_byte_array = try load_symbol(FnStringFromByteArray, handle, "TVMFFIStringFromByteArray");
    fn_tensor_from_dlpack = try load_symbol(FnTensorFromDLPack, handle, "TVMFFITensorFromDLPack");

    symbols_ready = true;
}

pub fn TVMFFIObjectDecRef(handle: TVMFFIObjectHandle) c_int {
    const f = fn_object_dec_ref orelse return -1;
    return f(handle);
}

pub fn TVMFFIObjectIncRef(handle: TVMFFIObjectHandle) c_int {
    const f = fn_object_inc_ref orelse return -1;
    return f(handle);
}

pub fn TVMFFIFunctionCreate(
    self_ptr: ?*anyopaque,
    callback: ?PackedCFunc,
    destructor: ?PackedCFuncFinalizer,
    out: *TVMFFIObjectHandle,
) c_int {
    const f = fn_function_create orelse return -1;
    return f(self_ptr, callback, destructor, out);
}

pub fn TVMFFIFunctionCall(
    func: TVMFFIObjectHandle,
    args: ?[*]TVMFFIAny,
    num_args: i32,
    out: *TVMFFIAny,
) c_int {
    const f = fn_function_call orelse return -1;
    return f(func, args, num_args, out);
}

pub fn TVMFFIFunctionGetGlobal(name: *TVMFFIByteArray, out: *TVMFFIObjectHandle) c_int {
    const f = fn_function_get_global orelse return -1;
    return f(name, out);
}

pub fn TVMFFIFunctionSetGlobal(name: *TVMFFIByteArray, func: TVMFFIObjectHandle, override: i32) c_int {
    const f = fn_function_set_global orelse return -1;
    return f(name, func, override);
}

pub fn TVMFFIErrorMoveFromRaised(out: *TVMFFIObjectHandle) void {
    const f = fn_error_move_from_raised orelse return;
    f(out);
}

pub fn TVMFFIGetTypeInfo(type_index: i32) ?*const TVMFFITypeInfo {
    const f = fn_get_type_info orelse return null;
    return f(type_index);
}

pub fn TVMFFIStringFromByteArray(bytes: *TVMFFIByteArray, out: *TVMFFIAny) c_int {
    const f = fn_string_from_byte_array orelse return -1;
    return f(bytes, out);
}

pub fn TVMFFITensorFromDLPack(
    managed_tensor: [*c]DLManagedTensor,
    manager_ctx_offset: i32,
    is_view: i32,
    out: *TVMFFIObjectHandle,
) c_int {
    const f = fn_tensor_from_dlpack orelse return -1;
    return f(managed_tensor, manager_ctx_offset, is_view, out);
}

fn load_symbol(comptime T: type, handle: *anyopaque, comptime symbol: [:0]const u8) LoadError!T {
    const raw = dlsym(handle, symbol.ptr) orelse {
        if (dlerror()) |err| {
            log.err("missing symbol {s}: {s}", .{ symbol, std.mem.span(err) });
        } else {
            log.err("missing symbol {s}", .{symbol});
        }
        return error.TvmSymbolMissing;
    };
    return @ptrCast(raw);
}
