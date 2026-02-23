//! TVM FFI call infrastructure.
//!
//! Provides Value (wraps TVMFFIAny), ObjectHandle (refcounted), library
//! loading (dlopen with RTLD_GLOBAL), and call_global / create_packed_func
//! helpers. All typed wrappers (tir.zig, runtime.zig, etc.) are built on top of this layer.
const std = @import("std");
const c = @import("c.zig");

const log = std.log.scoped(.@"zg/tvm_api");

// ============================================================================
// Value — Zig wrapper for TVMFFIAny
// ============================================================================

/// Zig-side representation of a TVM value. Every TVM FFI call takes and
/// returns values of this type. Constructors set `type_index` correctly;
/// accessors check it and return null on type mismatch.
pub const Value = struct {
    raw: c.TVMFFIAny,

    pub fn none() Value {
        var v = std.mem.zeroes(c.TVMFFIAny);
        v.type_index = c.kTVMFFINone;
        return .{ .raw = v };
    }

    pub fn int(val: i64) Value {
        var v = std.mem.zeroes(c.TVMFFIAny);
        v.type_index = c.kTVMFFIInt;
        v.unnamed_1.v_int64 = val;
        return .{ .raw = v };
    }

    pub fn boolean(val: bool) Value {
        var v = std.mem.zeroes(c.TVMFFIAny);
        v.type_index = c.kTVMFFIBool;
        v.unnamed_1.v_int64 = if (val) 1 else 0;
        return .{ .raw = v };
    }

    pub fn float(val: f64) Value {
        var v = std.mem.zeroes(c.TVMFFIAny);
        v.type_index = c.kTVMFFIFloat;
        v.unnamed_1.v_float64 = val;
        return .{ .raw = v };
    }

    pub fn str(s: [*:0]const u8) Value {
        var v = std.mem.zeroes(c.TVMFFIAny);
        v.type_index = c.kTVMFFIRawStr;
        v.unnamed_1.v_c_str = s;
        return .{ .raw = v };
    }

    pub fn from_object(handle: c.TVMFFIObjectHandle, type_index: i32) Value {
        var v = std.mem.zeroes(c.TVMFFIAny);
        v.type_index = type_index;
        v.unnamed_1.v_obj = @ptrCast(@alignCast(handle));
        return .{ .raw = v };
    }

    pub fn as_object(self: Value) c.TVMFFIObjectHandle {
        if (self.raw.type_index < c.kTVMFFIStaticObjectBegin) return null;
        const obj = self.raw.unnamed_1.v_obj;
        return @ptrCast(@alignCast(obj));
    }

    pub fn as_int(self: Value) ?i64 {
        if (self.raw.type_index != c.kTVMFFIInt) return null;
        return self.raw.unnamed_1.v_int64;
    }

    pub fn as_float(self: Value) ?f64 {
        if (self.raw.type_index != c.kTVMFFIFloat) return null;
        return self.raw.unnamed_1.v_float64;
    }

    /// Extract a Zig string from a TVM string value (kTVMFFISmallStr or kTVMFFIStr).
    /// Caller owns the returned slice.
    pub fn as_string(self: *Value, allocator: std.mem.Allocator) ![]u8 {
        if (self.raw.type_index == c.kTVMFFISmallStr) {
            const n: usize = @intCast(self.raw.unnamed_0.small_str_len);
            return try allocator.dupe(u8, self.raw.unnamed_1.v_bytes[0..n]);
        }
        if (self.raw.type_index == c.kTVMFFIStr) {
            const obj: c.TVMFFIObjectHandle = @ptrCast(self.raw.unnamed_1.v_obj);
            defer _ = c.TVMFFIObjectDecRef(obj);

            const hdr_size = @sizeOf(c.TVMFFIObject);
            const ba_ptr: *const c.TVMFFIByteArray = @ptrCast(@alignCast(@as([*]const u8, @ptrCast(obj)) + hdr_size));
            if (ba_ptr.data == null or ba_ptr.size == 0) return try allocator.dupe(u8, "");
            return try allocator.dupe(u8, ba_ptr.data[0..ba_ptr.size]);
        }
        return error.UnexpectedTvmType;
    }

    /// Extract an integer, handling both raw kTVMFFIInt and IntImm objects.
    ///
    /// TVM sometimes returns integers as raw `kTVMFFIInt` values and sometimes
    /// as `IntImm` objects with a `value` field. This method tries both.
    pub fn to_int(self: Value) ?i64 {
        if (self.as_int()) |v| return v;
        if (get_field(self, "value")) |field_val| return field_val.as_int() else |_| {}
        return null;
    }

    pub fn is_none(self: Value) bool {
        return self.raw.type_index == c.kTVMFFINone;
    }

    /// Decrement the refcount of the underlying object, if any. Safe to call
    /// on non-object values (no-op). Use with `defer val.decref()`.
    pub fn decref(self: Value) void {
        if (self.raw.type_index >= c.kTVMFFIStaticObjectBegin) {
            if (self.raw.unnamed_1.v_obj) |obj| {
                _ = c.TVMFFIObjectDecRef(@ptrCast(@alignCast(obj)));
            }
        }
    }
};

// ============================================================================
// ObjectHandle — refcounted TVM object
// ============================================================================

/// Refcounted TVM object handle. Base building block for all typed wrappers.
/// On deinit, decrements the TVM-side reference count.
pub const ObjectHandle = struct {
    ptr: c.TVMFFIObjectHandle,

    pub fn deinit(self: *ObjectHandle) void {
        if (self.ptr) |p| {
            _ = c.TVMFFIObjectDecRef(p);
            self.ptr = null;
        }
    }

    pub fn incref(self: ObjectHandle) void {
        if (self.ptr) |p| _ = c.TVMFFIObjectIncRef(p);
    }

    pub fn to_value(self: ObjectHandle, type_index: i32) Value {
        return Value.from_object(self.ptr, type_index);
    }
};

// ============================================================================
// Comptime helpers — generate boilerplate for ObjectHandle-based types
// ============================================================================

/// Comptime generators for ObjectHandle-based TVM types. Adapted from the
/// pattern in `src/c/mlir/mlir.zig:helpers`, tailored for TVM's ObjectHandle
/// convention instead of MLIR's `_inner` convention.
pub const helpers = struct {
    /// Generate deinit for types with a `handle: ObjectHandle` field.
    pub fn deinit(T: type) fn (*T) void {
        return struct {
            fn f(self: *T) void {
                self.handle.deinit();
            }
        }.f;
    }

    /// Generate as_value for types that carry a runtime `type_index: c_int`.
    pub fn as_value(T: type) fn (T) Value {
        return struct {
            fn f(self: T) Value {
                return self.handle.to_value(self.type_index);
            }
        }.f;
    }

    /// Generate as_value for types with a fixed comptime-known type_index.
    pub fn as_value_fixed(T: type, comptime type_index: c_int) fn (T) Value {
        return struct {
            fn f(self: T) Value {
                return self.handle.to_value(type_index);
            }
        }.f;
    }

    /// Generate wrap that constructs `?T` from a Value (null if not an object).
    pub fn wrap(T: type) fn (Value) ?T {
        return struct {
            fn f(val: Value) ?T {
                const obj = val.as_object() orelse return null;
                return if (@hasField(T, "type_index"))
                    .{ .handle = .{ .ptr = obj }, .type_index = val.raw.type_index }
                else
                    .{ .handle = .{ .ptr = obj } };
            }
        }.f;
    }
};

// ============================================================================
// Library loading
// ============================================================================

const RTLD_NOW: c_int = 0x2;
const RTLD_GLOBAL: c_int = 0x100;
extern "c" fn dlopen(filename: [*:0]const u8, flags: c_int) ?*anyopaque;
extern "c" fn dlerror() ?[*:0]const u8;

var ffi_lib_handle: ?*anyopaque = null;
var compiler_lib_handle: ?*anyopaque = null;

/// Initialize TVM runtime: dlopen libtvm_ffi.so and libtvm.so with RTLD_GLOBAL.
///
/// The Zig linker loads TVM with RTLD_LOCAL. We reload with RTLD_GLOBAL so
/// compiled TVM modules can find runtime symbols. Also loads libtvm.so
/// (full compiler with TE/codegen) which registers TE functions via static
/// initializers.
pub fn ensure_loaded(allocator: std.mem.Allocator) !void {
    // libtvm_ffi.so
    if (ffi_lib_handle == null) {
        ffi_lib_handle = dlopen("libtvm_ffi.so", RTLD_NOW | RTLD_GLOBAL);
        if (ffi_lib_handle == null) {
            ffi_lib_handle = dlopen("libtvm_runtime.so", RTLD_NOW | RTLD_GLOBAL);
        }
        if (ffi_lib_handle == null) {
            if (dlerror()) |err| {
                log.err("failed to load TVM FFI runtime: {s}", .{std.mem.span(err)});
            }
            return error.TvmLoadFailed;
        }
        log.info("loaded libtvm_ffi.so with RTLD_GLOBAL", .{});
    }

    c.ensure_loaded(ffi_lib_handle.?) catch |err| {
        log.err("failed to resolve TVM FFI symbols: {s}", .{@errorName(err)});
        return error.TvmLoadFailed;
    };

    if (compiler_lib_handle != null) return;

    const lib_path = try find_tvm_lib_path(allocator);
    defer if (lib_path) |p| allocator.free(p);

    if (lib_path) |path| {
        const path_z = try allocator.allocSentinel(u8, path.len, 0);
        defer allocator.free(path_z);
        @memcpy(path_z, path);

        compiler_lib_handle = dlopen(path_z, RTLD_NOW | RTLD_GLOBAL);
        if (compiler_lib_handle == null) {
            if (dlerror()) |err| {
                log.err("dlopen({s}) failed: {s}", .{ path, std.mem.span(err) });
            }
            return error.TvmLoadFailed;
        }
        log.info("loaded libtvm.so (compiler)", .{});
        return;
    }

    compiler_lib_handle = dlopen("libtvm.so", RTLD_NOW | RTLD_GLOBAL);
    if (compiler_lib_handle == null) {
        if (dlerror()) |err| {
            log.err("dlopen(libtvm.so) failed: {s}", .{std.mem.span(err)});
        }
        return error.TvmLoadFailed;
    }
    log.info("loaded libtvm.so (compiler)", .{});
}

/// Find libtvm.so by locating libtvm_ffi.so in /proc/self/maps and
/// looking in the same directory.
fn find_tvm_lib_path(allocator: std.mem.Allocator) !?[]const u8 {
    const maps_file = std.fs.openFileAbsolute("/proc/self/maps", .{}) catch return null;
    defer maps_file.close();

    var read_buf: [8192]u8 = undefined;
    var file_reader = maps_file.reader(&read_buf);
    const reader = &file_reader.interface;

    while (true) {
        const line = reader.takeDelimiter('\n') catch break;
        if (line == null) break;
        const l = line.?;
        if (std.mem.indexOf(u8, l, "libtvm_ffi.so")) |_| {
            if (std.mem.indexOf(u8, l, "/")) |path_start| {
                const path = l[path_start..];
                if (std.mem.lastIndexOf(u8, path, "/")) |slash| {
                    return try std.fmt.allocPrint(allocator, "{s}/libtvm.so", .{path[0..slash]});
                }
            }
        }
    }
    return null;
}

// ============================================================================
// Error handling
// ============================================================================

pub const TvmError = error{
    TvmCallFailed,
    TvmLoadFailed,
    TvmFunctionNotFound,
    UnexpectedTvmType,
    OutOfMemory,
};

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

// ============================================================================
// Call helpers
// ============================================================================

/// Look up a TVM global function by name. Caller must DecRef the returned handle.
pub fn get_global(allocator: std.mem.Allocator, name: []const u8) TvmError!c.TVMFFIObjectHandle {
    var name_arr: c.TVMFFIByteArray = .{ .data = name.ptr, .size = name.len };
    var out: c.TVMFFIObjectHandle = null;
    if (c.TVMFFIFunctionGetGlobal(&name_arr, &out) != 0 or out == null) {
        const msg = get_last_error_message(allocator) catch "(OOM reading error)";
        defer if (msg.len > 0) allocator.free(msg);
        log.err("TVMFFIFunctionGetGlobal({s}) failed: {s}", .{ name, msg });
        return error.TvmFunctionNotFound;
    }
    return out;
}

/// Call a TVM function handle with args, writing the result to `out`.
///
/// Critical: pre-initializes `out` to Value.none() before the call.
/// TVM's SafeCallImpl checks `result->type_index < kTVMFFIStaticObjectBegin`
/// BEFORE executing — uninitialized memory causes spurious CHECK failures.
pub fn call(allocator: std.mem.Allocator, func: c.TVMFFIObjectHandle, args: []const c.TVMFFIAny, out: *c.TVMFFIAny) TvmError!void {
    out.* = Value.none().raw;
    const arg_ptr = if (args.len == 0) null else @constCast(args.ptr);
    if (c.TVMFFIFunctionCall(func, arg_ptr, @intCast(args.len), out) != 0) {
        const msg = get_last_error_message(allocator) catch "(OOM reading error)";
        defer if (msg.len > 0) allocator.free(msg);
        log.err("TVMFFIFunctionCall failed: {s}", .{msg});
        return error.TvmCallFailed;
    }
}

/// Call a TVM global function by name with Value args, returning a Value.
pub fn call_global(allocator: std.mem.Allocator, func_name: []const u8, args: []const Value) TvmError!Value {
    const func = try get_global(allocator, func_name);
    defer _ = c.TVMFFIObjectDecRef(func);
    return call_with_values(allocator, func, args);
}

/// Call a TVM function handle with Value arguments.
pub fn call_handle(allocator: std.mem.Allocator, func: c.TVMFFIObjectHandle, args: []const Value) TvmError!Value {
    return call_with_values(allocator, func, args);
}

/// Shared implementation: convert Value args to raw TVMFFIAny and call.
/// Uses a stack buffer for <=16 args, heap for larger lists.
fn call_with_values(allocator: std.mem.Allocator, func: c.TVMFFIObjectHandle, args: []const Value) TvmError!Value {
    if (args.len <= 16) {
        var raw_args: [16]c.TVMFFIAny = undefined;
        for (args, 0..) |a, i| raw_args[i] = a.raw;
        var out: c.TVMFFIAny = undefined;
        try call(allocator, func, raw_args[0..args.len], &out);
        return .{ .raw = out };
    } else {
        const raw_args = try allocator.alloc(c.TVMFFIAny, args.len);
        defer allocator.free(raw_args);
        for (args, 0..) |a, i| raw_args[i] = a.raw;
        var out: c.TVMFFIAny = undefined;
        try call(allocator, func, raw_args, &out);
        return .{ .raw = out };
    }
}

/// Register a TVM global function by name.
pub fn set_global(name: []const u8, func_handle: c.TVMFFIObjectHandle, override: bool) TvmError!void {
    var name_arr: c.TVMFFIByteArray = .{ .data = name.ptr, .size = name.len };
    if (c.TVMFFIFunctionSetGlobal(&name_arr, func_handle, if (override) 1 else 0) != 0) {
        log.err("TVMFFIFunctionSetGlobal({s}) failed", .{name});
        return error.TvmCallFailed;
    }
}

// ============================================================================
// String helpers
// ============================================================================

// ============================================================================
// Packed function creation
// ============================================================================

/// TVM packed function callback signature.
pub const PackedFuncCallback = *const fn (
    self_ptr: ?*anyopaque,
    args: [*c]const c.TVMFFIAny,
    num_args: i32,
    result: [*c]c.TVMFFIAny,
) callconv(.c) c_int;

/// Destructor callback for packed function userdata.
pub const PackedFuncDestructor = *const fn (self_ptr: ?*anyopaque) callconv(.c) void;

/// Create a TVM packed function from a Zig callback with optional userdata.
/// Returns a Value wrapping the function object.
pub fn create_packed_func(
    self_ptr: ?*anyopaque,
    callback: PackedFuncCallback,
    destructor: ?PackedFuncDestructor,
) TvmError!Value {
    var func_handle: c.TVMFFIObjectHandle = null;
    if (c.TVMFFIFunctionCreate(
        self_ptr,
        callback,
        destructor,
        &func_handle,
    ) != 0) {
        log.err("TVMFFIFunctionCreate failed", .{});
        return error.TvmCallFailed;
    }
    return Value.from_object(func_handle, c.kTVMFFIFunction);
}

// ============================================================================
// Object reflection
// ============================================================================

/// Read a named field from a TVM object using the runtime reflection system.
///
/// Uses `TVMFFIGetTypeInfo` to look up the object's type metadata, finds the
/// field by name, then calls the field's getter at (obj_ptr + offset).
pub fn get_field(obj: Value, field_name: []const u8) TvmError!Value {
    if (obj.raw.type_index < c.kTVMFFIStaticObjectBegin) {
        log.warn("get_field: not an object (type_index={d})", .{obj.raw.type_index});
        return error.TvmCallFailed;
    }
    const obj_ptr = obj.raw.unnamed_1.v_obj orelse {
        log.warn("get_field: null object", .{});
        return error.TvmCallFailed;
    };

    const type_info: ?*const c.TVMFFITypeInfo = c.TVMFFIGetTypeInfo(obj.raw.type_index);
    if (type_info == null) {
        log.warn("get_field: TVMFFIGetTypeInfo({d}) returned null", .{obj.raw.type_index});
        return error.TvmCallFailed;
    }

    const num_fields: usize = @intCast(type_info.?.num_fields);
    const fields = type_info.?.fields orelse {
        log.warn("get_field: type has no fields (type_index={d})", .{obj.raw.type_index});
        return error.TvmCallFailed;
    };

    // Search for field by name
    var found_field: ?*const c.TVMFFIFieldInfo = null;
    for (0..num_fields) |i| {
        const field = &fields[i];
        const name = field.name;
        if (name.data != null and name.size == field_name.len) {
            if (std.mem.eql(u8, name.data[0..name.size], field_name)) {
                found_field = field;
                break;
            }
        }
    }

    const field = found_field orelse {
        log.debug("get_field: '{s}' not found (type_index={d}, {d} fields)", .{
            field_name, obj.raw.type_index, num_fields,
        });
        return error.TvmCallFailed;
    };

    const getter = field.getter orelse {
        log.warn("get_field: '{s}' has no getter", .{field_name});
        return error.TvmCallFailed;
    };

    // Compute field address: obj_ptr + offset
    const offset: usize = @intCast(field.offset);
    const field_ptr: *anyopaque = @ptrCast(@as([*]u8, @ptrCast(obj_ptr)) + offset);

    var result: c.TVMFFIAny = Value.none().raw;
    if (getter(field_ptr, &result) != 0) {
        log.warn("get_field: getter for '{s}' failed", .{field_name});
        return error.TvmCallFailed;
    }
    return .{ .raw = result };
}

/// List all TVM global function names, sorted alphabetically.
///
/// Uses `ffi.FunctionListGlobalNamesFunctor` to enumerate all registered
/// TVM functions. Caller owns the returned slice and each name within it.
pub fn list_global_names(allocator: std.mem.Allocator) ![][]const u8 {
    const functor_val = try call_global(allocator, "ffi.FunctionListGlobalNamesFunctor", &.{});
    defer functor_val.decref();

    const functor_handle = functor_val.as_object() orelse return error.TvmCallFailed;

    // functor(-1) returns the count
    const count_val = try call_handle(allocator, functor_handle, &.{Value.int(-1)});
    const count: usize = @intCast(count_val.as_int() orelse return error.TvmCallFailed);

    var names = try std.ArrayList([]const u8).initCapacity(allocator, count);
    errdefer {
        for (names.items) |n| allocator.free(n);
        names.deinit(allocator);
    }

    for (0..count) |i| {
        var name_val = call_handle(allocator, functor_handle, &.{Value.int(@intCast(i))}) catch continue;
        const s = name_val.as_string(allocator) catch continue;
        try names.append(allocator, s);
    }

    const items = try names.toOwnedSlice(allocator);
    std.mem.sort([]const u8, items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lessThan);
    return items;
}

/// Create a TVM String object from a Zig slice.
pub fn make_tvm_string(s: []const u8) TvmError!Value {
    var bytes: c.TVMFFIByteArray = .{ .data = s.ptr, .size = s.len };
    var out: c.TVMFFIAny = std.mem.zeroes(c.TVMFFIAny);
    if (c.TVMFFIStringFromByteArray(&bytes, &out) != 0) {
        log.err("TVMFFIStringFromByteArray failed", .{});
        return error.TvmCallFailed;
    }
    return .{ .raw = out };
}

// ============================================================================
// Array — TVM runtime Array wrapper (tvm/ffi/container/array.h)
// ============================================================================

/// Typed wrapper for TVM's `ffi.Array`.
///
/// Provides typed access to array construction, length, and element access.
/// Owns the underlying TVM object handle and decrements its refcount on deinit.
pub const Array = struct {
    handle: ObjectHandle,
    type_index: c_int = c.kTVMFFIStaticObjectBegin,

    /// Wrap an existing TVM array Value, taking a reference.
    ///
    /// Increments the refcount so `deinit` is safe and symmetric.
    /// Use this for callback arguments where the caller retains ownership.
    pub fn wrap(val: Value) !Array {
        const obj = val.as_object() orelse return error.TvmCallFailed;
        _ = c.TVMFFIObjectIncRef(obj);
        return .{ .handle = .{ .ptr = obj }, .type_index = val.raw.type_index };
    }

    /// Construct a TVM Array from a slice of Values.
    pub fn from_values(allocator: std.mem.Allocator, items: []const Value) !Array {
        const result = try call_global(allocator, "ffi.Array", items);
        return .{
            .handle = .{ .ptr = result.as_object() orelse return error.TvmCallFailed },
            .type_index = result.raw.type_index,
        };
    }

    /// Number of elements in the array.
    pub fn len(self: Array, allocator: std.mem.Allocator) !usize {
        const result = try call_global(allocator, "ffi.ArraySize", &.{self.as_value()});
        return @intCast(result.as_int() orelse return error.TvmCallFailed);
    }

    /// Get the element at `idx`.
    pub fn get(self: Array, allocator: std.mem.Allocator, idx: usize) !Value {
        return call_global(allocator, "ffi.ArrayGetItem", &.{
            self.as_value(), Value.int(@intCast(idx)),
        });
    }

    pub const as_value = helpers.as_value(Array);
    pub const deinit = helpers.deinit(Array);
};

// ============================================================================
// Map — TVM runtime Map wrapper (tvm/ffi/container/map.h)
// ============================================================================

/// Typed wrapper for TVM's `ffi.Map`.
///
/// Provides typed access to map construction, size, key existence, and value
/// lookup. Owns the underlying TVM object handle and decrements its refcount
/// on deinit.
pub const Map = struct {
    handle: ObjectHandle,
    type_index: c_int = c.kTVMFFIStaticObjectBegin,

    /// Wrap an existing TVM Map Value, taking a reference.
    ///
    /// Increments the refcount so `deinit` is safe and symmetric.
    pub fn wrap(val: Value) !Map {
        const obj = val.as_object() orelse return error.TvmCallFailed;
        _ = c.TVMFFIObjectIncRef(obj);
        return .{ .handle = .{ .ptr = obj }, .type_index = val.raw.type_index };
    }

    /// Construct a TVM Map from alternating key-value pairs.
    pub fn from_pairs(allocator: std.mem.Allocator, pairs: []const Value) !Map {
        const result = try call_global(allocator, "ffi.Map", pairs);
        return .{
            .handle = .{ .ptr = result.as_object() orelse return error.TvmCallFailed },
            .type_index = result.raw.type_index,
        };
    }

    /// Number of entries in the map.
    pub fn len(self: Map, allocator: std.mem.Allocator) !usize {
        const result = try call_global(allocator, "ffi.MapSize", &.{self.as_value()});
        return @intCast(result.as_int() orelse return error.TvmCallFailed);
    }

    /// Check if a key exists. Key must be a TVM object (e.g. TVM String).
    pub fn contains(self: Map, allocator: std.mem.Allocator, key: Value) !bool {
        const result = try call_global(allocator, "ffi.MapCount", &.{ self.as_value(), key });
        return (result.as_int() orelse 0) != 0;
    }

    /// Get value by key.
    pub fn get(self: Map, allocator: std.mem.Allocator, key: Value) !Value {
        return call_global(allocator, "ffi.MapGetItem", &.{ self.as_value(), key });
    }

    pub const as_value = helpers.as_value(Map);
    pub const deinit = helpers.deinit(Map);
};

// ============================================================================
// Node utilities
// ============================================================================

/// Deserialize a TVM object from its JSON representation.
pub fn load_json(allocator: std.mem.Allocator, json: [:0]const u8) TvmError!Value {
    return call_global(allocator, "node.LoadJSON", &.{Value.str(json)});
}
