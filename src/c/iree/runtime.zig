/// IREE Runtime C API bindings.
///
/// Thin wrappers around `iree/runtime/api.h` (which transitively includes
/// the HAL, VM, and base C APIs).  libIREERuntime.so is linked at build time
/// when `-Diree-backend=true`.
///
/// Several IREE C API functions are `static inline` or macros that Zig's
/// `@cImport` cannot translate.  These are wrapped in `src/c/iree/shim.c`
/// and accessed here via `@cImport("iree_shim.h")` or `extern` declarations.
///
/// Error handling: IREE returns `iree_status_t`; non-OK statuses are
/// converted to Zig errors and logged before propagation.
const std = @import("std");
const log = std.log.scoped(.@"zg/iree_runtime");

pub const c = @cImport({
    @cInclude("iree/runtime/api.h");
    @cInclude("iree/hal/api.h");
    @cInclude("iree/vm/api.h");
    @cInclude("iree/base/api.h");
});

// Re-export commonly used types for callers.
pub const Instance = c.iree_runtime_instance_t;
pub const Session = c.iree_runtime_session_t;
pub const Call = c.iree_runtime_call_t;
pub const HalDevice = c.iree_hal_device_t;
pub const HalAllocator = c.iree_hal_allocator_t;
pub const HalBuffer = c.iree_hal_buffer_t;
pub const HalBufferView = c.iree_hal_buffer_view_t;
pub const VmList = c.iree_vm_list_t;
pub const VmFunction = c.iree_vm_function_t;
pub const VmRef = c.iree_vm_ref_t;
pub const Status = c.iree_status_t;
pub const HalDim = c.iree_hal_dim_t;
pub const HalElementType = c.iree_hal_element_type_t;
pub const HalBufferParams = c.iree_hal_buffer_params_t;
pub const HalBufferMapping = c.iree_hal_buffer_mapping_t;
pub const Allocator = c.iree_allocator_t;

// ---------------------------------------------------------------------------
// Shim externs -- wrappers for static inline / macro functions that @cImport
// cannot translate.  Implementations live in shim.c.
// ---------------------------------------------------------------------------

extern fn zg_iree_allocator_system() callconv(.c) Allocator;
extern fn zg_iree_allocator_null() callconv(.c) Allocator;
extern fn zg_iree_status_is_ok(status: Status) callconv(.c) bool;
extern fn zg_iree_hal_element_bit_count(element_type: HalElementType) callconv(.c) usize;
extern fn zg_iree_hal_buffer_view_deref(ref: VmRef) callconv(.c) ?*HalBufferView;
extern fn zg_iree_hal_buffer_view_retain_ref(view: *HalBufferView) callconv(.c) VmRef;

// ---------------------------------------------------------------------------
// Status checking.
// ---------------------------------------------------------------------------

/// Check an IREE status; return a Zig error (and log the message) on failure.
pub fn check(status: Status) !void {
    if (zg_iree_status_is_ok(status)) return;

    // iree_status_to_string allocates; pass the system allocator by pointer.
    var alloc = zg_iree_allocator_system();
    var msg_ptr: [*c]u8 = null;
    var msg_len: c.iree_host_size_t = 0;
    if (c.iree_status_to_string(status, &alloc, &msg_ptr, &msg_len)) {
        log.err("iree status: {s}", .{msg_ptr[0..msg_len]});
        c.iree_allocator_free(alloc, msg_ptr);
    } else {
        log.err("iree status: (could not format message)", .{});
    }
    c.iree_status_free(status);
    return error.IreeError;
}

/// Wrap a Zig slice as an `iree_string_view_t`.
pub fn sv(s: []const u8) c.iree_string_view_t {
    return .{ .data = s.ptr, .size = s.len };
}

/// Wrap a Zig slice as an `iree_const_byte_span_t`.
pub fn span(s: []const u8) c.iree_const_byte_span_t {
    return .{ .data = s.ptr, .data_length = s.len };
}

// ---------------------------------------------------------------------------
// Instance lifecycle.
// ---------------------------------------------------------------------------

/// Create an IREE runtime instance with all available HAL drivers registered.
///
/// Caller must call `instance_release` when done.
pub fn instance_create() !*Instance {
    var opts: c.iree_runtime_instance_options_t = undefined;
    c.iree_runtime_instance_options_initialize(&opts);
    c.iree_runtime_instance_options_use_all_available_drivers(&opts);

    var out: ?*Instance = null;
    try check(c.iree_runtime_instance_create(&opts, zg_iree_allocator_system(), &out));
    return out.?;
}

pub fn instance_release(instance: *Instance) void {
    c.iree_runtime_instance_release(instance);
}

// ---------------------------------------------------------------------------
// Device management.
// ---------------------------------------------------------------------------

/// Create the default HAL device for `driver_name` (e.g. "local-sync").
pub fn create_default_device(instance: *Instance, driver_name: []const u8) !*HalDevice {
    var out: ?*HalDevice = null;
    try check(c.iree_runtime_instance_try_create_default_device(
        instance,
        sv(driver_name),
        &out,
    ));
    return out.?;
}

pub fn device_release(device: *HalDevice) void {
    c.iree_hal_device_release(device);
}

// ---------------------------------------------------------------------------
// Session lifecycle.
// ---------------------------------------------------------------------------

/// Create a session bound to `device`.
pub fn session_create(instance: *Instance, device: *HalDevice) !*Session {
    var opts: c.iree_runtime_session_options_t = undefined;
    c.iree_runtime_session_options_initialize(&opts);

    var out: ?*Session = null;
    try check(c.iree_runtime_session_create_with_device(
        instance,
        &opts,
        device,
        zg_iree_allocator_system(),
        &out,
    ));
    return out.?;
}

pub fn session_release(session: *Session) void {
    c.iree_runtime_session_release(session);
}

/// Append a VMFB module from a caller-managed memory slice.
///
/// Precondition: `vmfb` must outlive `session` (the session borrows the bytes
/// without copying when a null allocator is used).
pub fn session_append_module(session: *Session, vmfb: []const u8) !void {
    try check(c.iree_runtime_session_append_bytecode_module_from_memory(
        session,
        span(vmfb),
        zg_iree_allocator_null(),
    ));
}

/// Look up a function by its unqualified name (e.g. `"main"`).
pub fn session_lookup_function(session: *Session, name: []const u8) !VmFunction {
    var func: VmFunction = undefined;
    try check(c.iree_runtime_session_lookup_function(session, sv(name), &func));
    return func;
}

// ---------------------------------------------------------------------------
// Call lifecycle.
// ---------------------------------------------------------------------------

/// Initialize a reusable call object.  Caller must call `call_deinit`.
pub fn call_init(session: *Session, function: VmFunction) !Call {
    var call: Call = undefined;
    try check(c.iree_runtime_call_initialize(session, function, &call));
    return call;
}

pub fn call_deinit(call: *Call) void {
    c.iree_runtime_call_deinitialize(call);
}

/// Invoke the call synchronously.
pub fn call_invoke(call: *Call) !void {
    try check(c.iree_runtime_call_invoke(call, 0));
}

pub fn call_inputs(call: *Call) *VmList {
    return c.iree_runtime_call_inputs(call);
}

pub fn call_outputs(call: *Call) *VmList {
    return c.iree_runtime_call_outputs(call);
}

// ---------------------------------------------------------------------------
// VM list helpers (input/output passing).
// ---------------------------------------------------------------------------

pub fn list_size(list: *VmList) usize {
    return c.iree_vm_list_size(list);
}

/// Push `view` into `list`, retaining a new reference on the view.
pub fn list_push_buffer_view(list: *VmList, view: *HalBufferView) !void {
    var ref: VmRef = zg_iree_hal_buffer_view_retain_ref(view);
    try check(c.iree_vm_list_push_ref_move(list, &ref));
}

/// Get a buffer view from `list` at `index`.
///
/// Returns a retained reference -- caller must call `buffer_view_release`.
pub fn list_get_buffer_view(list: *VmList, index: usize) !*HalBufferView {
    var ref: VmRef = std.mem.zeroes(VmRef);
    try check(c.iree_vm_list_get_ref_retain(list, index, &ref));
    const view = zg_iree_hal_buffer_view_deref(ref) orelse return error.NullBufferView;
    return view;
}

// ---------------------------------------------------------------------------
// Buffer view lifecycle.
// ---------------------------------------------------------------------------

/// Allocate a device buffer view by copying `data` from host memory.
///
/// `shape` elements are in IREE's row-major (outermost-first) convention.
pub fn buffer_view_create_from_host(
    device: *HalDevice,
    data: []const u8,
    element_type: HalElementType,
    shape: []const HalDim,
) !*HalBufferView {
    const params: HalBufferParams = .{
        .type = c.IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        .usage = c.IREE_HAL_BUFFER_USAGE_DEFAULT,
        .access = c.IREE_HAL_MEMORY_ACCESS_ALL,
        .queue_affinity = c.IREE_HAL_QUEUE_AFFINITY_ANY,
        .min_compatibility = 0,
    };

    var out: ?*HalBufferView = null;
    try check(c.iree_hal_buffer_view_allocate_buffer_copy(
        device,
        c.iree_hal_device_allocator(device),
        shape.len,
        shape.ptr,
        element_type,
        c.IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        params,
        span(data),
        &out,
    ));
    return out.?;
}

pub fn buffer_view_retain(view: *HalBufferView) void {
    c.iree_hal_buffer_view_retain(view);
}

pub fn buffer_view_release(view: *HalBufferView) void {
    c.iree_hal_buffer_view_release(view);
}

/// Return the number of elements in `view`.
pub fn buffer_view_element_count(view: *HalBufferView) usize {
    return c.iree_hal_buffer_view_element_count(view);
}

/// Return the element type of `view`.
pub fn buffer_view_element_type(view: *HalBufferView) HalElementType {
    return c.iree_hal_buffer_view_element_type(view);
}

/// Copy buffer view contents to host slice `dst`.
///
/// Maps the buffer for read, copies, and unmaps.
pub fn buffer_view_to_host(view: *HalBufferView, dst: []u8) !void {
    const buf = c.iree_hal_buffer_view_buffer(view);
    var mapping: HalBufferMapping = undefined;
    try check(c.iree_hal_buffer_map_range(
        buf,
        c.IREE_HAL_MAPPING_MODE_SCOPED,
        c.IREE_HAL_MEMORY_ACCESS_READ,
        0,
        c.IREE_WHOLE_BUFFER,
        &mapping,
    ));
    defer _ = c.iree_hal_buffer_unmap_range(&mapping);

    const len = @min(mapping.contents.data_length, dst.len);
    @memcpy(dst[0..len], mapping.contents.data[0..len]);
}

/// Write host bytes `src` into a pre-existing buffer view (must be CPU-accessible).
pub fn buffer_view_from_host(view: *HalBufferView, src: []const u8) !void {
    const buf = c.iree_hal_buffer_view_buffer(view);
    var mapping: HalBufferMapping = undefined;
    try check(c.iree_hal_buffer_map_range(
        buf,
        c.IREE_HAL_MAPPING_MODE_SCOPED,
        c.IREE_HAL_MEMORY_ACCESS_WRITE,
        0,
        c.IREE_WHOLE_BUFFER,
        &mapping,
    ));
    defer _ = c.iree_hal_buffer_unmap_range(&mapping);

    const len = @min(mapping.contents.data_length, src.len);
    @memcpy(mapping.contents.data[0..len], src[0..len]);
}

// ---------------------------------------------------------------------------
// DType -> HAL element type mapping.
// ---------------------------------------------------------------------------

const pr = @import("../../pr/pr.zig");

/// Map a PR DType to the IREE HAL element type constant.
pub fn dtype_to_element_type(dtype: pr.DType) HalElementType {
    return switch (dtype) {
        .f32 => c.IREE_HAL_ELEMENT_TYPE_FLOAT_32,
        .f64 => c.IREE_HAL_ELEMENT_TYPE_FLOAT_64,
        .bf16 => c.IREE_HAL_ELEMENT_TYPE_BFLOAT_16,
        .i32 => c.IREE_HAL_ELEMENT_TYPE_SINT_32,
        .i64 => c.IREE_HAL_ELEMENT_TYPE_SINT_64,
        .u32 => c.IREE_HAL_ELEMENT_TYPE_UINT_32,
        .u64 => c.IREE_HAL_ELEMENT_TYPE_UINT_64,
        .bool => c.IREE_HAL_ELEMENT_TYPE_BOOL_8,
    };
}

/// Return byte width for a HAL element type (integer division of bit_count / 8).
pub fn element_byte_width(etype: HalElementType) usize {
    return zg_iree_hal_element_bit_count(etype) / 8;
}
