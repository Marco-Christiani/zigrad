//! IREE Runtime C API bindings (Zig-side).
//!
//! Thin wrappers over `./types.zig`, which hand-declares the IREE C ABI
//!  surface zigrad uses. We do not run translate-c on IREE because the
//!  upstream headers contain bitfield-bearing structs and `sizeof()`-based
//!  alignment expressions that translate-c cannot process. `types.zig`'s
//!  ABI test (`abi_test.zig`) protects the kept value-type layouts from
//!  upstream drift.
//!
//! libIREERuntime.so (linked from `libiree_runtime_unified.a`) is linked at
//!  build time when `-Diree-backend=true`. The C-side helpers in `shim.c`
//!  cover anything Zig cannot express directly.
//!
//! Error handling: IREE returns `iree_status_t`, non-OK statuses are
//!  converted to Zig errors and logged before propagation.
const std = @import("std");
const types = @import("types.zig");
const log = std.log.scoped(.@"zg/iree_runtime");

// Re-export commonly used types for callers.
pub const Instance = types.Instance;
pub const Session = types.Session;
pub const Call = types.RuntimeCall;
pub const HalDevice = types.HalDevice;
pub const HalAllocator = types.HalAllocator;
pub const HalBuffer = types.HalBuffer;
pub const HalBufferView = types.HalBufferView;
pub const VmFunction = types.VmFunction;
pub const Status = types.Status;
pub const HalDim = types.HalDim;
pub const HalElementType = types.HalElementType;
pub const Allocator = types.Allocator;

/// Element-type constants for callers that need to name types directly
///  (e.g. dtype dispatch in callers).
pub const HAL_ELEMENT_TYPE_BOOL_8 = types.HAL_ELEMENT_TYPE_BOOL_8;
pub const HAL_ELEMENT_TYPE_SINT_8 = types.HAL_ELEMENT_TYPE_SINT_8;
pub const HAL_ELEMENT_TYPE_UINT_8 = types.HAL_ELEMENT_TYPE_UINT_8;
pub const HAL_ELEMENT_TYPE_SINT_32 = types.HAL_ELEMENT_TYPE_SINT_32;
pub const HAL_ELEMENT_TYPE_UINT_32 = types.HAL_ELEMENT_TYPE_UINT_32;
pub const HAL_ELEMENT_TYPE_SINT_64 = types.HAL_ELEMENT_TYPE_SINT_64;
pub const HAL_ELEMENT_TYPE_UINT_64 = types.HAL_ELEMENT_TYPE_UINT_64;
pub const HAL_ELEMENT_TYPE_FLOAT_16 = types.HAL_ELEMENT_TYPE_FLOAT_16;
pub const HAL_ELEMENT_TYPE_FLOAT_32 = types.HAL_ELEMENT_TYPE_FLOAT_32;
pub const HAL_ELEMENT_TYPE_FLOAT_64 = types.HAL_ELEMENT_TYPE_FLOAT_64;
pub const HAL_ELEMENT_TYPE_BFLOAT_16 = types.HAL_ELEMENT_TYPE_BFLOAT_16;

// ---------------------------------------------------------------------------
// Status checking.
// ---------------------------------------------------------------------------

/// Check an IREE status, return a Zig error (and log the message) on failure.
pub fn check(status: Status) !void {
    if (types.zg_iree_status_is_ok(status)) return;

    // iree_status_to_string allocates, pass the system allocator by pointer.
    var alloc = types.zg_iree_allocator_system();
    var msg_ptr: [*c]u8 = null;
    var msg_len: types.HostSize = 0;
    if (types.iree_status_to_string(status, &alloc, &msg_ptr, &msg_len)) {
        log.err("iree status: {s}", .{msg_ptr[0..msg_len]});
        types.iree_allocator_free(alloc, msg_ptr);
    } else {
        log.err("iree status: (could not format message)", .{});
    }
    types.iree_status_free(status);
    return error.IreeError;
}

/// Wrap a Zig slice as an `iree_string_view_t`.
pub fn sv(s: []const u8) types.StringView {
    return .{ .data = s.ptr, .size = s.len };
}

/// Wrap a Zig slice as an `iree_const_byte_span_t`.
pub fn span(s: []const u8) types.ConstByteSpan {
    return .{ .data = s.ptr, .data_length = s.len };
}

// ---------------------------------------------------------------------------
// Instance lifecycle.
// ---------------------------------------------------------------------------

/// Create an IREE runtime instance with all available HAL drivers registered.
pub fn instance_create() !*Instance {
    var out: ?*Instance = null;
    try check(types.zg_iree_runtime_instance_create_all_drivers(&out));
    return out.?;
}

pub fn instance_release(instance: *Instance) void {
    types.iree_runtime_instance_release(instance);
}

// ---------------------------------------------------------------------------
// Device management.
// ---------------------------------------------------------------------------

/// Create the default HAL device for `driver_name` (e.g. "local-sync").
pub fn create_default_device(instance: *Instance, driver_name: []const u8) !*HalDevice {
    var out: ?*HalDevice = null;
    try check(types.zg_iree_runtime_instance_try_create_default_device(
        instance,
        sv(driver_name),
        &out,
    ));
    return out.?;
}

pub fn device_release(device: *HalDevice) void {
    types.iree_hal_device_release(device);
}

// ---------------------------------------------------------------------------
// Session lifecycle.
// ---------------------------------------------------------------------------

/// Create a session bound to `device`.
pub fn session_create(instance: *Instance, device: *HalDevice) !*Session {
    var out: ?*Session = null;
    try check(types.zg_iree_runtime_session_create_with_device_default(
        instance,
        device,
        &out,
    ));
    return out.?;
}

pub fn session_release(session: *Session) void {
    types.iree_runtime_session_release(session);
}

/// Append a VMFB module from a caller-managed memory slice.
///
/// Precondition: `vmfb` must outlive `session` (the session borrows the bytes
/// without copying when a null allocator is used).
pub fn session_append_module(session: *Session, vmfb: []const u8) !void {
    try check(types.iree_runtime_session_append_bytecode_module_from_memory(
        session,
        span(vmfb),
        types.zg_iree_allocator_null(),
    ));
}

/// Look up a function by its fully-qualified name (e.g. `"module.main"`).
pub fn session_lookup_function(session: *Session, name: []const u8) !VmFunction {
    var func: VmFunction = undefined;
    try check(types.iree_runtime_session_lookup_function(session, sv(name), &func));
    return func;
}

// ---------------------------------------------------------------------------
// Call lifecycle.
// ---------------------------------------------------------------------------

/// Initialize a reusable call object. Caller must call `call_deinit`.
pub fn call_init(session: *Session, function: VmFunction) !Call {
    var call: Call = undefined;
    try check(types.iree_runtime_call_initialize(session, function, &call));
    return call;
}

pub fn call_deinit(call: *Call) void {
    types.iree_runtime_call_deinitialize(call);
}

/// Invoke the call synchronously.
pub fn call_invoke(call: *Call) !void {
    try check(types.iree_runtime_call_invoke(call, 0));
}

/// Push a buffer view onto the call inputs list. The list retains a new
///  reference on the view, ownership of the caller's reference is unchanged.
pub fn call_push_buffer_view_input(call: *Call, view: *HalBufferView) !void {
    try check(types.iree_runtime_call_inputs_push_back_buffer_view(call, view));
}

/// Pop the next buffer view off the call outputs list.
///  Ownership transfers to the caller, who must call `buffer_view_release`.
pub fn call_pop_buffer_view_output(call: *Call) !*HalBufferView {
    var out: ?*HalBufferView = null;
    try check(types.iree_runtime_call_outputs_pop_front_buffer_view(call, &out));
    return out orelse error.NullBufferView;
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
    var out: ?*HalBufferView = null;
    try check(types.zg_iree_buffer_view_allocate_device_local_copy(
        device,
        shape.ptr,
        shape.len,
        element_type,
        data.ptr,
        data.len,
        &out,
    ));
    return out.?;
}

pub fn buffer_view_retain(view: *HalBufferView) void {
    types.iree_hal_buffer_view_retain(view);
}

pub fn buffer_view_release(view: *HalBufferView) void {
    types.iree_hal_buffer_view_release(view);
}

/// Return the number of elements in `view`.
pub fn buffer_view_element_count(view: *HalBufferView) usize {
    return types.iree_hal_buffer_view_element_count(view);
}

/// Return the element type of `view`.
pub fn buffer_view_element_type(view: *HalBufferView) HalElementType {
    return types.iree_hal_buffer_view_element_type(view);
}

/// Copy buffer view contents to host slice `dst`. Maps the buffer for read,
///  copies, and unmaps.
pub fn buffer_view_to_host(view: *HalBufferView, dst: []u8) !void {
    const buf = types.iree_hal_buffer_view_buffer(view) orelse return error.NullBuffer;
    try check(types.zg_iree_hal_buffer_read(buf, dst.ptr, dst.len));
}

/// Write host bytes `src` into a pre-existing buffer view (must be CPU-accessible).
pub fn buffer_view_from_host(view: *HalBufferView, src: []const u8) !void {
    const buf = types.iree_hal_buffer_view_buffer(view) orelse return error.NullBuffer;
    try check(types.zg_iree_hal_buffer_write(buf, src.ptr, src.len));
}

/// Return byte width for a HAL element type (integer division of bit_count / 8).
pub fn element_byte_width(etype: HalElementType) usize {
    return types.zg_iree_hal_element_bit_count(etype) / 8;
}

test {
    _ = @import("abi_test.zig");
}
