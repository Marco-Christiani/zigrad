//! IREE runtime C ABI adapter.
//!
//! Only this module imports the external declarations. Callers receive opaque
//!  Zigrad handles, Zig slices, scalars, and `ElementType`. IREE types and status
//!  values do not appear in its public API.
//!
//! The IREE runtime archive is linked when `-Diree=true`. Helpers in `shim.c`
//!  cover macros, inline functions, and bitfield-bearing structs that Zig
//!  cannot call or translate directly.

const std = @import("std");
const types = @import("types.zig");

const log = std.log.scoped(.@"zg/iree_runtime_abi");

/// Failures exposed by the IREE runtime adapter.
pub const Error = std.mem.Allocator.Error || error{
    IreeError,
    NullInstance,
    NullDevice,
    NullSession,
    NullBufferView,
    NullBuffer,
    InvalidDimension,
    UnsupportedElementType,
};

/// Element types supported by the Zigrad IREE adapter.
pub const ElementType = enum {
    bool,
    i8,
    u8,
    i32,
    u32,
    i64,
    u64,
    f16,
    bf16,
    f32,
    f64,

    /// Return the storage width of one element in bytes.
    pub fn byte_width(self: ElementType) usize {
        return switch (self) {
            .bool, .i8, .u8 => 1,
            .f16, .bf16 => 2,
            .i32, .u32, .f32 => 4,
            .i64, .u64, .f64 => 8,
        };
    }
};

/// Opaque handle for one IREE runtime instance.
pub const InstanceHandle = opaque {};

/// Opaque handle for one IREE HAL device.
pub const DeviceHandle = opaque {};

/// Opaque handle for one IREE runtime session.
pub const SessionHandle = opaque {};

/// Opaque handle for one resolved IREE VM function.
pub const FunctionHandle = opaque {};

/// Opaque handle for one initialized IREE runtime call.
pub const CallHandle = opaque {};

/// Opaque handle for one IREE HAL buffer view.
pub const BufferHandle = opaque {};

const FunctionState = struct {
    allocator: std.mem.Allocator,
    raw: types.VmFunction,
};

const CallState = struct {
    allocator: std.mem.Allocator,
    raw: types.RuntimeCall,
};

/// Create an IREE runtime instance with every linked HAL driver registered.
pub fn instance_create() Error!*InstanceHandle {
    var instance: ?*types.Instance = null;
    try check(types.zg_iree_runtime_instance_create_all_drivers(&instance));
    return instance_handle(instance orelse return error.NullInstance);
}

/// Create an IREE runtime instance without a HAL driver registry.
pub fn instance_create_without_drivers() Error!*InstanceHandle {
    var instance: ?*types.Instance = null;
    try check(types.zg_iree_runtime_instance_create(&instance));
    return instance_handle(instance orelse return error.NullInstance);
}

/// Release one runtime instance.
pub fn instance_release(instance: *InstanceHandle) void {
    types.iree_runtime_instance_release(raw_instance(instance));
}

/// Create the default device for one linked HAL driver.
pub fn create_default_device(
    instance: *InstanceHandle,
    driver_name: []const u8,
) Error!*DeviceHandle {
    var device: ?*types.HalDevice = null;
    try check(types.zg_iree_runtime_instance_try_create_default_device(
        raw_instance(instance),
        string_view(driver_name),
        &device,
    ));
    return device_handle(device orelse return error.NullDevice);
}

/// Create a synchronous CPU device with the embedded ELF loader.
pub fn create_embedded_elf_sync_device() Error!*DeviceHandle {
    var device: ?*types.HalDevice = null;
    try check(types.zg_iree_create_embedded_elf_sync_device(&device));
    return device_handle(device orelse return error.NullDevice);
}

/// Release one HAL device.
pub fn device_release(device: *DeviceHandle) void {
    types.iree_hal_device_release(raw_device(device));
}

/// Create a runtime session bound to `device`.
pub fn session_create(
    instance: *InstanceHandle,
    device: *DeviceHandle,
) Error!*SessionHandle {
    var session: ?*types.Session = null;
    try check(types.zg_iree_runtime_session_create_with_device_default(
        raw_instance(instance),
        raw_device(device),
        &session,
    ));
    return session_handle(session orelse return error.NullSession);
}

/// Release one runtime session.
pub fn session_release(session: *SessionHandle) void {
    types.iree_runtime_session_release(raw_session(session));
}

/// Append a borrowed VMFB module to `session`.
///
/// `vmfb` must outlive the session because IREE retains the supplied memory
///  without copying it.
pub fn session_append_module(
    session: *SessionHandle,
    vmfb: []const u8,
) Error!void {
    try check(types.iree_runtime_session_append_bytecode_module_from_memory(
        raw_session(session),
        byte_span(vmfb),
        types.zg_iree_allocator_null(),
    ));
}

/// Resolve one fully qualified VM function.
pub fn session_lookup_function(
    allocator: std.mem.Allocator,
    session: *SessionHandle,
    name: []const u8,
) Error!*FunctionHandle {
    const state = try allocator.create(FunctionState);
    errdefer allocator.destroy(state);
    state.* = .{
        .allocator = allocator,
        .raw = undefined,
    };
    try check(types.iree_runtime_session_lookup_function(
        raw_session(session),
        string_view(name),
        &state.raw,
    ));
    return @ptrCast(state);
}

/// Release one resolved VM function handle.
pub fn function_release(function: *FunctionHandle) void {
    const state = function_state(function);
    state.allocator.destroy(state);
}

/// Initialize a call that must be released with `call_deinit`.
pub fn call_init(
    allocator: std.mem.Allocator,
    session: *SessionHandle,
    function: *FunctionHandle,
) Error!*CallHandle {
    const state = try allocator.create(CallState);
    errdefer allocator.destroy(state);
    state.* = .{
        .allocator = allocator,
        .raw = undefined,
    };
    try check(types.iree_runtime_call_initialize(
        raw_session(session),
        function_state(function).raw,
        &state.raw,
    ));
    return @ptrCast(state);
}

/// Deinitialize and release one call.
pub fn call_deinit(call: *CallHandle) void {
    const state = call_state(call);
    types.iree_runtime_call_deinitialize(&state.raw);
    state.allocator.destroy(state);
}

/// Invoke one call synchronously.
pub fn call_invoke(call: *CallHandle) Error!void {
    try check(types.iree_runtime_call_invoke(&call_state(call).raw, 0));
}

/// Append one borrowed buffer to the call input list.
pub fn call_push_buffer_view_input(
    call: *CallHandle,
    buffer: *BufferHandle,
) Error!void {
    try check(types.iree_runtime_call_inputs_push_back_buffer_view(
        &call_state(call).raw,
        raw_buffer(buffer),
    ));
}

/// Pop an output buffer or return `null` when no outputs remain.
///
/// Release a returned buffer with `buffer_view_release`.
pub fn call_pop_buffer_view_output(
    call: *CallHandle,
) Error!?*BufferHandle {
    var buffer: ?*types.HalBufferView = null;
    const status = types.iree_runtime_call_outputs_pop_front_buffer_view(
        &call_state(call).raw,
        &buffer,
    );
    if (types.zg_iree_status_is_ok(status)) {
        return buffer_handle(buffer orelse return error.NullBufferView);
    }

    const code = types.zg_iree_status_code(status);
    if (code == types.STATUS_OUT_OF_RANGE or code == types.STATUS_NOT_FOUND) {
        types.iree_status_free(status);
        return null;
    }
    try check(status);
    unreachable;
}

/// Allocate a device buffer by copying row-major host data.
pub fn buffer_view_create_from_host(
    device: *DeviceHandle,
    data: []const u8,
    element_type: ElementType,
    shape: []const i64,
) Error!*BufferHandle {
    comptime std.debug.assert(@sizeOf(types.HalDim) == @sizeOf(i64));
    for (shape) |dimension| {
        if (dimension < 0) return error.InvalidDimension;
    }
    const raw_shape: []const types.HalDim = @ptrCast(shape);

    var buffer: ?*types.HalBufferView = null;
    try check(types.zg_iree_buffer_view_allocate_device_local_copy(
        raw_device(device),
        raw_shape.ptr,
        raw_shape.len,
        raw_element_type(element_type),
        data.ptr,
        data.len,
        &buffer,
    ));
    return buffer_handle(buffer orelse return error.NullBufferView);
}

/// Release one buffer view.
pub fn buffer_view_release(buffer: *BufferHandle) void {
    types.iree_hal_buffer_view_release(raw_buffer(buffer));
}

/// Return the number of logical elements in one buffer.
pub fn buffer_view_element_count(buffer: *BufferHandle) usize {
    return types.iree_hal_buffer_view_element_count(raw_buffer(buffer));
}

/// Return the Zigrad element type of one buffer.
pub fn buffer_view_element_type(buffer: *BufferHandle) Error!ElementType {
    return try element_type_from_raw(
        types.iree_hal_buffer_view_element_type(raw_buffer(buffer)),
    );
}

/// Copy one buffer's contents into host memory.
pub fn buffer_view_to_host(
    device: *DeviceHandle,
    buffer: *BufferHandle,
    destination: []u8,
) Error!void {
    const raw = types.iree_hal_buffer_view_buffer(raw_buffer(buffer)) orelse
        return error.NullBuffer;
    try check(types.zg_iree_hal_buffer_read(
        raw_device(device),
        raw,
        destination.ptr,
        destination.len,
    ));
}

fn check(status: types.Status) Error!void {
    if (types.zg_iree_status_is_ok(status)) return;

    var allocator = types.zg_iree_allocator_system();
    var message: [*c]u8 = null;
    var message_len: types.HostSize = 0;
    if (types.iree_status_to_string(
        status,
        &allocator,
        &message,
        &message_len,
    )) {
        log.err("IREE status: {s}", .{message[0..message_len]});
        types.iree_allocator_free(allocator, message);
    } else {
        log.err("IREE status could not be formatted", .{});
    }
    types.iree_status_free(status);
    return error.IreeError;
}

fn string_view(value: []const u8) types.StringView {
    return .{ .data = value.ptr, .size = value.len };
}

fn byte_span(value: []const u8) types.ConstByteSpan {
    return .{ .data = value.ptr, .data_length = value.len };
}

fn raw_element_type(element_type: ElementType) types.HalElementType {
    return switch (element_type) {
        .bool => types.HAL_ELEMENT_TYPE_BOOL_8,
        .i8 => types.HAL_ELEMENT_TYPE_SINT_8,
        .u8 => types.HAL_ELEMENT_TYPE_UINT_8,
        .i32 => types.HAL_ELEMENT_TYPE_SINT_32,
        .u32 => types.HAL_ELEMENT_TYPE_UINT_32,
        .i64 => types.HAL_ELEMENT_TYPE_SINT_64,
        .u64 => types.HAL_ELEMENT_TYPE_UINT_64,
        .f16 => types.HAL_ELEMENT_TYPE_FLOAT_16,
        .bf16 => types.HAL_ELEMENT_TYPE_BFLOAT_16,
        .f32 => types.HAL_ELEMENT_TYPE_FLOAT_32,
        .f64 => types.HAL_ELEMENT_TYPE_FLOAT_64,
    };
}

fn element_type_from_raw(
    element_type: types.HalElementType,
) Error!ElementType {
    return switch (element_type) {
        types.HAL_ELEMENT_TYPE_BOOL_8 => .bool,
        types.HAL_ELEMENT_TYPE_SINT_8 => .i8,
        types.HAL_ELEMENT_TYPE_UINT_8 => .u8,
        types.HAL_ELEMENT_TYPE_SINT_32 => .i32,
        types.HAL_ELEMENT_TYPE_UINT_32 => .u32,
        types.HAL_ELEMENT_TYPE_SINT_64 => .i64,
        types.HAL_ELEMENT_TYPE_UINT_64 => .u64,
        types.HAL_ELEMENT_TYPE_FLOAT_16 => .f16,
        types.HAL_ELEMENT_TYPE_BFLOAT_16 => .bf16,
        types.HAL_ELEMENT_TYPE_FLOAT_32 => .f32,
        types.HAL_ELEMENT_TYPE_FLOAT_64 => .f64,
        else => error.UnsupportedElementType,
    };
}

fn raw_instance(instance: *InstanceHandle) *types.Instance {
    return @ptrCast(instance);
}

fn instance_handle(instance: *types.Instance) *InstanceHandle {
    return @ptrCast(instance);
}

fn raw_device(device: *DeviceHandle) *types.HalDevice {
    return @ptrCast(device);
}

fn device_handle(device: *types.HalDevice) *DeviceHandle {
    return @ptrCast(device);
}

fn raw_session(session: *SessionHandle) *types.Session {
    return @ptrCast(session);
}

fn session_handle(session: *types.Session) *SessionHandle {
    return @ptrCast(session);
}

fn function_state(function: *FunctionHandle) *FunctionState {
    return @ptrCast(@alignCast(function));
}

fn call_state(call: *CallHandle) *CallState {
    return @ptrCast(@alignCast(call));
}

fn raw_buffer(buffer: *BufferHandle) *types.HalBufferView {
    return @ptrCast(buffer);
}

fn buffer_handle(buffer: *types.HalBufferView) *BufferHandle {
    return @ptrCast(buffer);
}

test "ElementType reports storage widths" {
    try std.testing.expectEqual(@as(usize, 1), ElementType.i8.byte_width());
    try std.testing.expectEqual(@as(usize, 2), ElementType.f16.byte_width());
    try std.testing.expectEqual(@as(usize, 4), ElementType.f32.byte_width());
    try std.testing.expectEqual(@as(usize, 8), ElementType.i64.byte_width());
}

test {
    _ = @import("abi_test.zig");
}
