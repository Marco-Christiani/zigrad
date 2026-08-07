//! IREE C API hand-written Zig bindings.
//!
//! The upstream IREE headers contain several constructs that translate-c
//!  rejects on Zig 0.16. Bitfield structs
//!  (`iree_vm_type_def_t`, `iree_hal_buffer_ref_t`) become `opaque {}` and
//!  poison every transitively-embedded type, and `iree_alignas(sizeof(long
//!  double))` cannot be evaluated at translation time. Hand-declaring keeps
//!  the C ABI surface narrow and auditable.
//!
//! ## ABI surface
//!
//! Opaque pointer-only types: callers hold `*T`, never destructure or take
//!  `@sizeOf`. Layout drift in their fields cannot reach Zig.
//!
//! Value structs (`Allocator`, `RuntimeCall`, `VmFunction`, `StringView`,
//!  `ConstByteSpan`) mirror upstream layout. The `abi_test.zig` build step
//!  asserts size, alignment, and field offsets against C `sizeof`/`_Alignof`/
//!  `offsetof` shims, so layout drift fails the build.
//!
//! Allocation flag constants and `iree_hal_buffer_params_t` are not declared
//!  here. The `zg_iree_buffer_view_allocate_device_local_copy` shim
//!  (`shim.c`) applies Zigrad's fixed device-local defaults.
//!
//! TODO(iree): Expose a small Zigrad options struct on the shim if call sites
//!  need different memory, usage, access, or queue combinations.
const std = @import("std");

/// IREE's configured headers define `iree_host_size_t` as `size_t`.
pub const HostSize = usize;

/// IREE's configured headers define `iree_device_size_t` as `size_t`.
pub const DeviceSize = usize;

/// `iree_hal_dim_t` is `iree_device_size_t` upstream.
pub const HalDim = DeviceSize;

/// `iree_hal_element_type_t` is `uint32_t`.
///
/// The bit pattern packs a numerical-type tag in the high byte and the bit
///  count in the low bits (`(numerical << 24) | bits`). See
///  `element_type_value`.
pub const HalElementType = u32;

/// `iree_hal_queue_affinity_t` is `uint64_t`.
pub const HalQueueAffinity = u64;

/// `iree_runtime_call_flags_t` is `uint32_t`.
pub const RuntimeCallFlags = u32;

/// `iree_status_t` is a pointer typedef (`struct iree_status_handle_t*`).
///
/// Status-OK is represented by `null` upstream. A non-null value carries a
///  reference-counted handle. Use the `zg_iree_status_is_ok` shim to
///  preserve macro semantics.
pub const Status = ?*anyopaque;

// Opaque handles. We only hold `*T` and pass to/from the C API.

pub const Instance = opaque {};
pub const Session = opaque {};
pub const HalDevice = opaque {};
pub const HalAllocator = opaque {};
pub const HalBuffer = opaque {};
pub const HalBufferView = opaque {};

// Value structs that must match upstream layout exactly.

/// `iree_string_view_t` upstream:
/// ```c
/// typedef struct iree_string_view_t {
///   const char* data;
///   iree_host_size_t size;
/// } iree_string_view_t;
/// ```
pub const StringView = extern struct {
    data: [*]const u8,
    size: HostSize,
};

/// `iree_const_byte_span_t` upstream:
/// ```c
/// typedef struct iree_const_byte_span_t {
///   const uint8_t* data;
///   iree_host_size_t data_length;
/// } iree_const_byte_span_t;
/// ```
pub const ConstByteSpan = extern struct {
    data: [*]const u8,
    data_length: HostSize,
};

/// `iree_allocator_t` upstream:
/// ```c
/// typedef struct iree_allocator_t {
///   void* self;
///   iree_status_t (*ctl)(...);
/// } iree_allocator_t;
/// ```
/// The control function pointer's signature does not matter for our binding
///  (we never call it directly from Zig).
pub const Allocator = extern struct {
    self: ?*anyopaque,
    ctl: ?*const anyopaque,
};

/// `iree_vm_function_t` upstream:
/// ```c
/// typedef struct iree_vm_function_t {
///   iree_vm_module_t* module;
///   uint16_t linkage;
///   uint16_t ordinal;
/// } iree_vm_function_t;
/// ```
pub const VmFunction = extern struct {
    module: ?*anyopaque,
    linkage: u16,
    ordinal: u16,
};

/// `iree_runtime_call_t` upstream:
/// ```c
/// typedef struct iree_runtime_call_t {
///   iree_runtime_session_t* session;
///   iree_vm_function_t function;
///   iree_vm_list_t* inputs;
///   iree_vm_list_t* outputs;
/// } iree_runtime_call_t;
/// ```
/// `iree_vm_list_t*` is opaque to Zigrad.
///
/// Calls use `iree_runtime_call_*_buffer_view` helpers instead of accessing
///  the list directly.
pub const RuntimeCall = extern struct {
    session: ?*Session,
    function: VmFunction,
    inputs: ?*anyopaque,
    outputs: ?*anyopaque,
};

// Element type encoding.

/// Compose an `iree_hal_element_type_t` value from the upstream encoding:
///  `(numerical_type << 24) | bits`. Mirrors `IREE_HAL_ELEMENT_TYPE_VALUE`.
pub fn element_type_value(numerical_type: u32, bits: u32) HalElementType {
    return (numerical_type << 24) | bits;
}

/// Subset of `iree_hal_numerical_type_bits_t` (upstream `buffer_view.h`).
///  Only the values zigrad ever names are listed, extend as needed.
const NumericalType = struct {
    const integer_signed: u32 = 0x10 | 0x01;
    const integer_unsigned: u32 = 0x10 | 0x02;
    const boolean: u32 = 0x10 | 0x03;
    const float_ieee: u32 = 0x20 | 0x01;
    const float_brain: u32 = 0x20 | 0x02;
};

pub const HAL_ELEMENT_TYPE_BOOL_8 = element_type_value(NumericalType.boolean, 8);
pub const HAL_ELEMENT_TYPE_SINT_8 = element_type_value(NumericalType.integer_signed, 8);
pub const HAL_ELEMENT_TYPE_UINT_8 = element_type_value(NumericalType.integer_unsigned, 8);
pub const HAL_ELEMENT_TYPE_SINT_32 = element_type_value(NumericalType.integer_signed, 32);
pub const HAL_ELEMENT_TYPE_UINT_32 = element_type_value(NumericalType.integer_unsigned, 32);
pub const HAL_ELEMENT_TYPE_SINT_64 = element_type_value(NumericalType.integer_signed, 64);
pub const HAL_ELEMENT_TYPE_UINT_64 = element_type_value(NumericalType.integer_unsigned, 64);
pub const HAL_ELEMENT_TYPE_FLOAT_16 = element_type_value(NumericalType.float_ieee, 16);
pub const HAL_ELEMENT_TYPE_FLOAT_32 = element_type_value(NumericalType.float_ieee, 32);
pub const HAL_ELEMENT_TYPE_FLOAT_64 = element_type_value(NumericalType.float_ieee, 64);
pub const HAL_ELEMENT_TYPE_BFLOAT_16 = element_type_value(NumericalType.float_brain, 16);

// Direct IREE entry points (exported symbols in libiree_runtime_unified.a).

pub extern "c" fn iree_runtime_instance_release(instance: *Instance) void;
pub extern "c" fn iree_hal_device_release(device: *HalDevice) void;
pub extern "c" fn iree_hal_device_allocator(device: *HalDevice) ?*HalAllocator;

pub extern "c" fn iree_runtime_session_release(session: *Session) void;
pub extern "c" fn iree_runtime_session_append_bytecode_module_from_memory(
    session: *Session,
    flatbuffer_data: ConstByteSpan,
    flatbuffer_allocator: Allocator,
) Status;
pub extern "c" fn iree_runtime_session_lookup_function(
    session: *Session,
    full_name: StringView,
    out_function: *VmFunction,
) Status;

pub extern "c" fn iree_runtime_call_initialize(
    session: *Session,
    function: VmFunction,
    out_call: *RuntimeCall,
) Status;
pub extern "c" fn iree_runtime_call_deinitialize(call: *RuntimeCall) void;
pub extern "c" fn iree_runtime_call_invoke(call: *RuntimeCall, flags: RuntimeCallFlags) Status;
pub extern "c" fn iree_runtime_call_inputs_push_back_buffer_view(
    call: *RuntimeCall,
    buffer_view: *HalBufferView,
) Status;
pub extern "c" fn iree_runtime_call_outputs_pop_front_buffer_view(
    call: *RuntimeCall,
    out_buffer_view: *?*HalBufferView,
) Status;

pub extern "c" fn iree_hal_buffer_view_retain(view: *HalBufferView) void;
pub extern "c" fn iree_hal_buffer_view_release(view: *HalBufferView) void;
pub extern "c" fn iree_hal_buffer_view_element_count(view: *HalBufferView) HostSize;
pub extern "c" fn iree_hal_buffer_view_element_type(view: *HalBufferView) HalElementType;
pub extern "c" fn iree_hal_buffer_view_buffer(view: *HalBufferView) ?*HalBuffer;

pub extern "c" fn iree_status_to_string(
    status: Status,
    allocator: *Allocator,
    out_buffer: *[*c]u8,
    out_buffer_length: *HostSize,
) bool;
pub extern "c" fn iree_status_free(status: Status) void;
pub extern "c" fn iree_allocator_free(allocator: Allocator, ptr: ?*anyopaque) void;

// Zigrad shim helpers (defined in shim.c). Each replaces an upstream pattern
//  that translate-c cannot handle (static inline, macro-expanded helper, or
//  bitfield struct), or hides a multi-step C ritual that would force us to
//  mirror more types in Zig (options structs, buffer params).

pub extern "c" fn zg_iree_allocator_system() Allocator;
pub extern "c" fn zg_iree_allocator_null() Allocator;
pub extern "c" fn zg_iree_status_is_ok(status: Status) bool;
pub extern "c" fn zg_iree_status_code(status: Status) u32;
pub extern "c" fn zg_iree_hal_element_bit_count(element_type: HalElementType) HostSize;

/// Subset of `iree_status_code_t` that callers want to recognize without
///  routing through the logging `check()` path. Values match the upstream
///  enum (see `iree/base/status.h`).
pub const STATUS_OK: u32 = 0;
pub const STATUS_NOT_FOUND: u32 = 5;
pub const STATUS_OUT_OF_RANGE: u32 = 11;

/// Wrap `iree_runtime_instance_options_initialize` +
///  `_use_all_available_drivers` + `_create`. Hides the options struct (whose
///  size we would otherwise need to mirror) entirely behind C.
pub extern "c" fn zg_iree_runtime_instance_create_all_drivers(
    out_instance: *?*Instance,
) Status;

/// Wrap `iree_runtime_session_options_initialize` + `_create_with_device`.
pub extern "c" fn zg_iree_runtime_session_create_with_device_default(
    instance: *Instance,
    device: *HalDevice,
    out_session: *?*Session,
) Status;

/// Wrap `iree_runtime_instance_try_create_default_device`. Pure passthrough,
///  exposed via shim so the Zig binding never references options structs and
///  the entire instance/session/device lifecycle goes through one C surface.
pub extern "c" fn zg_iree_runtime_instance_try_create_default_device(
    instance: *Instance,
    driver_name: StringView,
    out_device: *?*HalDevice,
) Status;

/// Allocate a device-local buffer view by copying host bytes. Hides
///  `iree_hal_buffer_params_t` and the four memory/usage/access/queue flag
///  constants. The shim applies Zigrad's fixed device-local defaults.
pub extern "c" fn zg_iree_buffer_view_allocate_device_local_copy(
    device: *HalDevice,
    shape: [*]const HalDim,
    shape_rank: HostSize,
    element_type: HalElementType,
    src: [*]const u8,
    src_len: HostSize,
    out_view: *?*HalBufferView,
) Status;

/// Map the buffer for read, copy `dst_len` bytes (or less if the buffer is
///  smaller), then unmap. Hides `iree_hal_buffer_mapping_t` whose bitfields
///  defeat translate-c.
pub extern "c" fn zg_iree_hal_buffer_read(
    buffer: *HalBuffer,
    dst: [*]u8,
    dst_len: HostSize,
) Status;

/// Symmetric to `zg_iree_hal_buffer_read`.
pub extern "c" fn zg_iree_hal_buffer_write(
    buffer: *HalBuffer,
    src: [*]const u8,
    src_len: HostSize,
) Status;
