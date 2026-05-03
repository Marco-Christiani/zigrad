//! Layout-drift defense for the hand-written IREE bindings in `types.zig`.
//!
//! Each test compares Zig `@sizeOf`/`@alignOf`/`@offsetOf` against probes
//!  defined in `shim.c` that read the upstream SDK headers via `sizeof`/
//!  `_Alignof`/`offsetof`. A mismatch fails the build, surfacing IREE ABI
//!  drift as a Zig test failure rather than a hard-to-diagnose crash.
const std = @import("std");
const testing = std.testing;
const types = @import("types.zig");

extern fn zg_abi_sizeof_iree_string_view_t() usize;
extern fn zg_abi_alignof_iree_string_view_t() usize;
extern fn zg_abi_offsetof_iree_string_view_t_data() usize;
extern fn zg_abi_offsetof_iree_string_view_t_size() usize;

extern fn zg_abi_sizeof_iree_const_byte_span_t() usize;
extern fn zg_abi_alignof_iree_const_byte_span_t() usize;
extern fn zg_abi_offsetof_iree_const_byte_span_t_data() usize;
extern fn zg_abi_offsetof_iree_const_byte_span_t_data_length() usize;

extern fn zg_abi_sizeof_iree_allocator_t() usize;
extern fn zg_abi_alignof_iree_allocator_t() usize;
extern fn zg_abi_offsetof_iree_allocator_t_self() usize;
extern fn zg_abi_offsetof_iree_allocator_t_ctl() usize;

extern fn zg_abi_sizeof_iree_vm_function_t() usize;
extern fn zg_abi_alignof_iree_vm_function_t() usize;
extern fn zg_abi_offsetof_iree_vm_function_t_module() usize;
extern fn zg_abi_offsetof_iree_vm_function_t_linkage() usize;
extern fn zg_abi_offsetof_iree_vm_function_t_ordinal() usize;

extern fn zg_abi_sizeof_iree_runtime_call_t() usize;
extern fn zg_abi_alignof_iree_runtime_call_t() usize;
extern fn zg_abi_offsetof_iree_runtime_call_t_session() usize;
extern fn zg_abi_offsetof_iree_runtime_call_t_function() usize;
extern fn zg_abi_offsetof_iree_runtime_call_t_inputs() usize;
extern fn zg_abi_offsetof_iree_runtime_call_t_outputs() usize;

test "iree StringView ABI matches upstream" {
    try testing.expectEqual(zg_abi_sizeof_iree_string_view_t(), @sizeOf(types.StringView));
    try testing.expectEqual(zg_abi_alignof_iree_string_view_t(), @alignOf(types.StringView));
    try testing.expectEqual(zg_abi_offsetof_iree_string_view_t_data(), @offsetOf(types.StringView, "data"));
    try testing.expectEqual(zg_abi_offsetof_iree_string_view_t_size(), @offsetOf(types.StringView, "size"));
}

test "iree ConstByteSpan ABI matches upstream" {
    try testing.expectEqual(zg_abi_sizeof_iree_const_byte_span_t(), @sizeOf(types.ConstByteSpan));
    try testing.expectEqual(zg_abi_alignof_iree_const_byte_span_t(), @alignOf(types.ConstByteSpan));
    try testing.expectEqual(zg_abi_offsetof_iree_const_byte_span_t_data(), @offsetOf(types.ConstByteSpan, "data"));
    try testing.expectEqual(zg_abi_offsetof_iree_const_byte_span_t_data_length(), @offsetOf(types.ConstByteSpan, "data_length"));
}

test "iree Allocator ABI matches upstream" {
    try testing.expectEqual(zg_abi_sizeof_iree_allocator_t(), @sizeOf(types.Allocator));
    try testing.expectEqual(zg_abi_alignof_iree_allocator_t(), @alignOf(types.Allocator));
    try testing.expectEqual(zg_abi_offsetof_iree_allocator_t_self(), @offsetOf(types.Allocator, "self"));
    try testing.expectEqual(zg_abi_offsetof_iree_allocator_t_ctl(), @offsetOf(types.Allocator, "ctl"));
}

test "iree VmFunction ABI matches upstream" {
    try testing.expectEqual(zg_abi_sizeof_iree_vm_function_t(), @sizeOf(types.VmFunction));
    try testing.expectEqual(zg_abi_alignof_iree_vm_function_t(), @alignOf(types.VmFunction));
    try testing.expectEqual(zg_abi_offsetof_iree_vm_function_t_module(), @offsetOf(types.VmFunction, "module"));
    try testing.expectEqual(zg_abi_offsetof_iree_vm_function_t_linkage(), @offsetOf(types.VmFunction, "linkage"));
    try testing.expectEqual(zg_abi_offsetof_iree_vm_function_t_ordinal(), @offsetOf(types.VmFunction, "ordinal"));
}

test "iree RuntimeCall ABI matches upstream" {
    try testing.expectEqual(zg_abi_sizeof_iree_runtime_call_t(), @sizeOf(types.RuntimeCall));
    try testing.expectEqual(zg_abi_alignof_iree_runtime_call_t(), @alignOf(types.RuntimeCall));
    try testing.expectEqual(zg_abi_offsetof_iree_runtime_call_t_session(), @offsetOf(types.RuntimeCall, "session"));
    try testing.expectEqual(zg_abi_offsetof_iree_runtime_call_t_function(), @offsetOf(types.RuntimeCall, "function"));
    try testing.expectEqual(zg_abi_offsetof_iree_runtime_call_t_inputs(), @offsetOf(types.RuntimeCall, "inputs"));
    try testing.expectEqual(zg_abi_offsetof_iree_runtime_call_t_outputs(), @offsetOf(types.RuntimeCall, "outputs"));
}
