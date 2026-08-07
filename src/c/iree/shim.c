// Bridges IREE C constructs that Zig's translate-c cannot represent.
//
// This object is compiled with the Zig bindings in `types.zig`. Defining
//  IREE_ALLOCATOR_SYSTEM_CTL as iree_allocator_libc_ctl provides
//  iree_allocator_system().

#include <stddef.h>

#include "iree/runtime/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"
#include "iree/base/api.h"

// Allocator macro wrappers.

iree_allocator_t zg_iree_allocator_system(void) {
  return iree_allocator_system();
}

iree_allocator_t zg_iree_allocator_null(void) {
  return iree_allocator_null();
}

// Status macro wrappers.

bool zg_iree_status_is_ok(iree_status_t status) {
  return iree_status_is_ok(status);
}

// Returns the status code without consuming `status`.
//
// Zig callers use this to recognize sentinels such as OUT_OF_RANGE.
uint32_t zg_iree_status_code(iree_status_t status) {
  return (uint32_t)iree_status_code(status);
}

// HAL inline wrappers.

iree_host_size_t zg_iree_hal_element_bit_count(
    iree_hal_element_type_t element_type) {
  return iree_hal_element_bit_count(element_type);
}

// Maps buffers whose mapping descriptor is opaque to translate-c.

iree_status_t zg_iree_hal_buffer_read(iree_hal_buffer_t* buffer,
                                       uint8_t* dst,
                                       iree_host_size_t dst_len) {
  iree_hal_buffer_mapping_t mapping;
  iree_status_t status = iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      0, IREE_HAL_WHOLE_BUFFER, &mapping);
  if (!iree_status_is_ok(status)) return status;
  iree_host_size_t len = mapping.contents.data_length < dst_len
                             ? mapping.contents.data_length
                             : dst_len;
  memcpy(dst, mapping.contents.data, len);
  return iree_hal_buffer_unmap_range(&mapping);
}

iree_status_t zg_iree_hal_buffer_write(iree_hal_buffer_t* buffer,
                                        const uint8_t* src,
                                        iree_host_size_t src_len) {
  iree_hal_buffer_mapping_t mapping;
  iree_status_t status = iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_WRITE,
      0, IREE_HAL_WHOLE_BUFFER, &mapping);
  if (!iree_status_is_ok(status)) return status;
  iree_host_size_t len = mapping.contents.data_length < src_len
                             ? mapping.contents.data_length
                             : src_len;
  memcpy(mapping.contents.data, src, len);
  return iree_hal_buffer_unmap_range(&mapping);
}

// Creates IREE runtime objects without exposing options struct layouts to Zig.

iree_status_t zg_iree_runtime_instance_create_all_drivers(
    iree_runtime_instance_t** out_instance) {
  iree_runtime_instance_options_t options;
  iree_runtime_instance_options_initialize(&options);
  iree_runtime_instance_options_use_all_available_drivers(&options);
  return iree_runtime_instance_create(&options, iree_allocator_system(),
                                       out_instance);
}

iree_status_t zg_iree_runtime_session_create_with_device_default(
    iree_runtime_instance_t* instance, iree_hal_device_t* device,
    iree_runtime_session_t** out_session) {
  iree_runtime_session_options_t options;
  iree_runtime_session_options_initialize(&options);
  return iree_runtime_session_create_with_device(
      instance, &options, device, iree_allocator_system(), out_session);
}

iree_status_t zg_iree_runtime_instance_try_create_default_device(
    iree_runtime_instance_t* instance, iree_string_view_t driver_name,
    iree_hal_device_t** out_device) {
  return iree_runtime_instance_try_create_default_device(instance, driver_name,
                                                          out_device);
}

// Allocates a dense device-local buffer view using Zigrad's runtime defaults.

iree_status_t zg_iree_buffer_view_allocate_device_local_copy(
    iree_hal_device_t* device, const iree_hal_dim_t* shape,
    iree_host_size_t shape_rank, iree_hal_element_type_t element_type,
    const uint8_t* src, iree_host_size_t src_len,
    iree_hal_buffer_view_t** out_view) {
  iree_hal_buffer_params_t params = {
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .min_alignment = 0,
  };
  iree_const_byte_span_t bytes = {.data = src, .data_length = src_len};
  return iree_hal_buffer_view_allocate_buffer_copy(
      device, iree_hal_device_allocator(device), shape_rank, shape,
      element_type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, params, bytes,
      out_view);
}

// Reports ABI layouts from the IREE headers used for linkage.
//
// `src/c/iree/abi_test.zig` compares these values with the Zig declarations.

iree_host_size_t zg_abi_sizeof_iree_string_view_t(void) {
  return sizeof(iree_string_view_t);
}
iree_host_size_t zg_abi_alignof_iree_string_view_t(void) {
  return _Alignof(iree_string_view_t);
}
iree_host_size_t zg_abi_offsetof_iree_string_view_t_data(void) {
  return offsetof(iree_string_view_t, data);
}
iree_host_size_t zg_abi_offsetof_iree_string_view_t_size(void) {
  return offsetof(iree_string_view_t, size);
}

iree_host_size_t zg_abi_sizeof_iree_const_byte_span_t(void) {
  return sizeof(iree_const_byte_span_t);
}
iree_host_size_t zg_abi_alignof_iree_const_byte_span_t(void) {
  return _Alignof(iree_const_byte_span_t);
}
iree_host_size_t zg_abi_offsetof_iree_const_byte_span_t_data(void) {
  return offsetof(iree_const_byte_span_t, data);
}
iree_host_size_t zg_abi_offsetof_iree_const_byte_span_t_data_length(void) {
  return offsetof(iree_const_byte_span_t, data_length);
}

iree_host_size_t zg_abi_sizeof_iree_allocator_t(void) {
  return sizeof(iree_allocator_t);
}
iree_host_size_t zg_abi_alignof_iree_allocator_t(void) {
  return _Alignof(iree_allocator_t);
}
iree_host_size_t zg_abi_offsetof_iree_allocator_t_self(void) {
  return offsetof(iree_allocator_t, self);
}
iree_host_size_t zg_abi_offsetof_iree_allocator_t_ctl(void) {
  return offsetof(iree_allocator_t, ctl);
}

iree_host_size_t zg_abi_sizeof_iree_vm_function_t(void) {
  return sizeof(iree_vm_function_t);
}
iree_host_size_t zg_abi_alignof_iree_vm_function_t(void) {
  return _Alignof(iree_vm_function_t);
}
iree_host_size_t zg_abi_offsetof_iree_vm_function_t_module(void) {
  return offsetof(iree_vm_function_t, module);
}
iree_host_size_t zg_abi_offsetof_iree_vm_function_t_linkage(void) {
  return offsetof(iree_vm_function_t, linkage);
}
iree_host_size_t zg_abi_offsetof_iree_vm_function_t_ordinal(void) {
  return offsetof(iree_vm_function_t, ordinal);
}

iree_host_size_t zg_abi_sizeof_iree_runtime_call_t(void) {
  return sizeof(iree_runtime_call_t);
}
iree_host_size_t zg_abi_alignof_iree_runtime_call_t(void) {
  return _Alignof(iree_runtime_call_t);
}
iree_host_size_t zg_abi_offsetof_iree_runtime_call_t_session(void) {
  return offsetof(iree_runtime_call_t, session);
}
iree_host_size_t zg_abi_offsetof_iree_runtime_call_t_function(void) {
  return offsetof(iree_runtime_call_t, function);
}
iree_host_size_t zg_abi_offsetof_iree_runtime_call_t_inputs(void) {
  return offsetof(iree_runtime_call_t, inputs);
}
iree_host_size_t zg_abi_offsetof_iree_runtime_call_t_outputs(void) {
  return offsetof(iree_runtime_call_t, outputs);
}
