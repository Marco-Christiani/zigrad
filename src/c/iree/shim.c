// IREE C API shim -- wraps static inline functions and macros that Zig's
// @cImport cannot translate.  Compiled as a .c object alongside the Zig code.
//
// Must be compiled with -DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl
// (matching the IREE cmake default) so that iree_allocator_system() is defined.

#include "iree/runtime/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"
#include "iree/vm/ref.h"
#include "iree/base/api.h"

// IREE's installed buffer_view.h does not expand the VM type adapter macros,
// so we declare them here to get iree_hal_buffer_view_deref (static inline)
// and iree_hal_buffer_view_retain_ref (extern, defined in the runtime archive).
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_hal_buffer_view,
                              iree_hal_buffer_view_t);

// ---------------------------------------------------------------------------
// Allocator helpers (static inline in allocator.h).
// ---------------------------------------------------------------------------

iree_allocator_t zg_iree_allocator_system(void) {
  return iree_allocator_system();
}

iree_allocator_t zg_iree_allocator_null(void) {
  return iree_allocator_null();
}

// ---------------------------------------------------------------------------
// Status helpers (macros in status.h).
// ---------------------------------------------------------------------------

bool zg_iree_status_is_ok(iree_status_t status) {
  return iree_status_is_ok(status);
}

// ---------------------------------------------------------------------------
// HAL helpers (macros / generated inlines).
// ---------------------------------------------------------------------------

iree_host_size_t zg_iree_hal_element_bit_count(
    iree_hal_element_type_t element_type) {
  return iree_hal_element_bit_count(element_type);
}

// iree_hal_buffer_view_deref: inline VM ref cast.
iree_hal_buffer_view_t* zg_iree_hal_buffer_view_deref(iree_vm_ref_t ref) {
  return iree_hal_buffer_view_deref(ref);
}

// iree_hal_buffer_view_retain_ref: inline VM ref retain + wrap.
iree_vm_ref_t zg_iree_hal_buffer_view_retain_ref(
    iree_hal_buffer_view_t* view) {
  return iree_hal_buffer_view_retain_ref(view);
}

// ---------------------------------------------------------------------------
// Buffer mapping helpers.
//
// iree_hal_buffer_mapping_t contains bitfields which @cImport translates to
// an opaque type.  These wrappers handle the mapping/unmapping in C.
// ---------------------------------------------------------------------------

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
