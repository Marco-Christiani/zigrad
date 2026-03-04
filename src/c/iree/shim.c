// IREE C API shim -- wraps static inline functions and macros that Zig's
// @cImport cannot translate.  Compiled as a .c object alongside the Zig code.
//
// Must be compiled with -DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl
// (matching the IREE cmake default) so that iree_allocator_system() is defined.

#include "iree/runtime/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"
#include "iree/base/api.h"

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
