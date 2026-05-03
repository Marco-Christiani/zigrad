// iree_zig.h -- Wrapper header for @cImport compatibility.
//
// Zig's C translator cannot handle C11 _Atomic qualifiers used in
// iree/base/internal/atomics_clang.h.  This header forces the GCC atomics
// path (plain int32_t typedefs) by overriding the compiler detection after
// target_platform.h has been included.
//
// The GCC atomics path uses __sync_* builtins instead of _Atomic.
// The typedef differences (int32_t vs _Atomic int32_t) are ABI-compatible
// and irrelevant for Zig since we never perform atomic ops from Zig code.

#include "iree/base/target_platform.h"

// target_platform.h sets IREE_COMPILER_CLANG when __clang__ is defined.
// Override to select atomics_gcc.h instead of atomics_clang.h.
#ifdef IREE_COMPILER_CLANG
#undef IREE_COMPILER_CLANG
#endif
#ifndef IREE_COMPILER_GCC
#define IREE_COMPILER_GCC 1
#endif

// Zig 0.16 translate-c cannot evaluate `sizeof(long double)` at translation
// time, so it rejects `iree_alignas(iree_max_align_t)` in loop_inline.h with
// "requested alignment is not a power of 2". Include alignment.h first to
// let it set up its other macros, then override `iree_max_align_t` to the
// known x86_64-Linux long-double size (16) before loop_inline.h consumes it.
#include "iree/base/alignment.h"
#undef iree_max_align_t
#define iree_max_align_t 16

#include "iree/runtime/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"
#include "iree/base/api.h"
