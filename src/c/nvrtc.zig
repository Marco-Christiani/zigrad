//! NVRTC C API FFI bindings.
//!
//! This module provides bindings to NVIDIA's NVRTC API which is used by TVM for runtime CUDA kernel compilation.

const build_options = @import("build_options");

const tvm_enabled = build_options.enable_tvm;

// When TVM is enabled, import the NVRTC API
// When disabled, provide stub types
const Impl = if (tvm_enabled)
    @cImport({
        @cInclude("nvrtc.h");
    })
else
    struct {
        // Stub types for when TVM/NVRTC is disabled
        pub const nvrtcProgram = ?*anyopaque;
        pub const nvrtcResult = c_int;
        pub const NVRTC_SUCCESS: c_int = 0;
    };

// Re-export all types and functions
pub const nvrtcProgram = Impl.nvrtcProgram;
pub const nvrtcResult = Impl.nvrtcResult;
pub const NVRTC_SUCCESS = Impl.NVRTC_SUCCESS;

pub const nvrtcCreateProgram = if (tvm_enabled) Impl.nvrtcCreateProgram else @compileError("TVM not enabled");
pub const nvrtcDestroyProgram = if (tvm_enabled) Impl.nvrtcDestroyProgram else @compileError("TVM not enabled");
pub const nvrtcCompileProgram = if (tvm_enabled) Impl.nvrtcCompileProgram else @compileError("TVM not enabled");
pub const nvrtcGetPTXSize = if (tvm_enabled) Impl.nvrtcGetPTXSize else @compileError("TVM not enabled");
pub const nvrtcGetPTX = if (tvm_enabled) Impl.nvrtcGetPTX else @compileError("TVM not enabled");
pub const nvrtcGetProgramLogSize = if (tvm_enabled) Impl.nvrtcGetProgramLogSize else @compileError("TVM not enabled");
pub const nvrtcGetProgramLog = if (tvm_enabled) Impl.nvrtcGetProgramLog else @compileError("TVM not enabled");
