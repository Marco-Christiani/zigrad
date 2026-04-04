//! IREE C API bindings.
//!
//! `compiler`: embedding API for libIREECompiler.so (dlopen at runtime).
//! `runtime`:  HAL + VM + runtime API for libIREERuntime.so (linked at build time).
pub const compiler = @import("compiler.zig");
pub const runtime = @import("runtime.zig");
