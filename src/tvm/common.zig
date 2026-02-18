//! Common types and enums for TVM integration.
//!
//! Re-exports from the FFI layer for convenience within src/tvm/.

pub const TargetKind = @import("../ffi/tvm/types.zig").TargetKind;
