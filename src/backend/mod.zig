/// Backend Module
///
/// Unified backend abstractions that merge Toolchain + Runtime.
///
/// Currently provides:
///   - pjrt.Backend: PJRT-based backend for XLA compilation and execution
///
/// See: .internal/2026-01-16-03_PASS_BASED_PIPELINE.md
pub const pjrt = @import("pjrt.zig");

pub const PjrtBackend = pjrt.Backend;
