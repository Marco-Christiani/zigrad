//! Common types and enums for TVM integration.
//!
//! This module defines shared types used across the TVM subsystem.

/// Target kind for TVM compilation.
pub const TargetKind = enum {
    cpu,
    cuda,
};
