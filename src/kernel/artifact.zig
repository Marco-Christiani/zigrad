//! Portable kernel artifacts.

const std = @import("std");

/// Portable artifact produced by a kernel provider.
///
/// The artifact contains no runtime pointers. Its provider resolves any
///  process-local state before execution or when dispatch begins.
pub const Artifact = struct {
    /// Opaque data allocated with the provider compilation allocator.
    data: []const u8,

    /// Workspace bytes required at dispatch time.
    workspace_bytes: usize = 0,

    /// Byte alignment required for workspace allocation.
    ///
    /// This value must be a nonzero power of two when `workspace_bytes` is
    ///  nonzero.
    workspace_alignment: usize = 1,

    /// Release artifact data with the provider compilation allocator.
    pub fn deinit(self: *Artifact, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        self.* = undefined;
    }
};
