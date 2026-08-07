//! Integration-free device identity.

const std = @import("std");

/// Open execution-platform name.
///
/// Known names have constants, and integrations can retain other reported
///  names. The referenced bytes must outlive every value that stores them.
pub const Platform = struct {
    name: []const u8,

    pub const cpu: Platform = .{ .name = "cpu" };
    pub const cuda: Platform = .{ .name = "cuda" };
    pub const rocm: Platform = .{ .name = "rocm" };
    pub const tpu: Platform = .{ .name = "tpu" };

    /// Compare platform names without ASCII case distinctions.
    pub fn eql(self: Platform, other: Platform) bool {
        return std.ascii.eqlIgnoreCase(self.name, other.name);
    }
};

/// One device selected from an execution platform.
pub const Device = struct {
    platform: Platform,
    ordinal: i32 = 0,

    /// Return whether both values select the same platform and ordinal.
    pub fn eql(self: Device, other: Device) bool {
        return self.ordinal == other.ordinal and self.platform.eql(other.platform);
    }
};

test Platform {
    try std.testing.expect((Platform{ .name = "CUDA" }).eql(.cuda));
    try std.testing.expect(!Platform.cuda.eql(.rocm));
}

test Device {
    const first = Device{ .platform = .cuda, .ordinal = 1 };
    try std.testing.expect(first.eql(.{ .platform = .{ .name = "CUDA" }, .ordinal = 1 }));
    try std.testing.expect(!first.eql(.{ .platform = .cuda, .ordinal = 0 }));
}
