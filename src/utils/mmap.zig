//! Memory mapping helpers.
const std = @import("std");

/// POSIX memory-map `path` read-only (SHARED).
///
/// Returns a page-aligned read-only slice backed by the kernel page cache owned by the caller
///  who must release it via `std.posix.munmap`.
pub fn mmap_file(
    /// Absolute or relative path
    path: []const u8,
) ![]align(std.heap.page_size_min) u8 {
    var file = if (std.fs.path.isAbsolute(path))
        try std.fs.openFileAbsolute(path, .{})
    else
        try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const stat = try file.stat();
    const size: usize = @intCast(stat.size);

    return std.posix.mmap(
        null,
        size,
        std.posix.PROT.READ,
        .{ .TYPE = .SHARED },
        file.handle,
        0,
    );
}

pub const munmap = std.posix.munmap;
