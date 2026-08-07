//! Memory mapping helpers.
const std = @import("std");

/// POSIX memory-map `path` read-only (SHARED).
///
/// Returns a page-aligned read-only slice backed by the kernel page cache.
///
/// The caller releases the slice with `std.posix.munmap`.
///
/// Uses POSIX `open`, `lseek`, and `mmap` because no I/O-threaded mmap API is
///  available. The descriptor closes after mapping. Relative paths resolve
///  against the current working directory.
pub fn mmap_file(
    /// Absolute or relative path
    path: []const u8,
) ![]align(std.heap.page_size_min) u8 {
    const fd = try std.posix.openat(std.posix.AT.FDCWD, path, .{ .ACCMODE = .RDONLY }, 0);
    defer _ = std.posix.system.close(fd);

    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end < 0) return error.FileSeekError;
    _ = std.c.lseek(fd, 0, std.c.SEEK.SET);
    const size: usize = @intCast(end);

    return std.posix.mmap(
        null,
        size,
        .{ .READ = true },
        .{ .TYPE = .SHARED },
        fd,
        0,
    );
}

pub const munmap = std.posix.munmap;
