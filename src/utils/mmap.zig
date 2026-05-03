//! Memory mapping helpers.
const std = @import("std");

/// POSIX memory-map `path` read-only (SHARED).
///
/// Returns a page-aligned read-only slice backed by the kernel page cache owned by the caller
///  who must release it via `std.posix.munmap`.
///
/// Uses raw posix for `open`/`lseek`/`mmap` because mmap is itself a posix
///  primitive without an `io`-threaded equivalent. The fd is opened only
///  to be passed to mmap, then closed; relative paths resolve against CWD.
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
