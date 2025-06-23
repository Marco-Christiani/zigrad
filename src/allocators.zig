pub const DeviceData = @import("allocators/device_data.zig").DeviceData;
pub const CachingAllocator = @import("allocators/caching_allocator.zig").CachingAllocator;
pub const ArenaUnmanaged = @import("allocators/arena_unmanaged.zig");
pub const constants = @import("allocators/constants.zig");
pub const Error = @import("allocators/device_data.zig").Error;

const std = @import("std");

pub fn round_to_next_page(size: usize, page_size: usize) usize {
    std.debug.assert(page_size > 0);
    return size + ((page_size - (size % page_size)) % page_size);
}
