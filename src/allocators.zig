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

pub fn round_to_prev_page(size: usize, page_size: usize) usize {
    std.debug.assert(page_size > 0);
    return size - (page_size - (size % page_size));
}

pub fn adjust_map_size(size: ?usize, page_size: usize, total_mem: usize) usize {
    const req_size = size orelse total_mem;
    const resized = round_to_next_page(req_size, page_size);

    return if (resized > total_mem)
        round_to_prev_page(total_mem, page_size)
    else
        resized;
}
