// Note that scratch memory returns raw slices. This should be a
// signal to the user that scratch memory shouldn't be used
// in the same contexts as tensor data.
const ScratchAllocator = @This();

ptr: [*]u8 = undefined,
total: usize = 0,
/// Scratch memory does not have to be freed after calling this
/// this function Instead, scratch is freed upon calling deinit.
/// Also, using anytype because this is a sub-allocator.
pub fn alloc(self: *ScratchAllocator, data_handler: anytype, T: type, n: usize) []T {
    if (n == 0) return &.{};

    const total: usize = @sizeOf(T) * n;
    // check if we have enough scratch to provide a payload
    if (self.total < total) {
        if (self.total != 0)
            data_handler.free(self.ptr[0..self.total]);

        // Hard error - we cannot fail to allocate scratch memory.
        // After warmup, you'll likely have sufficient scratch.
        self.ptr = data_handler.alloc(total) orelse
            @panic("Cannot allocate scratch memory.");

        self.total = total;
    }
    const tptr: [*]T = @ptrCast(@alignCast(self.ptr));
    return tptr[0..n];
}

pub fn deinit(self: *ScratchAllocator, data_handler: anytype) void {
    self.clear(data_handler);
    self.* = undefined;
}

fn clear(self: *ScratchAllocator, data_handler: anytype) void {
    if (self.total != 0)
        data_handler.free(self.ptr[0..self.total]);

    self.total = 0;
}
