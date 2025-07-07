const std = @import("std");
const assert = std.debug.assert;
const mem = std.mem;
const math = std.math;
const Allocator = std.mem.Allocator;
const Alignment = std.mem.Alignment;
const Error = Allocator.Error;
const ArenaUnmanaged = @This();

/// Copied from std.ArenaAllocator and modified. This is now an unmanged version
/// that isn't meant to be used with the allocator interface. Also changing functions
/// to snake case. Not including independent "free" function because we don't have
/// a use-case for that in zigrad. This will typically be used to contain and
/// teardown whole computation graphs. This exists to facility "create" and "reset".
const BufNode = std.SinglyLinkedList(usize).Node;
const BufNode_alignment: mem.Alignment = .fromByteUnits(@alignOf(BufNode));

pub const empty: ArenaUnmanaged = .{
    .buffer_list = .{},
    .end_index = 0,
};

buffer_list: std.SinglyLinkedList(usize),
end_index: usize,

pub fn deinit(self: ArenaUnmanaged, allocator: Allocator) void {
    var it = self.buffer_list.first;
    while (it) |node| {
        // this has to occur before the free because the free frees node
        const next_it = node.next;
        const alloc_buf = @as([*]u8, @ptrCast(node))[0..node.data];
        allocator.rawFree(alloc_buf, BufNode_alignment, @returnAddress());
        it = next_it;
    }
}

pub fn alloc(self: *ArenaUnmanaged, allocator: Allocator, comptime T: type, n: usize) Error![]T {
    if (@sizeOf(T) == 0) @compileError("Allocating zero-size objects not supported.");
    const ptr: [*]T = @ptrCast(try self.alloc_bytes_with_alignment(allocator, @alignOf(T), n * @sizeOf(T)));
    return ptr[0..n];
}

/// Modified from standard library arena allocator
pub fn free(self: *ArenaUnmanaged, slice: anytype) void {
    const cur_node = self.buffer_list.first orelse return;
    const cur_buf = @as([*]u8, @ptrCast(cur_node))[@sizeOf(BufNode)..cur_node.data];
    const buf = std.mem.sliceAsBytes(slice);

    if (@intFromPtr(cur_buf.ptr) + self.end_index == @intFromPtr(buf.ptr) + buf.len) {
        self.end_index -= buf.len;
    }
}

pub fn dupe(self: *ArenaUnmanaged, allocator: Allocator, comptime T: type, slice: []const T) Error![]T {
    const out = try self.alloc(allocator, T, slice.len);
    @memcpy(out, slice);
    return out;
}
/// Copied from standard library allocator interface
pub fn create(self: *ArenaUnmanaged, allocator: Allocator, comptime T: type) Error!*T {
    if (@sizeOf(T) == 0) return @as(*T, @ptrFromInt(math.maxInt(usize)));
    const ptr: *T = @ptrCast(try self.alloc_bytes_with_alignment(allocator, @alignOf(T), @sizeOf(T)));
    return ptr;
}

/// Modified from standard library arena allocator
pub fn destroy(self: *ArenaUnmanaged, ptr: anytype) void {
    const cur_node = self.buffer_list.first orelse return;
    const cur_buf = @as([*]u8, @ptrCast(cur_node))[@sizeOf(BufNode)..cur_node.data];
    const buf = std.mem.asBytes(ptr);

    if (@intFromPtr(cur_buf.ptr) + self.end_index == @intFromPtr(buf.ptr) + buf.len) {
        self.end_index -= buf.len;
    }
}

/// Queries the current memory use of this arena.
/// This will **not** include the storage required for internal keeping.
pub fn query_capacity(self: ArenaUnmanaged) usize {
    var size: usize = 0;
    var it = self.buffer_list.first;
    while (it) |node| : (it = node.next) {
        // Compute the actually allocated size excluding the
        // linked list node.
        size += node.data - @sizeOf(BufNode);
    }
    return size;
}

pub const Mode = union(enum) {
    retain_capacity: void,
    retain_with_limit: usize,
    free_all: void,
};

pub fn reset(self: *ArenaUnmanaged, allocator: Allocator, mode: Mode) bool {
    const requested_capacity = switch (mode) {
        .retain_capacity => self.query_capacity(),
        .retain_with_limit => |limit| @min(limit, self.query_capacity()),
        .free_all => 0,
    };
    if (requested_capacity == 0) {
        self.deinit(allocator);
        self.* = .empty;
        return true;
    }
    const total_size = requested_capacity + @sizeOf(BufNode);
    // Free all nodes except for the last one
    var it = self.buffer_list.first;
    const maybe_first_node = while (it) |node| {
        // this has to occur before the free because the free frees node
        const next_it = node.next;
        if (next_it == null)
            break node;
        const alloc_buf = @as([*]u8, @ptrCast(node))[0..node.data];
        allocator.rawFree(alloc_buf, BufNode_alignment, @returnAddress());
        it = next_it;
    } else null;
    std.debug.assert(maybe_first_node == null or maybe_first_node.?.next == null);
    // reset the state before we try resizing the buffers, so we definitely have reset the arena to 0.
    self.end_index = 0;
    if (maybe_first_node) |first_node| {
        self.buffer_list.first = first_node;
        // perfect, no need to invoke the child_allocator
        if (first_node.data == total_size)
            return true;
        const first_alloc_buf = @as([*]u8, @ptrCast(first_node))[0..first_node.data];
        if (allocator.rawResize(first_alloc_buf, BufNode_alignment, total_size, @returnAddress())) {
            // successful resize
            first_node.data = total_size;
        } else {
            // manual realloc
            const new_ptr = allocator.rawAlloc(total_size, BufNode_alignment, @returnAddress()) orelse {
                // we failed to preheat the arena properly, signal this to the user.
                return false;
            };
            allocator.rawFree(first_alloc_buf, BufNode_alignment, @returnAddress());
            const node: *BufNode = @ptrCast(@alignCast(new_ptr));
            node.* = .{ .data = total_size };
            self.buffer_list.first = node;
        }
    }
    return true;
}

fn create_node(self: *ArenaUnmanaged, allocator: Allocator, prev_len: usize, minimum_size: usize) ?*BufNode {
    const actual_min_size = minimum_size + (@sizeOf(BufNode) + 16);
    const big_enough_len = prev_len + actual_min_size;
    const len = big_enough_len + big_enough_len / 2;
    const ptr = allocator.rawAlloc(len, BufNode_alignment, @returnAddress()) orelse
        return null;
    const buf_node: *BufNode = @ptrCast(@alignCast(ptr));
    buf_node.* = .{ .data = len };
    self.buffer_list.prepend(buf_node);
    self.end_index = 0;
    return buf_node;
}

/// Copied from standard library allocator interface
fn alloc_bytes_with_alignment(
    self: *ArenaUnmanaged,
    allocator: Allocator,
    comptime alignment: u29,
    byte_count: usize,
) Error![*]align(alignment) u8 {
    if (byte_count == 0) {
        const ptr = comptime std.mem.alignBackward(usize, math.maxInt(usize), alignment);
        return @as([*]align(alignment) u8, @ptrFromInt(ptr));
    }
    const byte_ptr = self.raw_alloc(allocator, byte_count, .fromByteUnits(alignment)) orelse return Error.OutOfMemory;
    @memset(byte_ptr[0..byte_count], undefined);
    return @alignCast(byte_ptr);
}

fn raw_alloc(self: *ArenaUnmanaged, allocator: Allocator, n: usize, alignment: mem.Alignment) ?[*]u8 {
    const ptr_align = alignment.toByteUnits();
    var cur_node = if (self.buffer_list.first) |first_node|
        first_node
    else
        (self.create_node(allocator, 0, n + ptr_align) orelse return null);
    while (true) {
        const cur_alloc_buf = @as([*]u8, @ptrCast(cur_node))[0..cur_node.data];
        const cur_buf = cur_alloc_buf[@sizeOf(BufNode)..];
        const addr = @intFromPtr(cur_buf.ptr) + self.end_index;
        const adjusted_addr = mem.alignForward(usize, addr, ptr_align);
        const adjusted_index = self.end_index + (adjusted_addr - addr);
        const new_end_index = adjusted_index + n;

        if (new_end_index <= cur_buf.len) {
            const result = cur_buf[adjusted_index..new_end_index];
            self.end_index = new_end_index;
            return result.ptr;
        }

        const bigger_buf_size = @sizeOf(BufNode) + new_end_index;
        if (allocator.rawResize(cur_alloc_buf, BufNode_alignment, bigger_buf_size, @returnAddress())) {
            cur_node.data = bigger_buf_size;
        } else {
            // Allocate a new node if that's not possible
            cur_node = self.create_node(allocator, cur_buf.len, n + ptr_align) orelse return null;
        }
    }
}
