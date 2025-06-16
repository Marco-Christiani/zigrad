//! Device Caching Allocator:
//! This works like an arena backed buddy-allocator to support
//! block fusion.
//!
//! This data structure's pools will always allocate MAX_POOL_SIZE buffers.
//! To reduce the footprint, it may be valuable to use mmap-style allocations
//! instead eagerly allocating pools upfront.
//!
//! This currently uses fixed sized arrays that will be adjustable
//! via compiler flags, but a dynamic variant is possible.

const std = @import("std");
const Allocator = std.mem.Allocator;
const constants = @import("constants.zig");
const ArenaUnmanaged = @import("arena_unmanaged.zig");
const DeviceData = @import("device_data.zig").DeviceData;

// Block Orders:
//
// Fundamentally, orders are block split sizes that move up
// in powers of 2. The lowest order is 2^MIN_ORDER and the
// highest order is 2^MAX_ORDER.
//
// Orders work based on the contextual requirements of the
// pool. When a requesting memory, the pool will try to find
// the "uppder" order if it's not an exact power of two. This
// is because the we need to split a block that is larger
// because it the allocation is inbetween two orders.
//
// Conversely, when storing blocks, we have to find the
// "lower" order. This is because a block must be able
// to support all allocations at that order and below.
// Thus, it must be greater than or equal to the order
// that it is stored at to facilitate splitting for
// lower order memory requests.
//
// Note that the "MAX_ORDER" is set to 64Gb of memory
// per pool. This does not imply that the pool will
// use that much memory. This allows for block headroom
// to be several gigabytes instead of just one. Also,
// the caching allocator that uses this pooling object
// can spawn more pools if needed.

pub const MAX_ORDER = 36;
pub const MIN_ORDER = 7;
pub const NUM_ORDERS = (MAX_ORDER - MIN_ORDER + 1);
pub const MAX_BLOCK_SPLITS = 100; // TODO: unusec

// Since the number of pools can grow over time, scanning for which
// pool a block belongs to will be slow. Instead, we will use the
// @fieldParentPtr builtin to immediately cast to cast freed
// data back to it's parent block to begin block fusion.

// A key insight here is that we always split blocks to gain a new block.
// Since splits occur withn a block, and a block spans a contiguous
// range of memory, all blocks always have some link to another
// contiguous block. The only caveat is the only block created
// without a prior splitting operation. This is fine though, because
// the top level block never needs to merge with anything. Thus we
// get the following properties:
//
//    1) The left-most block always has prev set to null
//    2) the right-most block always has next set to null
//    3) The top level block is both left-most and right-most
//
// To speed up searching even more, we can make a 2D block web
// that enables registering blocks with an order that matches
// their data size. Thus we get O(1) lookup, and O(1) merge.
// This works because blocks can unlink-themselves without
// deference to the base list.
//
// The left-right orientation will be split-siblings. These are
// siblings that occured when a block was split and thus should
// always point to the next contiguous block over.
//
// The up-down orientation will be order-siblings. These point
// to blocks that are in the same order but not necessarly
// adjacent in memory.
//
// If a block has "ordered" siblings, then it must be in a
// cache (thus it is unused).
const Pool = @This();

pub const Block = struct {
    pool: *Pool,
    data: []u8,
    used: bool = true,
    split: struct {
        next: ?*Block = null,
        prev: ?*Block = null,
    } = .{},
    order: struct {
        next: ?*Block = null,
        prev: ?*Block = null,
    } = .{},
};

// Here to make block freeing more clear. I don't want to
// use existing block fields to avoid overloaded meanings.
const FreeList = struct {
    const empty: FreeList = .{
        .head = null,
    };
    const Node = struct {
        next: ?*Node,
    };
    head: ?*Node,
    pub fn prepend(self: *FreeList, block: *Block) void {
        block.* = undefined;
        const node: *Node = @ptrCast(@alignCast(block));
        node.* = .{ .next = self.head };
        self.head = node;
    }
    pub fn pop(self: *FreeList) ?*Block {
        const head = self.head orelse return null;
        self.head = head.next;
        head.* = undefined;
        return @ptrCast(@alignCast(head));
    }
};

/// Overflow means that the Pool cannot support the requested
/// memory size requested for the incoming allocation.
pub const Error = error{Overflow} || std.mem.Allocator.Error;

/// The mapped buffer that references the
/// the available device data in this pool
mem_buf: []u8,
/// The remaining memory before and after
/// an allocation that a pool could theoretically
/// support (does not account for holes)
mem_rem: usize,
/// Small optimization for preventing scans
/// beyond the higest available order. At
/// most this is MAX_ORDER + 1 and acts like
/// an exclusive end ragne for orders.
order_sentinel: usize = 0,
/// Array of Linked lists where blocks can
/// register themselves as free for reuse
block_orders: [NUM_ORDERS]?*Block = @splat(null),
/// Arena for allocating blocks.
block_arena: ArenaUnmanaged = .empty,
/// Free list for unused blocks. Blocks become
/// unused when free fuses two blocks together
/// and therefore only requires one block.
free_blocks: FreeList = .empty,

// assumes pool is new - setting up an already created pool is UB
pub fn setup(pool: *Pool, data_handler: anytype, allocator: Allocator, config: struct {
    max_pool_size: usize = constants.@"1Gb",
}) !void {
    std.debug.assert(config.max_pool_size > 0);

    // adjust to a multiple of page_sizes
    const adjusted_pool_size = blk: {
        const ps = data_handler.page_size();
        const adjust = (ps - (config.max_pool_size % ps)) % ps;
        break :blk config.max_pool_size + adjust;
    };

    const mem_buf = try data_handler.map(adjusted_pool_size);
    errdefer data_handler.unmap(mem_buf);

    pool.* = .{
        .mem_buf = @ptrCast(@alignCast(mem_buf)),
        .mem_rem = adjusted_pool_size,
        .order_sentinel = @min(size_to_lower_order(adjusted_pool_size) + 1, NUM_ORDERS),
    };

    const block = try pool.create_block(allocator);

    block.* = .{
        .pool = pool,
        .data = mem_buf,
    };

    pool.reserve_block(block);
}

pub fn deinit(self: *Pool, data_handler: anytype, allocator: Allocator) void {
    self.block_arena.deinit(allocator);
    data_handler.unmap(self.mem_buf);
    self.* = undefined;
}

pub fn alloc(self: *Pool, T: type, allocator: Allocator, size: usize) Error!DeviceData(T) {
    const order = size_to_upper_order(size * @sizeOf(T));
    const split_size = order_to_size(order);

    // TODO: This current design doesn't account for holes,
    // but we'd have to get a lot more fancy to deal with that.
    if (self.mem_rem < size)
        return Error.Overflow;

    // scan up to and including the the max reserved order
    const lhs = scan: for (order..self.order_sentinel) |i| {
        const rhs = self.release_block(i) orelse
            continue;

        if (order == i)
            break :scan rhs;

        const lhs = try self.create_block(allocator);

        lhs.* = .{
            .pool = self,
            .data = rhs.data[0..split_size],
            .split = .{
                .prev = rhs.split.prev,
                .next = rhs,
            },
        };

        // since there could be a block already on the left,
        // we're going to insert the new block between them
        // so make sure the previous left side points to this.
        if (rhs.split.prev) |prev|
            prev.split.next = lhs;

        // connect the new left-side block
        rhs.split.prev = lhs;

        // reduce size and reserve.
        rhs.data.ptr += split_size;
        rhs.data.len -= split_size;
        self.reserve_block(rhs);

        break :scan lhs;
    } else return Error.Overflow;

    self.mem_rem -= lhs.data.len;

    return .{
        .raw = @alignCast(std.mem.bytesAsSlice(T, lhs.data)[0..size]),
        .ctx = @intFromPtr(lhs),
    };
}

pub fn free(self: *Pool, data: anytype) void {
    const block: *Block = @ptrFromInt(data.ctx);
    self.mem_rem += block.data.len;

    // fuse the left-side with block
    const fused: *Block = scope: {
        const left = block.split.prev orelse
            break :scope block;

        if (left.used)
            break :scope block;

        self.disconnect_from_order(left);

        // we're going to destroy this block,
        // so connect the right to the left
        if (block.split.next) |right| {
            right.split.prev = left;
        }
        // connect the left side to the right
        left.split.next = block.split.next;

        left.data.len += block.data.len;
        self.free_blocks.prepend(block);
        break :scope left;
    };

    scope: { // fuse the right hand side
        const right = fused.split.next orelse
            break :scope;

        if (right.used)
            break :scope;

        self.disconnect_from_order(right);

        // we're going to destroy the right block,
        // so only take what's to the right of it.
        if (right.split.next) |further_right| {
            further_right.split.prev = fused;
        }
        // connect to the further right block
        fused.split.next = right.split.next;

        fused.data.len += right.data.len;
        self.free_blocks.prepend(right);
    }
    self.reserve_block(fused);
}

// maintains that prev<->block<->block => prev<->next
fn disconnect_from_order(self: *Pool, block: *Block) void {
    if (block.order.prev) |prev| {
        prev.order.next = block.order.next;
    } else {
        // disconnect array from this pointer
        const index = size_to_lower_index(block.data.len);
        self.block_orders[index] = block.order.next;
    }
    if (block.order.next) |next| {
        next.order.prev = block.order.prev;
    }
}

fn create_block(self: *Pool, allocator: Allocator) Allocator.Error!*Block {
    return self.free_blocks.pop() orelse {
        return self.block_arena.create(allocator, Block);
    };
}

fn reserve_block(self: *Pool, block: *Block) void {
    std.debug.assert(block.pool == self);

    // we want the lowest index possible because if a
    // block can no longer support the largest allocation
    // at that order, then it must be demoted down.
    const index = size_to_lower_index(block.data.len);

    if (self.block_orders[index]) |head|
        head.order.prev = block;

    block.order.next = self.block_orders[index];
    block.order.prev = null; // used to detect head of array
    block.used = false;

    self.block_orders[index] = block;
}

fn release_block(self: *Pool, index: usize) ?*Block {
    const block = self.block_orders[index] orelse return null;

    if (block.order.next) |next|
        next.order.prev = null;

    self.block_orders[index] = block.order.next;

    block.order.next = null;
    block.order.prev = null;
    block.used = true;
    return block;
}

///////////////////////
//// Order Helpers ////
///////////////////////

pub fn order_to_size(order: usize) usize {
    std.debug.assert(MIN_ORDER <= order and order <= MAX_ORDER);
    return @as(usize, 1) << @as(u6, @truncate(order));
}

pub fn size_to_upper_order(size: usize) usize {
    var order: usize = MIN_ORDER;
    var bsize: usize = 1 << MIN_ORDER;
    while (bsize < size and order < MAX_ORDER) {
        bsize <<= 1;
        order += 1;
    }
    return order;
}

pub fn size_to_lower_order(size: usize) usize {
    // should this be an inline for? need to benchmark...
    return inline for (MIN_ORDER..MAX_ORDER + 1) |order| {
        const order_size = comptime order_to_size(order);
        if (size < order_size)
            break @max(order - 1, MIN_ORDER);
    } else MAX_ORDER;
}

pub fn size_to_lower_index(size: usize) usize {
    return order_to_index(size_to_lower_order(size));
}

pub fn order_to_index(order: usize) usize {
    std.debug.assert(order >= MIN_ORDER);
    return order - MIN_ORDER;
}

pub fn index_to_order(index: usize) usize {
    std.debug.assert(index < NUM_ORDERS);
    return index + MIN_ORDER;
}

pub fn size_to_upper_index(size: usize) usize {
    return order_to_index(size_to_upper_order(size));
}
