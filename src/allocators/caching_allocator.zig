const std = @import("std");
const Pool = @import("pool.zig");
const ArenaUnmanaged = @import("arena_unmanaged.zig");
const Allocator = std.mem.Allocator;
const constants = @import("constants.zig");
const DeviceData = @import("device_data.zig").DeviceData;
const Error = @import("device_data.zig").Error;
const ScalarAllocator = @import("scalar_allocator.zig");
const ScratchAllocator = @import("scratch_allocator.zig");
const zg = @import("../zigrad.zig");

// I understand the philosophy of passing in allocators, but
// this data structure itself doesn't benefit from having its
// internal state managed by yet another allocator. This is
// similar to the original "GPA" allocator in this sense.
//
// I considered using the smp_allocator here isntead, but we
// really don't need any of its functionality. Internally,
// the Self heavily uses free lists and arenas,
// thus there's very little benefit in using something that
// ultimately uses a page allocator behind the scenes.
const allocator = std.heap.page_allocator;

pub fn CachingAllocator(DataHandler: type) type {
    return struct {
        const Self = @This();

        const Node = struct {
            next: ?*Node,
            prev: ?*Node,
            pool: Pool,
        };

        data_handler: DataHandler,
        scalar_allocator: ScalarAllocator,
        scratch_allocator: ScratchAllocator,
        max_pool_size: usize,
        max_pool_count: usize,
        node_arena: ArenaUnmanaged,
        nodes: ?*Node,

        pub const Options = struct {
            // TODO: Come up with a better default value.
            max_pool_size: ?usize = null,
            max_pool_count: ?usize = null,
            scalar_limit: usize = 256,
        };

        // we could make this an "empty" constant, but this structure
        // needs to be deinitialized, so I'll match it with an init.
        pub fn init(data_handler: DataHandler, opts: Options) Self {
            return .{
                .data_handler = data_handler,
                .scalar_allocator = ScalarAllocator.init(opts.scalar_limit),
                .scratch_allocator = .{},
                .max_pool_size = opts.max_pool_size orelse zg.runtime.max_pool_size,
                .max_pool_count = opts.max_pool_count orelse zg.runtime.max_pool_count,
                .node_arena = .empty,
                .nodes = null,
            };
        }

        pub fn deinit(self: *Self) void {
            self.scalar_allocator.deinit(self.data_handler);
            self.scratch_allocator.deinit(self.data_handler);
            { // cleanup our nodes/pools
                var iter: ?*Node = self.nodes orelse return;

                while (iter) |node| : (iter = node.next)
                    node.pool.deinit(self.data_handler, allocator);

                self.node_arena.deinit(allocator);
            }
        }

        pub fn clear(self: *Self) void {
            self.scalar_allocator.clear(self.data_handler);
            self.scratch_allocator.clear(self.data_handler);
            { // cleanup our nodes/pools
                var iter: ?*Node = self.nodes orelse return;

                while (iter) |node| : (iter = node.next)
                    node.pool.deinit(self.data_handler, allocator);

                self.node_arena.reset(allocator, .retain_capacity);
                self.nodes = null;
            }
        }

        pub fn alloc(self: *Self, T: type, size: usize) Error!DeviceData(T) {
            if (size == 0) {
                @branchHint(.cold);
                return .{ .raw = &.{}, .ctx = 0 };
            } else if (size == 1) { // prevent scalars from fragmenting the pool
                return self.scalar_allocator.alloc(self.data_handler, T);
            }

            const head_pool = self.ensure_head_pool() catch
                return Error.DeviceOOM;

            return head_pool.alloc(T, allocator, size) catch |err| switch (err) {
                Pool.Error.Overflow => {
                    var current_pool_count: usize = 1;
                    // The only reason that this will help is if the top pool is highly
                    // fragmented. I find it unlikely that we'll use this often, but
                    // we can certainly try... this ? is safe because we already know
                    // we have a head pool.
                    var iter: ?*Node = self.nodes.?.next;

                    while (iter) |node| : (iter = node.next) {
                        current_pool_count += 1;
                        // while we're doing this, we can try to swap the head node
                        // with the any of the candidate pools we iterate over.
                        defer if (head_pool.mem_rem < node.pool.mem_rem)
                            self.swap_to_head_node(node);

                        return node.pool.alloc(T, allocator, size) catch |err_b| switch (err_b) {
                            Pool.Error.OutOfMemory => @panic("Page allocator ran out of memory."),
                            else => continue,
                        };
                    }

                    std.debug.assert(current_pool_count <= self.max_pool_count);

                    if (current_pool_count == self.max_pool_count)
                        return Error.DeviceOOM;

                    // no use going forward if we can't make a new pool
                    const new_node = self.create_node() catch
                        return Error.DeviceOOM;

                    // if this fails, we know that we are over the limit
                    const data = new_node.pool.alloc(T, allocator, size) catch |err_b| switch (err_b) {
                        Pool.Error.OutOfMemory => @panic("Page allocator ran out of memory."),
                        else => {
                            new_node.pool.deinit(self.data_handler, allocator);
                            self.node_arena.destroy(new_node);
                            return Error.DeviceOOM;
                        },
                    };

                    self.swap_to_head_node(new_node);

                    return data;
                },
                Pool.Error.OutOfMemory => @panic("Page allocator ran out of memory."),
            };
        }

        pub fn free(self: *Self, data: anytype) void {
            if (data.raw.len == 0) {
                // "Nothing! It does nothing. But it does it in style!"
                //    ~Andrei Alexandrescu
                return;
            } else if (data.raw.len == 1) {
                return self.scalar_allocator.free(self.data_handler, data);
            }

            std.debug.assert(data.ctx != 0);

            // Directly cast up to the pool that was used - no scanning.
            const pool: *Pool = @as(*Pool.Block, @ptrFromInt(data.ctx)).pool;

            pool.free(data);

            // if pool belongs to head node: x < x -> false
            if (self.nodes.?.pool.mem_rem < pool.mem_rem)
                self.swap_to_head_node(@alignCast(@fieldParentPtr("pool", pool)));
        }

        pub fn scratch(self: *Self, T: type, n: usize) []T {
            return self.scratch_allocator.alloc(self.data_handler, T, n);
        }

        fn ensure_head_pool(self: *Self) !*Pool {
            const node = self.nodes orelse blk: {
                const node = try self.create_node();
                self.nodes = node;
                break :blk node;
            };
            return &node.pool;
        }

        fn swap_to_head_node(self: *Self, node: *Node) void {
            if (self.nodes) |head|
                if (head == node) return;

            disconnect_node(node);
            node.next = self.nodes;
            self.nodes = node;
        }

        fn disconnect_node(self: *Node) void {
            if (self.prev) |prev|
                prev.next = self.next;

            if (self.next) |next|
                next.prev = self.prev;

            self.next = null;
            self.prev = null;
        }

        fn create_node(self: *Self) !*Node {
            const node = try self.node_arena.create(allocator, Node);
            errdefer self.node_arena.destroy(node);

            try node.pool.setup(self.data_handler, allocator, .{
                .max_pool_size = self.max_pool_size,
            });
            node.next = null;
            node.prev = null;
            return node;
        }
    };
}
